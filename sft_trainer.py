# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
from dataclasses import dataclass, field
from typing import Optional
from functools import partial

import torch
from datasets import load_dataset, concatenate_datasets
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, HfArgumentParser, TrainingArguments, AutoTokenizer

# from safe_save_trainer import SFT_SafeSaveTrainer
from trl import SFTTrainer as Trainer
from llama_flash_attn_monkey_patch import (
    replace_llama_attn_with_flash_attn
)


replace_llama_attn_with_flash_attn()


def encode_with_messages_format(example, eos_token):
    def _concat_messages(messages):
        message_text = ""
        for message in messages:
            role = message.get("role", None) or message.get("from", None)
            content = message.get("content", message.get("value", "")).strip()
            message["role"] = role
            message["content"] = content
            if role in {"system"}:
                message_text += "<|system|>\n" + content + "\n"
            elif role in {"user", 'human', '使用者'}:
                message_text += "<|user|>\n" + content + "\n"
            elif role in {'assistant', "助理", "助教", '助手', 'gpt', 'openai'}:
                message_text += "<|assistant|>\n" + content + eos_token + "\n"
            else:
                raise ValueError("Invalid role: {}".format(role))
        return message_text.strip()

    ret = []
    messages = example['zh_messages']
    for zh_message in messages:
        zh_message = json.loads(zh_message)
        example_text = _concat_messages(zh_message)
        ret.append(example_text)
    return ret

tqdm.pandas()


# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with SFTTrainer
    """

    model_name: Optional[str] = field(default="meta-llama/Llama-2-7b-hf", metadata={"help": "the model name"})
    dataset_name: Optional[str] = field(
        default="timdettmers/openassistant-guanaco", metadata={"help": "the dataset name"}
    )
    dataset_text_field: Optional[str] = field(default="text", metadata={"help": "the text field of the dataset"})
    log_with: Optional[str] = field(default='all', metadata={"help": "use 'wandb' to log with wandb"})
    learning_rate: Optional[float] = field(default=3e-4, metadata={"help": "the learning rate"})
    batch_size: Optional[int] = field(default=1, metadata={"help": "the batch size"})
    seq_length: Optional[int] = field(default=4096, metadata={"help": "Input sequence length"})
    gradient_accumulation_steps: Optional[int] = field(
        default=1, metadata={"help": "the number of gradient accumulation steps"}
    )
    load_in_8bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 8 bits precision"})
    load_in_4bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 4 bits precision"})
    use_peft: Optional[bool] = field(default=False, metadata={"help": "Wether to use PEFT or not to train adapters"})
    trust_remote_code: Optional[bool] = field(default=True, metadata={"help": "Enable `trust_remote_code`"})
    output_dir: Optional[str] = field(default="output", metadata={"help": "the output directory"})
    peft_lora_r: Optional[int] = field(default=64, metadata={"help": "the r parameter of the LoRA adapters"})
    peft_lora_alpha: Optional[int] = field(default=16, metadata={"help": "the alpha parameter of the LoRA adapters"})
    logging_steps: Optional[int] = field(default=1, metadata={"help": "the number of logging steps"})
    use_auth_token: Optional[bool] = field(default=True, metadata={"help": "Use HF auth token to access the model"})
    num_train_epochs: Optional[int] = field(default=1, metadata={"help": "the number of training epochs"})
    max_steps: Optional[int] = field(default=-1, metadata={"help": "the number of training steps"})
    debug: Optional[bool] = field(default=False, metadata={"help": "Enable debug mode"})


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

# Step 1: Load the model
if script_args.load_in_8bit and script_args.load_in_4bit:
    raise ValueError("You can't load the model in 8 bits and 4 bits at the same time")
elif script_args.load_in_8bit or script_args.load_in_4bit:
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=script_args.load_in_8bit, load_in_4bit=script_args.load_in_4bit
    )
    # This means: fit the entire model on the GPU:0
    device_map = {"": 0}
    torch_dtype = torch.bfloat16
else:
    device_map = None
    quantization_config = None
    torch_dtype = None

model = AutoModelForCausalLM.from_pretrained(
    script_args.model_name,
    quantization_config=quantization_config,
    device_map=device_map,
    trust_remote_code=script_args.trust_remote_code,
    torch_dtype=torch_dtype,
    use_auth_token=script_args.use_auth_token,
)
model.config.use_cache = False

tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    model_max_length=script_args.seq_length,
    padding_side="right",
    use_fast=False,
)
tokenizer.pad_token = tokenizer.eos_token

formatting_func = partial(encode_with_messages_format, eos_token=tokenizer.eos_token)

# Step 2: Load the dataset
dataset = load_dataset("yentinglin/zh_instruction", split='train')
if script_args.debug:
    dataset = dataset.shuffle(seed=42)  # Shuffle the dataset
    dataset = dataset.select(range(int(len(dataset) * 0.0001)))  # Select the first 10%

# Step 3: Define the training arguments
training_args = TrainingArguments(
    output_dir=script_args.output_dir,
    per_device_train_batch_size=script_args.batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    learning_rate=script_args.learning_rate,
    logging_steps=script_args.logging_steps,
    fsdp=["full_shard", "auto_wrap"],
    fsdp_transformer_layer_cls_to_wrap="LlamaDecoderLayer",
    bf16=True,
    tf32=True,
    # gradient_checkpointing=True,
    dataloader_num_workers=1 if script_args.debug else 96,
    save_strategy='steps',
    save_steps=5 if script_args.debug else 10000,
    save_total_limit=1,
    report_to=script_args.log_with,
    push_to_hub=True,
    hub_strategy="checkpoint",
    hub_private_repo=True,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    weight_decay=0.0,
    num_train_epochs=script_args.num_train_epochs,
    max_steps=script_args.max_steps,
    ddp_timeout=18000,
)

# Step 4: Define the LoraConfig
if script_args.use_peft:
    peft_config = LoraConfig(
        r=script_args.peft_lora_r,
        lora_alpha=script_args.peft_lora_alpha,
        bias="none",
        task_type="CAUSAL_LM",
    )
else:
    peft_config = None

# Step 5: Define the Trainer
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=dataset,
    peft_config=peft_config,
    max_seq_length=script_args.seq_length,
    formatting_func=formatting_func,
    # TODO: DataCollatorForCompletionOnlyLM in trl v0.5.0
)

trainer.train()

trainer.save_model(script_args.output_dir)
trainer.push_to_hub()
