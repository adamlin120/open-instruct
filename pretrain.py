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
from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import load_dataset, concatenate_datasets
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, HfArgumentParser, TrainingArguments, AutoTokenizer
from trl import SFTTrainer as Trainer
from llama_flash_attn_monkey_patch import (
    replace_llama_attn_with_flash_attn
)


replace_llama_attn_with_flash_attn()


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


def main():
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
        torch_dtype=torch.bfloat16,
        use_auth_token=script_args.use_auth_token,
    )
    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(
        script_args.model_name,
        model_max_length=script_args.seq_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.eos_token

    # Step 2: Load the dataset
    ptt = load_dataset("yentinglin/ptt-corpus", split='train')
    ptt = ptt.remove_columns([c for c in ptt.column_names if c != 'text'])

    dcard = load_dataset("yentinglin/dcard-corpus", split='train')
    dcard = dcard.remove_columns([c for c in dcard.column_names if c != 'text'])

    mag = load_dataset("yentinglin/tw-magazine", split='train')
    mag = mag.remove_columns([c for c in mag.column_names if c != 'text'])

    zh_c4 = load_dataset("yentinglin/zh_TW_c4", split='train')
    zh_c4 = zh_c4.remove_columns([c for c in zh_c4.column_names if c != 'text'])

    zh_wiki = load_dataset("yentinglin/zh_wiki", split='train')
    zh_wiki = zh_wiki.remove_columns([c for c in zh_wiki.column_names if c != 'text'])

    tw_news = load_dataset("yentinglin/tw_news", split='train')
    tw_news = tw_news.remove_columns([c for c in tw_news.column_names if c != 'text'])

    dataset = concatenate_datasets([ptt, dcard, mag, zh_c4, zh_wiki, tw_news])
    dataset = dataset.shuffle(seed=42)  # Shuffle the dataset
    if script_args.debug:
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
        save_steps=5 if script_args.debug else 100_000,
        save_total_limit=1,
        report_to=script_args.log_with,
        # push_to_hub=True,
        # hub_strategy="checkpoint",
        hub_private_repo=True,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.0,
        num_train_epochs=script_args.num_train_epochs,
        max_steps=script_args.max_steps,
        ddp_timeout=18000,
        gradient_checkpointing=True,
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
        dataset_text_field=script_args.dataset_text_field,
        peft_config=peft_config,
        max_seq_length=script_args.seq_length,
        packing=True
    )

    trainer.train()

    trainer.save_model(script_args.output_dir)
    trainer.push_to_hub()

if __name__ == "__main__":
    main()
