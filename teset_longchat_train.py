import json
from typing import Dict

import transformers
from transformers.trainer_pt_utils import LabelSmoother

from conversation import get_default_conv_template, SeparatorStyle

IGNORE_TOKEN_ID = LabelSmoother.ignore_index
HUGGING_FACE_TOKEN = "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"
rank0_print = print


def preprocess(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """ The template for training data is:
        conv_vicuna_v1_1 = Conversation(
        system="A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions.",
        roles=("USER", "ASSISTANT"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.TWO,
        sep=" ",
        sep2="</s>",
    )
    """
    conv = get_default_conv_template("vicuna").copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]} # map human to USER and gpt to ASSISTANT

    # Apply prompt templates
    conversations = []
    """
    sources = {
                "id": "identity_0",
                "conversations": [
                    {
                    "from": "human",
                    "value": "Who are you?"
                    },
                    {
                    "from": "gpt",
                    "value": "I am Vicuna, a language model trained by researchers from Large Model Systems Organization (LMSYS)."
                    },
                    {
                    "from": "human",
                    "value": "What can you do?"
                    },
                    {
                    "from": "gpt",
                    "value": "I can chat with you."
                    }
                ]
            }
    """
    misaligned_count = 0
    for i, source in enumerate(sources):
        #if source[0]["from"] not in roles.keys() or roles[source[0]["from"]] != conv.roles[0]:
        if  roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        skipped = 0
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            if role != conv.roles[(j - skipped) % 2]:
                print("skipping misaligned rounds")
                skipped += 1
                misaligned_count += 1
                print(f"Misaligned count: {misaligned_count}")
                continue # skipp if two rounds are from the user or two round are from assistant
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    print("Start tokenizing")
    # Tokenize conversations
    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    targets = input_ids.clone()
    import ipdb
    ipdb.set_trace()
    print("Finish tokenizing")

    assert conv.sep_style == SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            round_len = len(tokenizer(rou).input_ids)
            instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_TOKEN_ID # mask the turns of human in the target
            cur_len += round_len
        target[cur_len:] = IGNORE_TOKEN_ID

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                rank0_print(
                    f"WARNING: tokenization mismatch " f"{cur_len} vs. {total_len}"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )


# tokenizer = transformers.AutoTokenizer.from_pretrained("meta-llama/Llama-2-13b-hf", use_fast=False)
tokenizer = transformers.AutoTokenizer.from_pretrained(
        "meta-llama/Llama-2-13b-hf",
        model_max_length=1024,
        padding_side="right",
        use_fast=False,
        # use_auth_token=HUGGING_FACE_TOKEN
    )
tokenizer.pad_token = tokenizer.unk_token
data_path = "./zh_tw_instruction_sharegpt_format.json"
list_data_dict = json.load(open(data_path, "r"))
sources = [example["conversations"] for example in list_data_dict]
preprocess(sources, tokenizer)
