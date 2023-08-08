from datasets import load_dataset
import json
from tqdm import tqdm

dataset = load_dataset("yentinglin/zh_instruction")
sharegpt = []

for row in tqdm(dataset['train']):
    conversation = []
    for turn in json.loads(row['zh_messages']):
        if 'role' in turn:
            if turn['role'] in {"assistant", "助手", "助理", "助教", 'openai'}:
                role = 'gpt'
            elif turn['role'] in {"user", '使用者'}:
                role = 'human'
            else:
                print('role')
                print(turn['role'])
            conversation.append({"from": role, 'value': turn['content']})
        elif 'from' in turn:
            if turn['from'] in {"gpt", 'assistant'}:
                role = 'gpt'
            elif turn['from'] in {"human"}:
                role = 'human'
            else:
                print('from')
                print(turn['from'])
            # conversation.append(turn)
            conversation.append({"from": role, 'value': turn['value']})
    sharegpt.append({"id": row['id'], "conversations": conversation})


""" input data format: list[dict]
        [
            {
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
            },
            {
            "id": "identity_1",
            "conversations": [
                {
                "from": "human",
                "value": "Who are you?"
                },
                {
                "from": "gpt",
                "value": "My name is Vicuna, and I'm a language model developed by Large Model Systems Organization (LMSYS)."
                }
            ]
            }
        ]
        """
# Make sure sharegpt is in the same format as the above example
for row in tqdm(sharegpt):
    if 'id' not in row:
        print('id')
        print(row)
    if 'conversations' not in row:
        print('conversations')
        print(row)
    for turn in row['conversations']:
        if 'from' not in turn:
            print('from')
            print(turn)
        if 'value' not in turn:
            print('value')
            print(turn)
import codecs
#make sure two keywords "林彥廷", "Miulab" are not in the "value"
print(f"Before filtering, there are {len(sharegpt)} conversations")
new_sharegpt = []
for row in tqdm(sharegpt):
    for turn in row['conversations']:
        if '林彥廷' in turn['value'] or 'Miulab' in turn['value']:
            break
    else:
        new_sharegpt.append(row)
sharegpt = new_sharegpt
print(f"After filtering, there are {len(sharegpt)} conversations")


# sharegpt_str = json.dumps(sharegpt, ensure_ascii=False, indent=4)
#
# with codecs.open('zh_tw_instruction_sharegpt_format.json', 'w', encoding='utf-8') as f:
#     f.write(sharegpt_str)
#
with open('zh_tw_instruction_sharegpt_format.json', 'w', encoding='utf-8', errors='surrogateescape') as f:
    sharegpt_str = json.dumps(sharegpt, ensure_ascii=False, indent=4).encode('utf-8', 'replace').decode()
    f.write(sharegpt_str)

# with open('zh_tw_instruction_sharegpt_format.json', 'w',encoding='utf-8', errors='surrogateescape') as f:
#     json.dump(sharegpt, f, ensure_ascii=False, indent=4)
