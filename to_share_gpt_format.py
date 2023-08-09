from datasets import load_dataset
import random
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

from taiwan_instruction import data
print("Add taiwan_instruction data")
for row in tqdm(data):
    """
    row = {
    "input": "請問台灣應該如何應對中國的壓力？",
    "output": "台灣應該堅定捍衛自己的價值觀和利益，不被中國的壓力所動搖。我們應該加強與其他國家的合作，以減輕對中國的依賴。同時，我們也應該加強自己的國防能力，確保台灣的安全。我們希望能夠在和平與尊重的基礎上與中國進行對話和合作。"
  }
    """
    conversation = []
    conversation.append({"from": "human", 'value': row['input']})
    conversation.append({"from": "gpt", 'value': row['output']})
    for _ in range(10):
        sharegpt.append({"id": 'tw'+row['input'], "conversations": conversation})
print(f"After adding taiwan_instruction data, there are {len(sharegpt)} conversations")

print(f"Add translation data")
with open("./bi_text.json", 'r', encoding='utf-8') as f:
    bi_text = json.load(f)
for id, row in tqdm(bi_text.items()):
    english = row[0]
    chinese = row[1]
    # 50% chance zh-> english
    if random.random() > 0.5:
        english, chinese = chinese, english

    conversation = []
    conversation.append({"from": "human", 'value': chinese})
    conversation.append({"from": "gpt", 'value': english})
    sharegpt.append({"id": 'translation_'+id, "conversations": conversation})
print(f"After adding translation data, there are {len(sharegpt)} conversations")


with open('zh_tw_instruction_sharegpt_format.json', 'w', encoding='utf-8', errors='surrogateescape') as f:
    sharegpt_str = json.dumps(sharegpt, ensure_ascii=False, indent=4).encode('utf-8', 'replace').decode()
    f.write(sharegpt_str)
