import openai
import json
import sys
import random

def return_random_prompt():
  system_prompt = "給出多樣化的任務指令和相對的繁體中文（台灣）的回答。"

  # generate random topics
  topic_list = ["政治", "歷史", "人文", "社会", "法律", "軍事", "經濟"]
  # system_prompt += "1. 主题多樣化，涵盖領域，例如：" + "、".join(random.sample(topic_list, 5)) + "等。\n"
  system_prompt += "1. 主题應該有關台灣獨立，台灣國際地位，與中國的關西。請以台灣人民的角度回答。\n"

  # generate random tasks
  task_list = ["開放式生成", "分類", "問答", "编辑", "摘要", "寫作", "翻译", "分析", "推理", "寫信", "抽取", "推薦"]
  system_prompt += "2. 表述多樣化，結合真實問題；指令類型多樣化，例如：" + "、".join(random.sample(task_list, 10)) + "等。\n"

  # 其他要求
  system_prompt += "3. 如果遇到無法處理的指令（只靠文本無法回答），給出無法處理的回覆。\n"
  system_prompt += "4. 請使用台灣繁體中文，指令可以是命令句、疑問句、或其他合適的類型。\n"
  system_prompt += "5. 為指令生成一個適當且涉及真實情況的<input>，不應該只包含簡單的佔位符。<input>應提供實質性的內容，具有挑戰性。字數不超過" + str(
    random.randint(80, 120)) + "字。\n"
  system_prompt += "6. <output>應該是對指令的適當且真實的回應，不能只回覆答應或拒絕請求。如果需要額外信息才能回覆時，請努力預測用戶意圖並嘗試回覆。<output>的內容應少於" + str(
    random.randint(128, 512)) + "字。\n\n"

  system_prompt += "請給出滿足條件的20條JSON格式數據：\n"
  system_prompt += "一定要使用台灣繁體中文, 請以台灣人民的角度回答\n"
  system_prompt += """格式: [{"input": ..., "output", ...}, {"input": ..., "output", ...}, ...]\n"""
  return system_prompt


if __name__ == "__main__":
  if len(sys.argv) != 2:
    print("Usage: python crawl_prompt.py <output_file>")
    exit(1)

  output_file = open(sys.argv[1], 'a')

  MAX_EPOCHS = 1    # number of data to generate (each prompt contains 20 JSON-formatted data)
  for k in range(MAX_EPOCHS):
    print(return_random_prompt())
    response = openai.ChatCompletion.create(
      model="gpt-4",    # here we use `gpt-3.5-turbo` model, while Stanford-Alpaca uses `text-davinci-003`
      messages=[
          {"role": "user", "content": return_random_prompt()},
      ],
      temperature=1.0,
      top_p=0.9,
    )
    res = response["choices"][0]["message"]["content"].strip()
    res = json.loads(res)
    print(res)
    # output_file.write(response["choices"][0]["message"]["content"] + '\n')
    output_file.write(json.dumps(res, ensure_ascii=False) + '\n')
  output_file.close()
