import os

import gradio as gr
from text_generation import Client
# text-generation 0.6.0

eos_token = "</s>"

def _concat_messages(messages):
    message_text = ""
    for message in messages:
        if message["role"] == "system":
            message_text += "<|system|>\n" + message["content"].strip() + "\n"
        elif message["role"] == "user":
            message_text += "<|user|>\n" + message["content"].strip() + "\n"
        elif message["role"] == "assistant":
            message_text += "<|assistant|>\n" + message["content"].strip() + eos_token + "\n"
        else:
            raise ValueError("Invalid role: {}".format(message["role"]))
    return message_text

endpoint_url = os.environ.get("ENDPOINT_URL")
client = Client(endpoint_url, timeout=120)

def generate_response(user_input, max_new_token: 100, top_p, temperature, top_k, do_sample, repetition_penalty):
    msg = _concat_messages([
        {"role": "system", "content": "你是一個由國立台灣大學的NLP實驗室開發的大型語言模型。你基於Transformer架構被訓練，並已經經過大量的台灣中文語料庫的訓練。你的設計目標是理解和生成優雅的繁體中文，並具有跨語境和跨領域的對話能力。使用者可以向你提問任何問題或提出任何話題，並期待從你那裡得到高質量的回答。你應該要盡量幫助使用者解決問題，提供他們需要的資訊，並在適當時候給予建議。"},
        {"role": "user", "content": user_input},
    ])
    msg += "<|assistant|>\n"
    res = client.generate(msg, stop_sequences=["<|assistant|>", eos_token, "<|system|>", "<|user|>"],
                          max_new_tokens=1000, top_p=0.9)
    return [("assistant", res.generated_text)]

with gr.Blocks() as demo:
    # github_banner_path = 'https://raw.githubusercontent.com/ymcui/Chinese-LLaMA-Alpaca/main/pics/banner.png'
    # gr.HTML(f'<p align="center"><a href="https://github.com/ymcui/Chinese-LLaMA-Alpaca"><img src={github_banner_path} width="700"/></a></p>')
    # gr.Markdown("> 为了促进大模型在中文NLP社区的开放研究，本项目开源了中文LLaMA模型和指令精调的Alpaca大模型。这些模型在原版LLaMA的基础上扩充了中文词表并使用了中文数据进行二次预训练，进一步提升了中文基础语义理解能力。同时，中文Alpaca模型进一步使用了中文指令数据进行精调，显著提升了模型对指令的理解和执行能力。")
    chatbot = gr.Chatbot()
    with gr.Row():
        with gr.Column(scale=4):
            with gr.Column(scale=12):
                user_input = gr.Textbox(
                    show_label=False,
                    placeholder="Shift + Enter傳送...",
                    lines=10).style(
                    container=False)
            with gr.Column(min_width=32, scale=1):
                submitBtn = gr.Button("Submit", variant="primary")
        with gr.Column(scale=1):
            emptyBtn = gr.Button("Clear History")
            max_new_token = gr.Slider(
                0,
                4096,
                value=512,
                step=1.0,
                label="Maximum New Token Length",
                interactive=True)
            top_p = gr.Slider(0, 1, value=0.9, step=0.01,
                              label="Top P", interactive=True)
            temperature = gr.Slider(
                0,
                1,
                value=0.5,
                step=0.01,
                label="Temperature",
                interactive=True)
            top_k = gr.Slider(1, 40, value=40, step=1,
                              label="Top K", interactive=True)
            do_sample = gr.Checkbox(
                value=True,
                label="Do Sample",
                info="use random sample strategy",
                interactive=True)
            repetition_penalty = gr.Slider(
                1.0,
                3.0,
                value=1.1,
                step=0.1,
                label="Repetition Penalty",
                interactive=True)

    params = [user_input, chatbot]
    predict_params = [
        chatbot,
        max_new_token,
        top_p,
        temperature,
        top_k,
        do_sample,
        repetition_penalty]

    submitBtn.click(
        generate_response,
        [user_input],
        [chatbot],
        queue=False).then(
        None,
        None,
        [user_input],
        queue=False)

    user_input.submit(
        generate_response,
        [user_input],
        [chatbot],
        queue=False).then(
        None,
        None,
        [user_input],
        queue=False)

    submitBtn.click(lambda: None, [], [user_input])

    emptyBtn.click(lambda: chatbot.reset(), outputs=[chatbot], show_progress=True)

demo.launch(share=True)
