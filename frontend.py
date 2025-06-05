import gradio as gr
import requests


# default_prompt = 'Find the current temperature in Tokyo, then, respond with a flashcard summarizing this information'
# default_prompt = 'Tell me the current temperature in Seoul, and then tell me the result of adding 3 to that temperature.'
# default_prompt = '지금 한국에서 제일 유명한 영화가 뭔지 찾아서, 제목이랑 평점 알려줘. 그리고 서울의 온도를 찾아서 평점이랑 더한 결과를 알려줘.'
# default_prompt = 'Find out what the most popular movie in Korea is right now, and tell me its title and rating. Also, check the temperature in Seoul, and give me the result of adding the movie’s rating and the temperature.'
default_prompt = 'Get the current temperature of Seoul and send it mail to ionia.kim@samsung.com'


with gr.Blocks() as demo:
    gr.Markdown('# [DEMO] Multi-Agents Orchestration System 🤖')

    with gr.Row():
        chatbot = gr.Chatbot(show_label=False)
    with gr.Row():
        input_textbox = gr.Textbox(show_label=False, placeholder='요청을 입력하세요.', lines=2, value=default_prompt)
    send_button = gr.Button('SEND')

    def _chat(user_message, history):
        history = history or []
        history.append([user_message, ''])
        return default_prompt, history

    def stream_result(history):
        url = 'http://localhost:8000/test'
        data = {"message": history[-1][0]}
        with requests.post(url, json=data, stream=True) as response:
            response.raise_for_status()
            for chunk in response.iter_content(chunk_size=None):
                if chunk:
                    history[-1][1] = chunk.decode('utf-8')
                    history.append([None, ''])
                    yield history
        if history[-1][0] is None:
            history = history[:-1]
        yield history

    send_button.click(
        _chat, inputs=[input_textbox, chatbot], outputs=[input_textbox, chatbot]
    ).then(
        stream_result, inputs=chatbot, outputs=chatbot
    )


if __name__ == "__main__":
    demo.launch()
