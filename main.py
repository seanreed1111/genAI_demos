from utilities import *
from pathlib import Path
import gradio as gr

# setup API key
openai_env_path, openai.api_key = None, None
cwd = Path.cwd()
openai_env_path = cwd / "openai.env"
set_open_ai_key(openai_env_path)

#Set up location for pdfs
pdf_uris = Path(cwd, "Prescribing-Info", "accupril_quinapril.pdf")

# actual resumes start on page 2 of this pdf compilation
drug1 = load_pdfs(pdf_uris)

create_index(drug1)
qa = create_conversation()

with gr.Blocks() as demo:
    chatbot = gr.Chatbot([], elem_id="chatbot",
                         label='Resume GPT').style(height=750)
    with gr.Row():
        with gr.Column(scale=0.80):
            txt = gr.Textbox(
                show_label=False,
                placeholder="Enter text and press enter",
            ).style(container=False)
        with gr.Column(scale=0.10):
            submit_btn = gr.Button(
                'Submit',
                variant='primary'
            )
        with gr.Column(scale=0.10):
            clear_btn = gr.Button(
                'Clear',
                variant='stop'
            )

    txt.submit(add_text, [chatbot, txt], [chatbot, txt]).then(
        bot, chatbot, chatbot
    )

    submit_btn.click(add_text, [chatbot, txt], [chatbot, txt]).then(
        bot, chatbot, chatbot
    )

    clear_btn.click(lambda: None, None, chatbot, queue=False)

demo.queue(concurrency_count=3)
demo.launch()
