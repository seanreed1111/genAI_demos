import os
import argparse, json, time, datetime, openai
from pathlib import Path
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS
from langchain.vectorstores import Chroma

import gradio as gr

def set_open_ai_key(env_path=None):
  #import json, os
  #from pathlib import Path
  try:
    with open(env_path, "r") as f:
        env_vars = json.load(f)
    os.environ["OPENAI_API_KEY"] = env_vars["OPENAI_API_KEY"]
    openai.api_key = os.environ["OPENAI_API_KEY"]
    #os.environ['OPENAI_API_KEY'] = getpass.getpass('OpenAI API Key:')
    openai.Model.list() #test a random command on the openai API
    return True
  except Exception as e:
    print(e)
  return False


def load_pdfs(path, skip_pages=0):
  loader = PyPDFLoader(path)
  pages = loader.load_and_split()
  pages_clean = pages[skip_pages:]
  return pages_clean

def create_index(resumes) -> None:

    # text_splitter = RecursiveCharacterTextSplitter(
    #     chunk_size = 500,
    #     chunk_overlap  = 10,
    #     length_function = len,
    # )

    text_splitter = CharacterTextSplitter(
        separator='\n',
        chunk_size=500,
        chunk_overlap=10
    )

    texts = text_splitter.split_documents(resumes)

    # embeddings = OpenAIEmbeddings(
    #     openai_api_key=os.getenv('OPENAI_API_KEY')
    # )

    embeddings = OpenAIEmbeddings()

    #db = Chroma.from_documents(texts, embeddings)

    persist_directory = 'db'

    vectordb = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory=persist_directory
    )

    vectordb.persist()


def create_conversation() -> ConversationalRetrievalChain:

    persist_directory = 'db'

    embeddings = OpenAIEmbeddings(
        openai_api_key=os.getenv('OPENAI_API_KEY')
    )

    db = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )

    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=False
    )

    #add initial prompt to system - use prompt template

    qa = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(),
        chain_type='stuff',
        retriever=db.as_retriever(),
        memory=memory,
        get_chat_history=lambda h: h,
        verbose=True
    )

    return qa

def add_text(history, text):
    history = history + [(text, None)]
    return history, ""


def bot(history):
    res = qa(
        {
            'question': history[-1][0],
            'chat_history': history[:-1]
        }
    )
    history[-1][1] = res['answer']
    return history


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
