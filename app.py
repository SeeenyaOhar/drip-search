import logging
import streamlit as st
from streamlit.logger import get_logger
import time
import os

from chat import Chat
from models.chunker import DefaultChunker
from models.combined_retriever import CombinedRetriever
from models.groqllm import GroqModel
from models.keyword_retriever import KeywordRetriever
from models.reranker import CrossEncoderReranker
from models.semantic_retriever import SemanticRetriever
from models.document import Document
from dotenv import load_dotenv

load_dotenv()
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
CHUNK_DIR = os.path.join(DATA_DIR, 'chunks')
logger = get_logger(__file__)
logger.setLevel(logging.DEBUG)

def response_generator(prompt: str, chat: Chat):
    response = chat.answer_question(prompt)
    for word in response.split():
        yield word + " "
        time.sleep(0.05)
def get_docs():
    documents = []
    
    for item in sorted(os.listdir(DATA_DIR)):
        file_path = os.path.join(DATA_DIR, item)

        if os.path.isfile(file_path) and item.endswith('.txt'):
            with open(file_path, 'r') as file:
                content = file.read()
            
            document = Document(content, id=item)
            documents.append(document)
    return documents

def chat_config() -> dict:
    docs = get_docs()
    retriever_mode = os.environ.get("RETRIEVER_MODE")

    config = {
        "retriever": CombinedRetriever(docs,
                                       SemanticRetriever(docs=docs,
                                                         logger=logger,
                                                         precalc=True) if (retriever_mode == "combined") or (retriever_mode == "semantic") else None,
                                       KeywordRetriever(logger=logger, docs=docs) if (retriever_mode == "combined") or (retriever_mode == "keyword") else None),
        "llm": GroqModel(logger=logger),
        "docs": docs,
        "chunker": DefaultChunker(logger=logger, 
                                  save_on_chunk=True,
                                  chunk_save_dir=CHUNK_DIR),
        "reranker": CrossEncoderReranker(logger=logger),
        "logger": logger,
        "prechunk": True
    }
    return config

if 'chat' not in st.session_state:
    st.session_state['chat'] = Chat(**chat_config())
st.title("Drip Chat")

if retriever_mode := os.environ.get("RETRIEVER_MODE"):
    st.markdown(f"**Retriever Mode:** {retriever_mode.capitalize()}")
else:
    st.markdown("**Retriever Mode:** Not Set")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = st.write_stream(response_generator(prompt, st.session_state['chat']))
    st.session_state.messages.append({"role": "assistant", "content": response})
