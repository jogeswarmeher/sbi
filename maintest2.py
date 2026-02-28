# pdf_chat_surya_optimized.py

import streamlit as st
from pathlib import Path
from typing import List
from PIL import Image
import pypdfium2
from surya.models import load_predictors
import torch
import os
import requests
import tempfile
import hashlib

# ---------------- LangChain ----------------
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# ---------------- CONFIG ----------------
OLLAMA_BASE_URL = "http://localhost:8890/v1"
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")  # set your API key in env
OLLAMA_MODEL = "meta/llama-3.1-8b-instruct"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

st.set_page_config(page_title="💬 Enterprise Document Intellegence ", layout="wide")
st.title("💬 Chat with your PDF (Surya OCR + Ollama LLM)")

# ---------------- DEVICE AUTO DETECT ----------------
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# ---------------- SESSION STATE ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "predictors" not in st.session_state:
    st.session_state.predictors = load_predictors(device=DEVICE)

# ---------------- PDF → IMAGES ----------------
def pdf_to_images(pdf_file_path: str, dpi: int = 300) -> List[Image.Image]:
    images = []
    doc = pypdfium2.PdfDocument(pdf_file_path)

    for page_idx in range(len(doc)):
        renderer = doc.render(
            pypdfium2.PdfBitmap.to_pil,
            page_indices=[page_idx],
            scale=dpi / 72
        )
        images.append(list(renderer)[0].convert("RGB"))

    doc.close()
    return images

# ---------------- OCR ----------------
def extract_text_from_pdf(pdf_file_path: str) -> List[str]:
    images = pdf_to_images(pdf_file_path)
    predictors = st.session_state.predictors

    recognitions = predictors["recognition"](
        images=images,
        task_names=["ocr"] * len(images)
    )

    texts = []
    for rec in recognitions:
        page_text = "\n".join([line.text for line in rec.text_lines])
        texts.append(page_text.strip())

    return texts

# ---------------- BUILD VECTORSTORE ----------------
@st.cache_resource
def build_vectorstore(text_pages: List[str], persist_dir: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    docs = []
    for page_num, page_text in enumerate(text_pages):
        for chunk in splitter.split_text(page_text):
            docs.append(
                Document(
                    page_content=chunk,
                    metadata={"page": page_num + 1}
                )
            )

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_dir
    )

    return vectorstore

# ---------------- OLLAMA CALL ----------------
def ask_llama(prompt: str):
    url = f"{OLLAMA_BASE_URL}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OLLAMA_API_KEY}"
    }
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant. Use ONLY the provided context."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    }

    response = requests.post(url, json=payload)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

# ---------------- QUERY FUNCTION ----------------
def query_pdf(question: str):
    retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 4})
    docs = retriever.get_relevant_documents(question)

    context = "\n\n".join(
        [f"(Page {d.metadata['page']})\n{d.page_content}" for d in docs]
    )

    prompt = f"""
Context:
{context}

Question:
{question}

Answer clearly and cite page numbers if possible.
"""

    return ask_llama(prompt)

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:
    file_hash = hashlib.md5(uploaded_file.read()).hexdigest()
    uploaded_file.seek(0)

    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    tmp_file.write(uploaded_file.read())
    tmp_file.close()

    pdf_path = tmp_file.name
    persist_dir = f"./chroma_db_{file_hash}"

    with st.spinner("🔍 Extracting text using Surya OCR..."):
        extracted_pages = extract_text_from_pdf(pdf_path)

    with st.spinner("📚 Building vector database..."):
        st.session_state.vectorstore = build_vectorstore(extracted_pages, persist_dir)

    st.success("✅ PDF processed successfully!")
    os.remove(pdf_path)

# ---------------- CHAT UI ----------------
if st.session_state.vectorstore:

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask something about the PDF...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = query_pdf(user_input)
                st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})
