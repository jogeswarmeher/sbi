import streamlit as st
from pathlib import Path
from typing import List
from PIL import Image
import pypdfium2
from surya.models import load_predictors
import os
import requests
import tempfile
import hashlib
import torch
import shutil

# ---------------- LangChain imports ----------------
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# ---------------- CONFIG ----------------
OLLAMA_BASE_URL = "http://localhost:8890/v1"
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")
OLLAMA_MODEL = "meta/llama-3.1-8b-instruct"

st.set_page_config(page_title="💬 Chat with PDF/Image (Surya OCR + Ollama)", layout="wide")
st.title("💬 Chat with your PDF or Image (Surya OCR + Ollama LLM)")

# ---------------- SESSION STATE ----------------
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "last_answer" not in st.session_state:
    st.session_state.last_answer = None

if "current_file_hash" not in st.session_state:
    st.session_state.current_file_hash = None

# ---------------- DEVICE AUTO DETECTION ----------------
def get_device():
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"

DEVICE = get_device()
st.sidebar.write(f"🖥 Using device: **{DEVICE}**")

# ---------------- Load Surya Once ----------------
@st.cache_resource
def load_surya(device):
    return load_predictors(device=device)

predictors = load_surya(DEVICE)

# ---------------- PDF → Images ----------------
def pdf_to_images(pdf_path: str, dpi: int = 300) -> List[Image.Image]:
    images = []
    doc = pypdfium2.PdfDocument(str(pdf_path))

    try:
        for page_idx in range(len(doc)):
            renderer = doc.render(
                pypdfium2.PdfBitmap.to_pil,
                page_indices=[page_idx],
                scale=dpi / 72
            )
            images.append(list(renderer)[0].convert("RGB"))
    finally:
        doc.close()

    return images

# ---------------- OCR For Images List ----------------
def ocr_images(images: List[Image.Image]) -> List[str]:
    recognitions = predictors["recognition"](
        images=images,
        task_names=["ocr_with_boxes"] * len(images),
        det_predictor=predictors["detection"]
    )

    texts = []
    for rec in recognitions:
        page_text = " ".join([line.text for line in rec.text_lines])
        texts.append(page_text.strip())

    return texts

# ---------------- Vector Store ----------------
@st.cache_resource
def build_vectorstore(text_pages: List[str], persist_dir: str):

    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    docs = []

    for page_num, page_text in enumerate(text_pages, start=1):
        for chunk in splitter.split_text(page_text):
            docs.append(
                Document(
                    page_content=chunk,
                    metadata={"page": page_num}
                )
            )

    embeddings = HuggingFaceEmbeddings(
        #model_name="sentence-transformers/all-MiniLM-L6-v2"intfloat/multilingual-e5-base
        model_name="intfloat/multilingual-e5-base",
        model_kwargs={"device": "cuda"}
    )

    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_dir
    )

    return vectorstore

# ---------------- Ollama LLM ----------------
def ask_llama(prompt: str) -> str:
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
                "content": (
                    "Use ONLY the provided context to answer. "
                    "If not found, say 'Not found in the document.' "
                    "Cite page numbers."
                )
            },
            {"role": "user", "content": prompt}
        ]
    }

    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

# ---------------- Query ----------------
def query_document(vectorstore, question: str, top_k: int = 4) -> str:
    docs = vectorstore.similarity_search(question, k=top_k)

    context = "\n\n".join(
        [
            f"(Page {d.metadata.get('page', 'Unknown')})\n{d.page_content}"
            for d in docs
        ]
    )

    prompt = f"""
Answer strictly in Hindi.    
Context:
{context}

Question:
{question}

Answer clearly and cite page numbers if possible.
"""

    return ask_llama(prompt)

# ---------------- File Upload ----------------
uploaded_file = st.file_uploader(
    "Upload PDF or Image",
    type=["pdf", "jpg", "jpeg", "png"]
)

if uploaded_file:
    file_bytes = uploaded_file.read()
    file_hash = hashlib.md5(file_bytes).hexdigest()
    uploaded_file.seek(0)

    # Process only if new file
    if st.session_state.current_file_hash != file_hash:
        st.session_state.current_file_hash = file_hash
        st.session_state.vectorstore = None
        st.session_state.last_answer = None

        suffix = Path(uploaded_file.name).suffix.lower()

        with st.spinner("🔍 Processing file with Surya OCR..."):

            if suffix == ".pdf":
                # Save PDF temporarily
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                tmp.write(file_bytes)
                tmp.close()

                images = pdf_to_images(tmp.name)
                text_pages = ocr_images(images)
                os.remove(tmp.name)

            else:
                # Image file
                image = Image.open(uploaded_file).convert("RGB")
                text_pages = ocr_images([image])

        persist_dir = f"./chroma_db_{file_hash}"

        with st.spinner("📚 Building vector database..."):
            st.session_state.vectorstore = build_vectorstore(
                text_pages,
                persist_dir
            )

        st.success("✅ File processed successfully!")

# ---------------- Chat Interface ----------------
if st.session_state.vectorstore:

    if st.session_state.last_answer:
        st.markdown("### 📌 Answer")
        st.write(st.session_state.last_answer)

    def handle_question():
        question = st.session_state.question_input

        if question.strip():
            with st.spinner("🤖 Thinking..."):
                answer = query_document(
                    st.session_state.vectorstore,
                    question
                )

            st.session_state.last_answer = answer
            st.session_state.question_input = ""

    st.text_input(
        "Ask something about the document:",
        key="question_input",
        on_change=handle_question
    )
