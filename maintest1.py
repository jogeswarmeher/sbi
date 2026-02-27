# pdf_text_extractor_surya_streamlit.py
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

# ---------------- LangChain imports ----------------
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# ---------------- CONFIG ----------------
OLLAMA_BASE_URL = "http://localhost:8890/v1"
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")  # set your API key in env
OLLAMA_MODEL = "meta/llama-3.1-8b-instruct"

st.set_page_config(page_title="💬 Chat with PDF (Surya OCR + Ollama)", layout="wide")
st.title("💬 Chat with your PDF (Surya OCR + Ollama LLM)")

# ---------------- PDF → Images ----------------
def pdf_to_images(pdf_file_path: str, dpi: int = 300) -> List[Image.Image]:
    pdf_file_path = str(pdf_file_path)
    if not Path(pdf_file_path).is_file() or not pdf_file_path.lower().endswith(".pdf"):
        raise ValueError(f"Input must be a valid PDF file: {pdf_file_path}")

    images = []
    doc = pypdfium2.PdfDocument(pdf_file_path)
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

# ---------------- OCR Extraction ----------------
def extract_text_from_pdf(pdf_file_path: str, device: str = "cuda:0") -> List[str]:
    images = pdf_to_images(pdf_file_path)
    predictors = load_predictors(device=device)
    detections = predictors["detection"](images)

    # Flatten detections
    detections_for_recognition = []
    for page_dets in detections:
        flat_list = []
        for det in page_dets:
            if hasattr(det, "bbox") and hasattr(det, "text"):
                flat_list.append({"bbox": det.bbox, "text": det.text})
            elif isinstance(det, (tuple, list)) and len(det) >= 2:
                flat_list.append({"bbox": det[0], "text": det[1]})
        detections_for_recognition.append(flat_list)

    task_names_list = ["ocr_with_boxes"] * len(images)
    recognitions = predictors["recognition"](
        images=images,
        task_names=task_names_list,
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
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    docs = []
    for page_text in text_pages:
        for chunk in splitter.split_text(page_text):
            docs.append(Document(page_content=chunk))
    hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(documents=docs, embedding=hf_embeddings, persist_directory=persist_dir)
    return vectorstore

# ---------------- Ollama RAG ----------------
def ask_llama(question: str, context: str):
    url = f"{OLLAMA_BASE_URL}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OLLAMA_API_KEY}"
    }
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant. Use ONLY the provided context to answer."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{question}"}
        ]
    }
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]

# ---------------- Query PDF ----------------
def query_pdf(vectorstore, question: str, top_k: int = 4) -> str:
    results = vectorstore.similarity_search(question, k=top_k)
    context = "\n\n".join([doc.page_content for doc in results])
    answer = ask_llama(question, context)
    return answer

# ---------------- Streamlit App ----------------
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:
    file_hash = hashlib.md5(uploaded_file.read()).hexdigest()
    uploaded_file.seek(0)
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    tmp_file.write(uploaded_file.read())
    tmp_file.close()
    pdf_path = tmp_file.name

    persist_dir = f"./chroma_db_{file_hash}"

    with st.spinner("Extracting text from PDF..."):
        extracted_pages = extract_text_from_pdf(pdf_path, device="cuda:0")

    with st.spinner("Building vector store..."):
        st.session_state.vectorstore = build_vectorstore(extracted_pages, persist_dir)

    st.success("✅ PDF processed! You can now ask questions below.")
    os.remove(pdf_path)

if st.session_state.vectorstore:
    user_question = st.text_input("Ask something about the PDF:")

    if user_question:
        with st.spinner("Thinking..."):
            answer = query_pdf(st.session_state.vectorstore, user_question)
            st.markdown(f"**Answer:** {answer}")
