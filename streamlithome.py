import streamlit as st
import fitz
import pytesseract
import cv2
import numpy as np
import tempfile
import hashlib

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

TESS_LANG = "eng+hin"

st.set_page_config(page_title="Chat with PDF", layout="wide")
st.title("💬 Chat with your PDF (Hindi + English)")

# ---------------- SESSION MEMORY ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------- TEXT EXTRACTION ----------------
@st.cache_data
def extract_text(pdf_path):
    text = ""
    doc = fitz.open(pdf_path)

    for page in doc:
        text += page.get_text()

    # OCR fallback
    if len(text.strip()) < 50:
        for page in doc:
            pix = page.get_pixmap()
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, pix.n
            )

            if pix.n == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            text += pytesseract.image_to_string(gray, lang=TESS_LANG)

    return text


# ---------------- VECTOR STORE ----------------
@st.cache_resource
def build_vectorstore(text, persist_dir):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    docs = [Document(page_content=chunk) for chunk in splitter.split_text(text)]

    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_dir
    )

    return vectorstore


# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:

    file_hash = hashlib.md5(uploaded_file.read()).hexdigest()
    uploaded_file.seek(0)

    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name

    persist_dir = f"./chroma_db_{file_hash}"

    with st.spinner("Processing PDF..."):
        full_text = extract_text(pdf_path)
        vectorstore = build_vectorstore(full_text, persist_dir)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    llm = Ollama(model="llama3.1:latest")

    # ---------------- PROMPT ----------------
    prompt = ChatPromptTemplate.from_template(
        """You are a helpful assistant. 
        Use ONLY the provided context to answer.

        Conversation History:
        {history}

        Context:
        {context}

        Question:
        {question}

        Answer:"""
    )

    # ---------------- RAG CHAIN ----------------
    def build_history():
        history_text = ""
        for msg in st.session_state.messages:
            role = "User" if msg["role"] == "user" else "Assistant"
            history_text += f"{role}: {msg['content']}\n"
        return history_text

    def generate_answer(question):
        docs = retriever.invoke(question)
        context = "\n\n".join([doc.page_content for doc in docs])
        history = build_history()

        formatted_prompt = prompt.format(
            context=context,
            question=question,
            history=history
        )

        response = llm.invoke(formatted_prompt)
        return response


    st.success("PDF Ready! Start chatting 👇")

    # ---------------- CHAT DISPLAY ----------------
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # ---------------- USER INPUT ----------------
    user_input = st.chat_input("Ask something about the PDF...")

    if user_input:
        # Add user message
        st.session_state.messages.append(
            {"role": "user", "content": user_input}
        )

        with st.chat_message("user"):
            st.markdown(user_input)

        # Generate answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = generate_answer(user_input)
                st.markdown(answer)

        # Save assistant response
        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )
