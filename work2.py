import streamlit as st
import fitz
import cv2
import numpy as np
import tempfile
import os
import pandas as pd
from PIL import Image
import pytesseract
import re
import langdetect
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# ---------------- SURYA ----------------
from surya.models import load_predictors
from surya.detection import batch_text_detection
from surya.recognition import batch_text_recognition
from surya.layout import batch_layout_detection
from surya.table_rec import batch_table_recognition

# ---------------- RAG ----------------
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document


# ==========================================================
# CONFIG
# ==========================================================
st.set_page_config(page_title="Enterprise Document AI", layout="wide")
st.title("🏢 Enterprise Document Intelligence Platform")


# ==========================================================
# LOAD MODELS (Cached)
# ==========================================================
@st.cache_resource
def load_models():
    predictors = load_predictors()
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    translator = pipeline("translation", model="Helsinki-NLP/opus-mt-hi-en")
    ner_model = pipeline("ner", grouped_entities=True)
    return predictors, summarizer, translator, ner_model

predictors, summarizer, translator, ner_model = load_models()


# ==========================================================
# DOCUMENT PROCESSOR
# ==========================================================
class DocumentProcessor:

    def pdf_to_images(self, pdf_path):
        doc = fitz.open(pdf_path)
        images = []
        for page in doc:
            pix = page.get_pixmap(dpi=300)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, pix.n
            )
            if pix.n == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            images.append(self.preprocess(img))
        return images

    def preprocess(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(
            blur, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

    def process(self, images):
        det = batch_text_detection(images, predictors)
        rec = batch_text_recognition(images, det, predictors)
        layout = batch_layout_detection(images, predictors)
        tables = batch_table_recognition(images, layout, predictors)

        results = []
        for i in range(len(images)):
            results.append({
                "image": images[i],
                "text": rec[i],
                "layout": layout[i],
                "tables": tables[i]
            })
        return results


# ==========================================================
# DOCUMENT UNDERSTANDING ENGINE
# ==========================================================
class DocumentUnderstanding:

    def classify_document(self, text):
        labels = ["Invoice", "Resume", "Contract", "Report"]
        scores = [text.lower().count(k.lower()) for k in labels]
        return labels[np.argmax(scores)]

    def detect_language(self, text):
        return langdetect.detect(text)

    def extract_entities(self, text):
        return ner_model(text)

    def detect_pii(self, text):
        emails = re.findall(r'\S+@\S+', text)
        phones = re.findall(r'\b\d{10}\b', text)
        return {"emails": emails, "phones": phones}

    def mask_pii(self, text):
        text = re.sub(r'\S+@\S+', '[EMAIL]', text)
        text = re.sub(r'\b\d{10}\b', '[PHONE]', text)
        return text

    def summarize(self, text):
        if len(text) < 50:
            return text
        return summarizer(text[:1024])[0]["summary_text"]

    def translate(self, text):
        return translator(text[:512])[0]["translation_text"]

    def confidence_score(self, text_blocks):
        scores = [t.confidence for t in text_blocks if hasattr(t, "confidence")]
        return np.mean(scores) if scores else 0.85


# ==========================================================
# ADVANCED TABLE ENGINE
# ==========================================================
class AdvancedTableEngine:

    def to_dataframe(self, table):
        matrix = {}
        for cell in table.cells:
            matrix.setdefault(cell.row, {})
            matrix[cell.row][cell.col] = cell.text

        df = pd.DataFrame.from_dict(matrix, orient="index").fillna("")
        return df

    def detect_numeric_columns(self, df):
        numeric_cols = []
        for col in df.columns:
            try:
                df[col].astype(float)
                numeric_cols.append(col)
            except:
                pass
        return numeric_cols

    def detect_financial_table(self, df):
        keywords = ["amount", "total", "balance"]
        for col in df.columns:
            if any(k in str(col).lower() for k in keywords):
                return True
        return False

    def confidence(self, df):
        filled_ratio = df.replace("", np.nan).count().sum() / (df.shape[0]*df.shape[1])
        return round(filled_ratio, 2)


# ==========================================================
# ADVANCED CHART ENGINE
# ==========================================================
class AdvancedChartEngine:

    def crop(self, img, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        return img[y1:y2, x1:x2]

    def extract_axis_labels(self, img):
        text = pytesseract.image_to_string(img)
        return text.split("\n")

    def detect_bars(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)[1]
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bars = [cv2.boundingRect(c)[3] for c in contours if cv2.boundingRect(c)[3] > 30]
        return sorted(bars)

    def confidence(self, bars):
        return min(1.0, len(bars)/5)


# ==========================================================
# MAIN PIPELINE
# ==========================================================
uploaded = st.file_uploader("Upload PDF / PNG / JPG", type=["pdf","png","jpg","jpeg"])

if uploaded:

    tmp = tempfile.NamedTemporaryFile(delete=False)
    tmp.write(uploaded.read())
    tmp.close()
    path = tmp.name

    processor = DocumentProcessor()
    understanding = DocumentUnderstanding()
    table_engine = AdvancedTableEngine()
    chart_engine = AdvancedChartEngine()

    if path.endswith(".pdf"):
        images = processor.pdf_to_images(path)
    else:
        images = [cv2.imread(path)]

    results = processor.process(images)

    full_text = ""

    for page in results:

        text_blocks = page["text"]
        ordered = sorted(text_blocks, key=lambda x: (x.bbox[1], x.bbox[0]))
        page_text = " ".join([t.text for t in ordered])
        full_text += page_text

        st.subheader("📄 Extracted Text")
        st.write(page_text)

        # ---------------- Understanding ----------------
        st.subheader("🧠 Document Intelligence")

        doc_type = understanding.classify_document(page_text)
        st.write("Document Type:", doc_type)

        lang = understanding.detect_language(page_text)
        st.write("Language:", lang)

        entities = understanding.extract_entities(page_text)
        st.json(entities)

        pii = understanding.detect_pii(page_text)
        st.json(pii)

        st.write("Confidence Score:", understanding.confidence_score(text_blocks))

        st.subheader("📝 Summary")
        st.write(understanding.summarize(page_text))

        st.subheader("🌍 Translation")
        st.write(understanding.translate(page_text))


        # ---------------- Tables ----------------
        for table in page["tables"]:
            df = table_engine.to_dataframe(table)
            st.dataframe(df)

            st.write("Numeric Columns:", table_engine.detect_numeric_columns(df))
            st.write("Financial Table:", table_engine.detect_financial_table(df))
            st.write("Table Confidence:", table_engine.confidence(df))


        # ---------------- Charts ----------------
        for block in page["layout"]:
            if block.label in ["Figure", "Chart"]:
                crop = chart_engine.crop(page["image"], block.bbox)
                st.image(crop)

                bars = chart_engine.detect_bars(crop)
                axis_labels = chart_engine.extract_axis_labels(crop)

                st.json({
                    "bars": bars,
                    "axis_labels": axis_labels,
                    "confidence": chart_engine.confidence(bars)
                })


    # ---------------- RAG ----------------
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    docs = [Document(page_content=chunk) for chunk in splitter.split_text(full_text)]
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(docs, embeddings)

    st.success("✅ Enterprise Document Intelligence Completed")
