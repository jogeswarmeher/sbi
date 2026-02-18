import fitz  # PyMuPDF
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import cv2
import numpy as np
from docling.document_converter import DocumentConverter


PDF_PATH = "pdf/pan.pdf"
OUTPUT_PDF = "output_with_text.pdf"
TESS_LANG = "eng+hin"


# ---------- 1️⃣ OCR WITH POSITION ----------
def extract_ocr_with_position(pdf_path):
    pages = convert_from_path(pdf_path, dpi=300)
    ocr_results = []

    for i, page in enumerate(pages):
        open_cv_image = np.array(page)
        gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)

        data = pytesseract.image_to_data(
            gray,
            lang=TESS_LANG,
            output_type=pytesseract.Output.DICT
        )

        page_blocks = []

        for j in range(len(data["text"])):
            if int(data["conf"][j]) > 50 and data["text"][j].strip():
                x = data["left"][j]
                y = data["top"][j]
                w = data["width"][j]
                h = data["height"][j]

                page_blocks.append({
                    "text": data["text"][j],
                    "bbox": [x, y, x + w, y + h]
                })

        ocr_results.append(page_blocks)

    return ocr_results


# ---------- 2️⃣ WRITE OCR TEXT BACK INTO PDF ----------
def write_text_into_pdf(input_pdf, ocr_data):
    doc = fitz.open(input_pdf)

    for page_index, page in enumerate(doc):
        if page_index >= len(ocr_data):
            continue

        page_width = page.rect.width
        page_height = page.rect.height

        # PDF image resolution scaling (300 DPI used in convert)
        scale_x = page_width / 2480  # approx A4 width at 300 dpi
        scale_y = page_height / 3508  # approx A4 height at 300 dpi

        for block in ocr_data[page_index]:
            x0, y0, x1, y1 = block["bbox"]

            rect = fitz.Rect(
                x0 * scale_x,
                y0 * scale_y,
                x1 * scale_x,
                y1 * scale_y
            )

            page.insert_textbox(
                rect,
                block["text"],
                fontsize=8,
                fontname="helv",
                overlay=True  # keeps original content
            )

    doc.save(OUTPUT_PDF)
    doc.close()


# ---------- 3️⃣ OPTIONAL: DOCILING STRUCTURE ----------
def extract_docling_structure(pdf_path):
    converter = DocumentConverter()
    result = converter.convert(pdf_path)
    return result.document


# ---------- MAIN ----------
if __name__ == "__main__":
    print("Running OCR...")
    ocr_data = extract_ocr_with_position(PDF_PATH)

    print("Writing text into PDF...")
    write_text_into_pdf(PDF_PATH, ocr_data)

    print("Done! Saved as:", OUTPUT_PDF)
