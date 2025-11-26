import fitz  # PyMuPDF
import docx
import re

def extract_text_from_pdf(file_obj):
    pdf_bytes = file_obj.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def extract_text_from_docx(file_obj):
    doc = docx.Document(file_obj)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return "\n".join(full_text)

def load_resume_from_fileobj(file_obj):
    filename = file_obj.name.lower()

    if filename.endswith(".txt"):
        return file_obj.read().decode("utf-8", errors="ignore")

    elif filename.endswith(".pdf"):
        return extract_text_from_pdf(file_obj)

    elif filename.endswith(".docx"):
        return extract_text_from_docx(file_obj)

    else:
        raise ValueError("Unsupported resume file format. Use PDF, DOCX, or TXT.")


def clean_text(text):
    if not text:
        return ""
    text = re.sub(r'<[^>]*?>', '', text)
    text = re.sub(r'http[s]?://\S+', '', text)
    text = text.replace("\r", "\n")
    text = re.sub(r"[^\S\n]+", " ", text)
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    return "\n".join(lines)


def load_resume_from_path():
    return None