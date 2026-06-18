"""
doc_parser.py — converts uploaded bytes to plain text.
Supports: .pdf (PyMuPDF), .docx (python-docx), .txt/.md (utf-8)
"""

import io
import re


def parse_document(raw: bytes, ext: str) -> str:
    ext = ext.lower()
    if ext == ".pdf":
        return _parse_pdf(raw)
    elif ext == ".docx":
        return _parse_docx(raw)
    elif ext in (".txt", ".md"):
        return raw.decode("utf-8", errors="replace")
    else:
        raise ValueError(f"Unsupported extension: {ext}")


def _parse_pdf(raw: bytes) -> str:
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise RuntimeError("PyMuPDF not installed. Run: pip install pymupdf")

    doc = fitz.open(stream=raw, filetype="pdf")
    parts = []
    for page in doc:
        text = page.get_text("text")
        if text.strip():
            parts.append(text)
    doc.close()
    return _clean("\n\n".join(parts))


def _parse_docx(raw: bytes) -> str:
    try:
        from docx import Document
    except ImportError:
        raise RuntimeError("python-docx not installed. Run: pip install python-docx")

    doc = Document(io.BytesIO(raw))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return _clean("\n\n".join(paragraphs))


def _clean(text: str) -> str:
    # Collapse multiple blank lines, strip trailing whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    return text.strip()
