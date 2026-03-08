"""
document_processor.py
Multi-format academic document processor.
Supports: PDF, DOCX, PPTX, TXT
Detects content type: textbook, notes, question_paper, lab_manual
Returns plain dicts so app.py can do doc["filename"] = ... etc.
"""

import os
import re
from typing import Optional


# ─── Keyword heuristics for content-type detection ────────────────────────────
_TYPE_KEYWORDS = {
    "question_paper": [
        r"\bq\d+\b", r"\bquestion\s+\d+\b", r"\bmark[s]?\b",
        r"\bexam\b", r"\buniversity\b", r"\btime\s*:\s*\d",
        r"\banswer\s+all\b", r"\bsection\s+[a-z]\b",
    ],
    "lab_manual": [
        r"\baim\b", r"\bapparatus\b", r"\bprocedure\b",
        r"\bobservation\b", r"\bviva\b", r"\bexperiment\b",
        r"\bresult\b.*\btable\b",
    ],
    "textbook": [
        r"\bchapter\s+\d+\b", r"\bdefinition\b", r"\btheorem\b",
        r"\bexercise\b", r"\bsummary\b", r"\blearning\s+objective",
    ],
    "notes": [
        r"\blecture\s+\d+\b", r"\bnote[s]?\b", r"\btopic\s*:\s*",
        r"\bunit\s+\d+\b", r"\bimportant\b",
    ],
}


def _detect_content_type(text: str) -> str:
    text_lower = text.lower()
    scores = {t: 0 for t in _TYPE_KEYWORDS}

    for ctype, patterns in _TYPE_KEYWORDS.items():
        for pat in patterns:
            scores[ctype] += len(re.findall(pat, text_lower))

    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "general"


# ─── Text chunker ─────────────────────────────────────────────────────────────
def _chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> list:
    sentences = re.split(r"(?<=[.!?])\s+", text)

    chunks = []
    current = []
    current_len = 0

    for sent in sentences:
        slen = len(sent)

        if current_len + slen > chunk_size and current:
            chunks.append(" ".join(current))

            overlap_chars = 0
            overlap_sents = []

            for s in reversed(current):
                overlap_chars += len(s)
                overlap_sents.insert(0, s)
                if overlap_chars >= overlap:
                    break

            current = overlap_sents
            current_len = sum(len(s) for s in current)

        current.append(sent)
        current_len += slen

    if current:
        chunks.append(" ".join(current))

    return [c.strip() for c in chunks if c.strip()]


# ─── Format-specific extractors ───────────────────────────────────────────────
def _extract_pdf(filepath: str) -> str:
    try:
        import pdfplumber

        text_parts = []
        with pdfplumber.open(filepath) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text_parts.append(t)

        return "\n".join(text_parts)

    except ImportError:
        pass

    try:
        import PyPDF2

        text_parts = []
        with open(filepath, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                t = page.extract_text()
                if t:
                    text_parts.append(t)

        return "\n".join(text_parts)

    except Exception as e:
        return f"[PDF extraction error: {e}]"


def _extract_docx(filepath: str) -> str:
    try:
        from docx import Document

        doc = Document(filepath)
        parts = []

        for para in doc.paragraphs:
            if para.text.strip():
                parts.append(para.text)

        for table in doc.tables:
            for row in table.rows:
                row_text = " | ".join(cell.text.strip() for cell in row.cells)
                if row_text.strip():
                    parts.append(row_text)

        return "\n".join(parts)

    except Exception as e:
        return f"[DOCX extraction error: {e}]"


def _extract_pptx(filepath: str) -> str:
    try:
        from pptx import Presentation

        prs = Presentation(filepath)
        parts = []

        for i, slide in enumerate(prs.slides, 1):
            slide_texts = []
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text.strip():
                    slide_texts.append(shape.text.strip())
            if slide_texts:
                parts.append(f"[Slide {i}]\n" + "\n".join(slide_texts))

        return "\n\n".join(parts)

    except Exception as e:
        return f"[PPTX extraction error: {e}]"


def _extract_txt(filepath: str) -> str:
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


# ─── Main processing function ─────────────────────────────────────────────────
def process_document(filepath: str, force_type: Optional[str] = None) -> dict:
    """
    Process a document and return a plain dict with keys:
        filename, content_type, raw_text, chunks, num_chunks, file_type
    Returning a dict (not a dataclass) so app.py can do doc["key"] = value.
    """
    filename = os.path.basename(filepath)
    ext = os.path.splitext(filename)[1].lower()

    extractors = {
        ".pdf":  _extract_pdf,
        ".docx": _extract_docx,
        ".pptx": _extract_pptx,
        ".txt":  _extract_txt,
        ".md":   _extract_txt,
    }

    if ext not in extractors:
        raise ValueError(f"Unsupported file type: {ext}")

    raw_text = extractors[ext](filepath)
    content_type = force_type or _detect_content_type(raw_text)
    chunks = _chunk_text(raw_text)

    return {
        "filename":     filename,
        "content_type": content_type,
        "raw_text":     raw_text,
        "chunks":       chunks,
        "num_chunks":   len(chunks),
        "file_type":    ext,
    }