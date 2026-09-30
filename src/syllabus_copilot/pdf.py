from pathlib import Path

from pypdf import PdfReader


def pdf_to_text(path: Path) -> str:
    """Extract text from a PDF, with page markers so the LLM can cite pages later."""
    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(f"--- page {i} ---\n{text}" for i, text in enumerate(pages, start=1))
