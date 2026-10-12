import re
from pathlib import Path

from pypdf import PdfReader


def _page_text(page) -> str:
    # Layout mode keeps table columns aligned and avoids the one-word-per-line output that
    # plain mode gives for some PDFs (e.g. Google Docs exports).
    text = page.extract_text(extraction_mode="layout") or ""
    lines = [re.sub(r" {4,}", "   ", line.rstrip()) for line in text.splitlines()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def pdf_to_text(path: Path) -> str:
    """Extract text from a PDF, with page markers so the LLM can cite pages later."""
    reader = PdfReader(path)
    pages = [_page_text(page) for page in reader.pages]
    return "\n\n".join(f"--- page {i} ---\n{text}" for i, text in enumerate(pages, start=1))
