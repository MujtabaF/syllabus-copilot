"""Store syllabi as searchable chunks in Postgres + pgvector.

Each chunk keeps its page number, so answers can cite where they came from.
"""

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

from .pdf import pdf_to_text

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
CHUNK_CHARS = 800
CHUNK_OVERLAP_CHARS = 100

EmbedFn = Callable[[Sequence[str]], Sequence[Sequence[float]]]

_PAGE_MARKER = re.compile(r"^--- page (\d+) ---$", re.MULTILINE)

_SCHEMA = f"""
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS courses (
    code text PRIMARY KEY,
    name text,
    source text NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    id bigserial PRIMARY KEY,
    course_code text NOT NULL REFERENCES courses (code) ON DELETE CASCADE,
    page integer,
    position integer NOT NULL,
    content text NOT NULL,
    embedding vector({EMBEDDING_DIM}) NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_course_idx ON chunks (course_code);
CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON chunks USING hnsw (embedding vector_cosine_ops);
"""


@dataclass(frozen=True)
class Chunk:
    page: int | None
    position: int
    content: str


@dataclass(frozen=True)
class SearchHit:
    course_code: str
    course_name: str | None
    page: int | None
    score: float
    content: str


def _split_text(text: str, size: int, overlap: int) -> list[str]:
    """Split on paragraph breaks where possible, cutting long paragraphs with some overlap."""
    pieces: list[str] = []
    current = ""
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if current and len(current) + len(paragraph) + 2 > size:
            pieces.append(current)
            current = ""
        current = f"{current}\n\n{paragraph}" if current else paragraph
        while len(current) > size:
            cut = current.rfind(" ", size - overlap, size)
            cut = cut if cut > 0 else size
            pieces.append(current[:cut].strip())
            current = current[max(cut - overlap, 1) :].strip()
    if current:
        pieces.append(current)
    return pieces


def chunk_text(
    text: str, size: int = CHUNK_CHARS, overlap: int = CHUNK_OVERLAP_CHARS
) -> list[Chunk]:
    """Split syllabus text into chunks, tracking pages from `--- page N ---` markers."""
    parts = _PAGE_MARKER.split(text)
    # re.split gives [before first marker, page number, page text, page number, ...]
    pages: list[tuple[int | None, str]] = [(None, parts[0])]
    pages += [(int(parts[i]), parts[i + 1]) for i in range(1, len(parts), 2)]

    chunks: list[Chunk] = []
    for page, page_text in pages:
        for piece in _split_text(page_text, size, overlap):
            chunks.append(Chunk(page=page, position=len(chunks), content=piece))
    return chunks


@lru_cache(maxsize=1)
def _load_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL)


def embed_local(texts: Sequence[str]) -> list[list[float]]:
    """Embed text on this machine, so syllabi never leave it and there's nothing to pay for."""
    vectors = _load_model().encode(list(texts), normalize_embeddings=True, show_progress_bar=False)
    return vectors.tolist()


def read_syllabus(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        return pdf_to_text(path)
    return path.read_text()


class DocStore:
    """Syllabus chunks in Postgres. `embed` is injectable so tests don't need the model."""

    def __init__(self, database_url: str, embed: EmbedFn = embed_local):
        self.database_url = database_url
        self.embed = embed

    def _connect(self) -> psycopg.Connection:
        conn = psycopg.connect(self.database_url)
        try:
            conn.execute(_SCHEMA)
            register_vector(conn)
        except Exception:
            conn.close()
            raise
        return conn

    def ingest(self, text: str, course_code: str, course_name: str | None, source: str) -> int:
        """Add a syllabus, replacing any earlier version of the course. Returns chunk count."""
        chunks = chunk_text(text)
        if not chunks:
            raise ValueError(f"No text found in {source}. It may be a scanned image.")
        vectors = self.embed([c.content for c in chunks])

        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("DELETE FROM courses WHERE code = %s", (course_code,))
            cur.execute(
                "INSERT INTO courses (code, name, source) VALUES (%s, %s, %s)",
                (course_code, course_name, source),
            )
            cur.executemany(
                "INSERT INTO chunks (course_code, page, position, content, embedding) "
                "VALUES (%s, %s, %s, %s, %s)",
                [
                    (course_code, c.page, c.position, c.content, np.array(v))
                    for c, v in zip(chunks, vectors, strict=True)
                ],
            )
        return len(chunks)

    def search(self, query: str, course_code: str | None = None, limit: int = 5) -> list[SearchHit]:
        vector = np.array(self.embed([query])[0])
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT c.course_code, co.name, c.page, 1 - (c.embedding <=> %(v)s), c.content "
                "FROM chunks c JOIN courses co ON co.code = c.course_code "
                "WHERE %(course)s::text IS NULL OR c.course_code = %(course)s "
                "ORDER BY c.embedding <=> %(v)s LIMIT %(limit)s",
                {"v": vector, "course": course_code, "limit": limit},
            ).fetchall()
        return [SearchHit(r[0], r[1], r[2], float(r[3]), r[4]) for r in rows]

    def courses(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT co.code, co.name, co.source, count(c.id) FROM courses co "
                "LEFT JOIN chunks c ON c.course_code = co.code "
                "GROUP BY co.code ORDER BY co.code"
            ).fetchall()
        return [{"code": r[0], "name": r[1], "source": r[2], "chunks": r[3]} for r in rows]
