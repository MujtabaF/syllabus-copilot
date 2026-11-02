import hashlib
import os
import re
import uuid

import numpy as np
import psycopg
import pytest
from mcp import Client
from psycopg.conninfo import make_conninfo

from syllabus_copilot.config import settings
from syllabus_copilot.docs import EMBEDDING_DIM, DocStore, chunk_text
from syllabus_copilot.servers.course_docs import create_server

SYLLABUS = """--- page 1 ---
BIO 200. Welcome to cell biology, taught in the spring.

--- page 2 ---
Grading: the midterm exam is worth 30 percent and the final exam is worth 40 percent.

--- page 3 ---
Office hours are on Tuesday afternoons in the science building.
"""


def fake_embed(texts):
    """Bag-of-words hashing: a small, offline stand-in that still ranks by shared words."""
    vectors = []
    for text in texts:
        vector = np.zeros(EMBEDDING_DIM)
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            vector[int(hashlib.md5(word.encode()).hexdigest(), 16) % EMBEDDING_DIM] += 1
        vectors.append((vector / (np.linalg.norm(vector) or 1)).tolist())
    return vectors


def test_chunks_keep_page_numbers():
    chunks = chunk_text(SYLLABUS)
    assert [c.page for c in chunks] == [1, 2, 3]
    assert "midterm" in chunks[1].content
    assert [c.position for c in chunks] == [0, 1, 2]


def test_text_without_page_markers_has_no_page():
    assert [c.page for c in chunk_text("Just some text.")] == [None]


def test_long_text_is_split_with_overlap_and_nothing_lost():
    words = [f"word{i}" for i in range(1000)]
    chunks = chunk_text(" ".join(words), size=200, overlap=50)
    assert all(len(c.content) <= 200 for c in chunks)
    assert set(words) <= set(" ".join(c.content for c in chunks).split())
    assert chunks[1].content.split()[0] in chunks[0].content  # consecutive chunks overlap


@pytest.fixture
def store():
    """A store in its own throwaway schema. Skips if there's no database, except in CI."""
    schema = f"test_{uuid.uuid4().hex}"
    try:
        admin = psycopg.connect(settings.database_url, autocommit=True)
    except psycopg.OperationalError:
        if os.getenv("CI"):
            raise
        pytest.skip("Postgres isn't running (docker compose up -d)")
    with admin:
        admin.execute(f"CREATE SCHEMA {schema}")
        url = make_conninfo(settings.database_url, options=f"-c search_path={schema},public")
        yield DocStore(url, embed=fake_embed)
        admin.execute(f"DROP SCHEMA {schema} CASCADE")


def test_search_finds_the_right_page(store):
    store.ingest(SYLLABUS, "BIO 200", "Cell Biology", "bio.txt")

    top = store.search("how much is the final exam worth", limit=1)[0]

    assert (top.course_code, top.course_name, top.page) == ("BIO 200", "Cell Biology", 2)
    assert top.score > 0


def test_search_can_be_limited_to_one_course(store):
    store.ingest(SYLLABUS, "BIO 200", None, "bio.txt")
    store.ingest("--- page 1 ---\nThe final exam is worth 50 percent.", "CHEM 100", None, "c.txt")

    hits = store.search("final exam", course_code="CHEM 100")

    assert {h.course_code for h in hits} == {"CHEM 100"}


def test_ingesting_again_replaces_the_old_version(store):
    store.ingest(SYLLABUS, "BIO 200", None, "old.txt")
    store.ingest("--- page 1 ---\nNew syllabus.", "BIO 200", None, "new.txt")

    assert store.courses() == [{"code": "BIO 200", "name": None, "source": "new.txt", "chunks": 1}]


def test_empty_document_gets_a_helpful_error(store):
    with pytest.raises(ValueError, match="scanned image"):
        store.ingest("--- page 1 ---\n   ", "BIO 200", None, "scan.pdf")


async def test_server_tools_end_to_end(store, tmp_path):
    (tmp_path / "bio.txt").write_text(SYLLABUS)

    async with Client(create_server(store, tmp_path)) as client:
        added = await client.call_tool(
            "docs_ingest_syllabus", {"path": "bio.txt", "course_code": "BIO 200"}
        )
        assert added.structured_content == {"course_code": "BIO 200", "chunks": 3}

        found = await client.call_tool("docs_search", {"query": "office hours tuesday", "limit": 1})
        assert found.structured_content["results"][0]["page"] == 3

        listed = await client.call_tool("docs_list_courses", {})
        assert listed.structured_content["courses"][0]["code"] == "BIO 200"


@pytest.mark.parametrize("path", ["../secret.txt", "/etc/passwd", "notes.docx", "missing.txt"])
async def test_server_refuses_files_it_should_not_read(path, tmp_path):
    root = tmp_path / "syllabi"
    root.mkdir()
    (tmp_path / "secret.txt").write_text("private")
    (root / "notes.docx").write_text("not a syllabus")
    server = create_server(DocStore("postgresql://unused", embed=fake_embed), root)

    async with Client(server) as client:
        result = await client.call_tool("docs_ingest_syllabus", {"path": path, "course_code": "X"})

    assert result.is_error
    assert path in result.content[0].text  # the reason reaches the agent
