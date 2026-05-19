"""
Module 2 tests — document parsing, chunking, vector store.

Run:  pytest test/module_test_2.py -v
"""

import tempfile
import os
from langchain_core.documents import Document
from services.document_parser import parse_document
from services.document_chunker import chunk_document, extract_chunk_metadata


# ── Parser ───────────────────────────────────────────

def test_parse_markdown():
    with tempfile.NamedTemporaryFile(suffix=".md", mode="w", delete=False) as f:
        f.write("# Title\n\nSome content here.\n\n## Section 2\n\nMore content.")
        name = f.name
    try:
        docs = parse_document(name)
        assert isinstance(docs, list)
        assert len(docs) > 0
        full_text = " ".join(d.page_content for d in docs)
        assert "Title" in full_text or "content" in full_text
    finally:
        os.unlink(name)


def test_parse_html():
    with tempfile.NamedTemporaryFile(suffix=".html", mode="w", delete=False) as f:
        f.write("<html><body><h1>Hello</h1><p>World</p></body></html>")
        name = f.name
    try:
        docs = parse_document(name)
        assert isinstance(docs, list)
        assert len(docs) > 0
        full_text = " ".join(d.page_content for d in docs)
        assert "Hello" in full_text or "World" in full_text
    finally:
        os.unlink(name)


def test_unsupported_extension_raises():
    import pytest
    with pytest.raises(ValueError, match="Unsupported"):
        parse_document("file.txt")


# ── Chunker ──────────────────────────────────────────

def test_chunk_document():
    docs = [Document(page_content="A " * 600, metadata={"source": "test.md"})]
    chunks = chunk_document(docs, document_id="test123")
    assert len(chunks) >= 1
    assert chunks[0]["document_id"] == "test123"
    assert chunks[0]["metadata"]["source_file"] == "test.md"


def test_chunk_preserves_metadata():
    docs = [Document(page_content="short content", metadata={"source": "doc.pdf", "page": 3})]
    chunks = chunk_document(docs, document_id="abc")
    assert chunks[0]["metadata"]["page_number"] == 3
    assert chunks[0]["metadata"]["source_file"] == "doc.pdf"


def test_metadata_extraction():
    text = "# My Header\n\nSome text\n\n```python\nprint('hi')\n```"
    meta = extract_chunk_metadata(text)
    assert meta["section_header"] == "My Header"
    assert meta["has_code_block"] is True


def test_metadata_no_header():
    meta = extract_chunk_metadata("Just plain text here.")
    assert meta["section_header"] is None
    assert meta["has_code_block"] is False
