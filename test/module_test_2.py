"""
Module 2 tests — document parsing, chunking, vector store.

Run:  pytest tests/test_module2.py -v
"""

import tempfile
from services.document_parser import parse_document
from services.document_chunker import chunk_document, extract_chunk_metadata


# ── Parser ───────────────────────────────────────────

def test_parse_markdown():
    with tempfile.NamedTemporaryFile(suffix=".md", mode="w", delete=False) as f:
        f.write("# Title\n\nSome content here.\n\n## Section 2\n\nMore content.")
        f.flush()
        doc = parse_document(f.name)

    assert doc.doc_type == "markdown"
    assert "Title" in doc.content
    assert "Section 2" in doc.content


def test_parse_html():
    with tempfile.NamedTemporaryFile(suffix=".html", mode="w", delete=False) as f:
        f.write("<html><body><h1>Hello</h1><p>World</p></body></html>")
        f.flush()
        doc = parse_document(f.name)

    assert doc.doc_type == "html"
    assert "Hello" in doc.content
    assert "World" in doc.content


# ── Chunker ──────────────────────────────────────────

def test_chunk_document():
    from app.services.document_parser import ParsedDocument
    doc = ParsedDocument(
        content="A " * 600,  # ~1200 chars → should split into 2+ chunks
        filename="test.md",
        doc_type="markdown",
    )
    chunks = chunk_document(doc, document_id="test123")
    assert len(chunks) >= 2
    assert chunks[0]["document_id"] == "test123"
    assert chunks[0]["metadata"]["source_file"] == "test.md"


def test_metadata_extraction():
    text = "# My Header\n\nSome text\n\n```python\nprint('hi')\n```"
    meta = extract_chunk_metadata(text)
    assert meta["section_header"] == "My Header"
    assert meta["has_code_block"] is True


def test_metadata_no_header():
    meta = extract_chunk_metadata("Just plain text here.")
    assert meta["section_header"] is None
    assert meta["has_code_block"] is False