"""
Splits parsed documents into chunks with metadata.

Uses LangChain's RecursiveCharacterTextSplitter for smart splitting,
then extracts metadata (section headers, code block presence) per chunk.
"""

import re
from uuid6 import uuid7
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import get_settings
from services.document_parser import ParsedDocument


def extract_chunk_metadata(text: str) -> dict:
    """Pull section header and code block info from a chunk."""
    # Find the last markdown-style header before content
    headers = re.findall(r"^#{1,4}\s+(.+)", text, re.MULTILINE)
    has_code = bool(re.search(r"```[\s\S]*?```|    \S", text))

    return {
        "section_header": headers[0] if headers else None,
        "has_code_block": has_code,
    }


def chunk_document(doc: ParsedDocument, document_id: str | None = None) -> list[dict]:
    """
    Split a parsed document into chunks ready for embedding.

    Returns list of dicts with keys: chunk_id, document_id, content, metadata
    """
    settings = get_settings()
    doc_id = document_id or uuid7().hex

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    texts = splitter.split_text(doc.content)

    chunks = []
    for i, text in enumerate(texts):
        meta = extract_chunk_metadata(text)
        meta.update({
            "source_file": doc.filename,
            "document_type": doc.doc_type,
            "chunk_index": i,
            "page_number": None,  # TODO: track page boundaries for PDFs
        })

        chunks.append({
            "chunk_id": f"{doc_id}_{i}",
            "document_id": doc_id,
            "content": text,
            "metadata": meta,
        })

    return chunks