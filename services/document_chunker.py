"""
Splits parsed documents into chunks with metadata.

Uses LangChain's RecursiveCharacterTextSplitter for smart splitting,
then extracts metadata (section headers, code block presence) per chunk.
"""

import re
from pathlib import Path
from uuid6 import uuid7
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import get_settings


def extract_chunk_metadata(text: str) -> dict:
    """Pull section header and code block info from a chunk."""
    headers = re.findall(r"^#{1,4}\s+(.+)", text, re.MULTILINE)
    has_code = bool(re.search(r"```[\s\S]*?```|    \S", text))

    return {
        "section_header": headers[0] if headers else None,
        "has_code_block": has_code,
    }


def chunk_document(docs: list[Document], document_id: str | None = None) -> list[dict]:
    """
    Split LangChain Documents into chunks ready for embedding.

    Returns list of dicts with keys: chunk_id, document_id, content, metadata
    """
    settings = get_settings()
    doc_id = document_id or uuid7().hex

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    split_docs = splitter.split_documents(docs)

    chunks = []
    for i, doc in enumerate(split_docs):
        lc_meta = doc.metadata
        source = lc_meta.get("source", "unknown")
        meta = extract_chunk_metadata(doc.page_content)
        page = lc_meta.get("page")
        meta.update({
            "source_file": Path(source).name if source else "unknown",
            "document_type": lc_meta.get("file_type", Path(source).suffix.lstrip(".") if source else "unknown"),
            "chunk_index": i,
            **({"page_number": page} if page is not None else {}),
        })
        # Chroma rejects None values — remove any that slipped through
        meta = {k: v for k, v in meta.items() if v is not None}

        chunks.append({
            "chunk_id": f"{doc_id}_{i}",
            "document_id": doc_id,
            "content": doc.page_content,
            "metadata": meta,
        })

    return chunks