"""
End-to-end document ingestion: parse → chunk → embed → store.
"""

from uuid6 import uuid7
from services.document_parser import parse_document
from services.document_chunker import chunk_document
from db.vector_database import add_chunks


def ingest_document(filepath: str, session_id: str) -> dict:
    """
    Ingest a single document file into the vector store.

    Returns: {document_id, filename, chunks_created}
    """
    from pathlib import Path

    document_id = uuid7().hex

    # 1. Parse
    docs = parse_document(filepath)

    # 2. Chunk
    chunks = chunk_document(docs, document_id=document_id)

    # 3. Embed & store
    count = add_chunks(chunks, session_id=session_id)

    return {
        "document_id": document_id,
        "filename": Path(filepath).name,
        "chunks_created": count,
    }


def ingest_directory(dirpath: str) -> list[dict]:
    """Ingest all supported files in a directory."""
    from pathlib import Path

    results = []
    supported = {".pdf", ".md", ".html", ".htm"}

    for f in Path(dirpath).iterdir():
        if f.suffix.lower() in supported:
            result = ingest_document(str(f))
            results.append(result)
            print(f"  ✓ {result['filename']} → {result['chunks_created']} chunks")

    return results