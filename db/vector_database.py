"""
Chroma vector store wrapper — embed chunks, search by query.

Each session gets its own Chroma collection so documents are isolated per session.
"""

import chromadb
from langchain_openai import OpenAIEmbeddings
from config import get_settings

settings = get_settings()

_client: chromadb.ClientAPI | None = None
_embeddings: OpenAIEmbeddings | None = None
_collections: dict[str, chromadb.Collection] = {}


def get_embeddings() -> OpenAIEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = OpenAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            api_key=settings.OPENAI_API_KEY,
        )
    return _embeddings


def _get_client() -> chromadb.ClientAPI:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)
    return _client


def get_collection(session_id: str) -> chromadb.Collection:
    if session_id not in _collections:
        _collections[session_id] = _get_client().get_or_create_collection(
            name=f"session_{session_id}",
            metadata={"hnsw:space": "cosine"},
        )
    return _collections[session_id]


def embed_texts(texts: list[str]) -> list[list[float]]:
    return get_embeddings().embed_documents(texts)


def add_chunks(chunks: list[dict], session_id: str) -> int:
    if not chunks:
        return 0

    collection = get_collection(session_id)
    ids = [c["chunk_id"] for c in chunks]
    documents = [c["content"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]
    embeddings = embed_texts(documents)

    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
    )
    return len(chunks)


def search(query: str, session_id: str, top_k: int = 5) -> list[dict]:
    collection = get_collection(session_id)
    query_embedding = get_embeddings().embed_query(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    for i in range(len(results["ids"][0])):
        hits.append({
            "chunk_id": results["ids"][0][i],
            "content": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "score": 1 - results["distances"][0][i],
        })
    return hits
