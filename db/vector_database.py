"""
Chroma vector store wrapper — embed chunks, search by query.

Uses OpenAI embeddings via langchain-openai.
"""

import chromadb
from langchain_openai import OpenAIEmbeddings
from config import get_settings

settings = get_settings()

client: chromadb.ClientAPI | None = None
collection: chromadb.Collection | None = None
embeddings: OpenAIEmbeddings | None = None


def get_embeddings() -> OpenAIEmbeddings:
    if embeddings is None:
        embeddings = OpenAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            api_key=settings.OPENAI_API_KEY,
        )
    return embeddings


def get_collection() -> chromadb.Collection:
    if collection is None:
        client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)
        collection = client.get_or_create_collection(
            name=settings.CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
    return collection


def embed_texts(texts: list[str]) -> list[list[float]]:
    return get_embeddings().embed_documents(texts)


def add_chunks(chunks: list[dict]) -> int:
    """
    Add document chunks to Chroma.
    Each chunk dict must have: chunk_id, content, metadata
    """
    if not chunks:
        return 0

    collection = get_collection()
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


def search(query: str, top_k: int = 5) -> list[dict]:
    """
    Semantic search — returns list of {chunk_id, content, metadata, score}.
    """
    collection = get_collection()
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
            "score": 1 - results["distances"][0][i],  # cosine distance → similarity
        })
    return hits
