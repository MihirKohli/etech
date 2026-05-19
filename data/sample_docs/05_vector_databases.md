# Vector Databases — Concepts and Comparison

## What is a Vector Database?

A vector database stores high-dimensional numerical vectors (embeddings) and supports efficient approximate nearest-neighbor (ANN) search. It is the backbone of semantic search and RAG systems.

## How Embeddings Work

Text is converted to a fixed-length vector using an embedding model:

```
"What is machine learning?" → [0.021, -0.134, 0.879, ..., 0.042]  (1536 dimensions)
```

Semantically similar texts produce vectors with high cosine similarity, enabling retrieval by meaning rather than exact keyword match.

## Similarity Metrics

| Metric | Formula | Best For |
|---|---|---|
| Cosine similarity | `dot(a,b) / (|a| * |b|)` | Normalized text embeddings |
| Euclidean distance | `sqrt(sum((a-b)^2))` | Image embeddings |
| Dot product | `sum(a * b)` | Fast retrieval when vectors are normalized |

## Popular Vector Databases

### Chroma
- Open source, embedded or client-server
- Perfect for prototypes and local development
- Python-native API

```python
import chromadb

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("my_docs")

collection.add(
    ids=["chunk_1", "chunk_2"],
    documents=["text one", "text two"],
    embeddings=[[0.1, 0.2], [0.3, 0.4]],
    metadatas=[{"source": "doc.pdf"}, {"source": "doc.pdf"}],
)

results = collection.query(query_embeddings=[[0.1, 0.2]], n_results=3)
```

### Pinecone
- Fully managed, cloud-native
- Scales to billions of vectors
- Built-in metadata filtering, namespaces

```python
import pinecone

pc = pinecone.Pinecone(api_key="your-key")
index = pc.Index("my-index")

index.upsert(vectors=[("id1", [0.1, 0.2, ...], {"source": "doc.pdf"})])
results = index.query(vector=[0.1, 0.2, ...], top_k=5, include_metadata=True)
```

### Weaviate
- Open source, GraphQL API
- Built-in BM25 + vector hybrid search
- Multi-modal support

### FAISS (Facebook AI Similarity Search)
- In-memory library (not a database)
- Extremely fast for large-scale ANN search
- No persistence layer — requires wrapping

## Indexing Algorithms

| Algorithm | Speed | Accuracy | Memory |
|---|---|---|---|
| Flat (brute force) | Slow | 100% | Low |
| HNSW | Fast | ~99% | High |
| IVF | Fast | ~95% | Medium |
| PQ (Product Quantization) | Very fast | ~90% | Very low |

Chroma uses **HNSW** by default with cosine space.

## Metadata Filtering

Combine vector similarity with structured filters for precision retrieval:

```python
results = collection.query(
    query_embeddings=[query_vec],
    n_results=5,
    where={"document_type": "pdf"},               # metadata filter
    where_document={"$contains": "machine learning"},  # full-text filter
)
```

## Session / Namespace Isolation

For multi-user systems, isolate each user's documents in separate namespaces or collections:

```python
# Chroma — one collection per session
collection = client.get_or_create_collection(f"session_{session_id}")

# Pinecone — one namespace per user
index.upsert(vectors=vectors, namespace=f"user_{user_id}")
```

## Performance Tuning

- **Top-k** — start with k=5; increase if recall is low
- **MMR (Maximal Marginal Relevance)** — reduces redundancy in results
- **Score threshold** — filter out low-confidence chunks (cosine < 0.7)
- **Re-ranking** — use a cross-encoder to re-rank top-k results by relevance
