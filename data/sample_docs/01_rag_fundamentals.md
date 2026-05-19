# Retrieval-Augmented Generation (RAG) — Fundamentals

## What is RAG?

Retrieval-Augmented Generation (RAG) is a technique that enhances Large Language Model (LLM) responses by grounding them in external, retrieved knowledge. Instead of relying solely on the model's parametric memory, RAG dynamically fetches relevant documents at inference time and provides them as context to the LLM.

## Core Components

### 1. Document Store
A corpus of documents (PDFs, Markdown, HTML, plain text) that serves as the knowledge base. Documents are pre-processed, chunked, and embedded before storage.

### 2. Retriever
Responsible for finding the most relevant chunks for a given query. Retrieval methods include:
- **Semantic search** — cosine similarity over dense vector embeddings
- **Keyword search (BM25)** — sparse TF-IDF-based matching
- **Hybrid search** — weighted combination of semantic and keyword results

### 3. Generator
An LLM that synthesizes a final answer from the retrieved context. Popular choices: GPT-4, Claude, Llama 3, Mistral.

## RAG Pipeline

```
User Query
    ↓
Query Embedding
    ↓
Vector Store Similarity Search
    ↓
Retrieved Chunks (top-k)
    ↓
Prompt Construction (query + context)
    ↓
LLM Generation
    ↓
Final Answer
```

## Chunking Strategies

| Strategy | Chunk Size | Best For |
|---|---|---|
| Fixed-size | 256–512 tokens | General documents |
| Sentence splitter | 1–3 sentences | High-precision Q&A |
| Recursive character | Variable | Code + prose mixed |
| Semantic chunking | Variable | Long-form narratives |

### Chunk Overlap
Overlapping adjacent chunks (typically 10–20% of chunk size) prevents context loss at boundaries.

## Embedding Models

| Model | Dimensions | Notes |
|---|---|---|
| `text-embedding-ada-002` | 1536 | OpenAI, strong general purpose |
| `text-embedding-3-small` | 1536 | OpenAI, cheaper |
| `all-MiniLM-L6-v2` | 384 | Open source, fast |
| `bge-large-en` | 1024 | Open source, high accuracy |

## Metadata Filtering

Filtering retrieved chunks by metadata (e.g., document type, date, section) before similarity ranking dramatically improves precision:

```python
results = collection.query(
    query_embeddings=[query_vec],
    where={"document_type": "pdf"},
    n_results=5,
)
```

## Evaluation Metrics

- **Faithfulness** — Does the answer stay within the retrieved context?
- **Answer Relevancy** — How relevant is the answer to the question?
- **Context Recall** — Were the right chunks retrieved?
- **Context Precision** — Are retrieved chunks free of irrelevant content?
