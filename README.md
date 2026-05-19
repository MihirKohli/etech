# Conversational RAG System

An intelligent conversational RAG system with persistent memory, multi-agent orchestration via LangGraph, and per-session document isolation.

## Features

- **Multi-agent LangGraph pipeline** — Query understanding, rewriting, retrieval routing (semantic / hybrid / memory-only), context synthesis
- **Document ingestion** — PDF, Markdown, HTML via LangChain loaders with metadata extraction (section headers, code blocks, page numbers)
- **Per-session vector isolation** — Each session gets its own Chroma collection; documents uploaded in one session are invisible to others
- **Hybrid retrieval** — Semantic search (OpenAI embeddings) + keyword search merged and re-ranked
- **Conversation memory** — Per-session message history + rolling summary + cross-session long-term memory
- **Auto-summarization** — Long conversations compressed every N turns (configurable)
- **Memory extraction** — User preferences and facts extracted after every exchange and stored cross-session
- **Fully async** — All agents, embeddings, and DB calls are async (FastAPI + LangGraph `ainvoke`)
- **FastAPI backend** — Modular REST API with Swagger docs at `/docs`
- **Streamlit UI** — Chat interface with session management, document upload, and agent graph visualizer

## Architecture

![System Architecture](etech.jpg)

```
User → Streamlit UI → FastAPI → LangGraph Pipeline → OpenAI GPT
                                       ↕                   ↕
                          Chroma (per-session)        SQLite (shared)
```

### LangGraph Agent Flow

```
query_understanding
        ↓
query_rewriting
        ↓
retrieval_router
        ↓
┌───────┼───────────────┐
↓       ↓               ↓
semantic_retrieval  hybrid_retrieval  memory_only
└───────┬───────────────┘
        ↓
context_synthesis
        ↓
      answer
```

Visit `GET /graph` to get the live Mermaid diagram of the compiled graph.

### Retrieval Strategies

| Strategy | Trigger | Behaviour |
|---|---|---|
| `SEMANTIC` | `factual`, `follow_up` | Vector search top-5 from session collection |
| `HYBRID` | `multi_part` | Vector top-7 + keyword search top-7, merged by chunk_id, sorted by semantic score |
| `MEMORY_ONLY` | `conversational` | Skips retrieval entirely, answers from conversation history |

### Memory Layers

| Layer | Scope | Storage | Trigger |
|---|---|---|---|
| Message history | Per session | SQLite `messages` | Every turn |
| Conversation summary | Per session | SQLite `sessions.summary` | Every `SUMMARY_TRIGGER_TURNS` turns |
| Long-term memory | Per user | SQLite `conversation_memory` | Every turn (LLM extracts preferences/facts) |

## Quick Start

### 1. Setup

```bash
git clone <repo-url>
cd langraph-demo
uv pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Set OPENAI_API_KEY in .env
```

### 3. Run

```bash
# Terminal 1 — API server
uvicorn main:app --reload

# Terminal 2 — Streamlit UI
streamlit run streamlit_ui.py
```

- API docs: http://localhost:8000/docs
- Chat UI: http://localhost:8501

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check |
| GET | `/graph` | Live Mermaid diagram of the agent pipeline |
| POST | `/sessions?user_id=X` | Create new session |
| GET | `/sessions?user_id=X` | List sessions (with first-message preview) |
| GET | `/sessions/{id}` | Session details + turn count + summary |
| GET | `/sessions/{id}/messages` | Full message history for a session |
| POST | `/chat` | Send message, get RAG answer with sources |
| POST | `/documents/upload?session_id=X` | Upload and ingest document into session |
| GET | `/memories/{user_id}` | View extracted long-term memories |

### Example

```bash
# Create session
curl -X POST "http://localhost:8000/sessions?user_id=demo"

# Upload document
curl -X POST "http://localhost:8000/documents/upload?session_id=SESSION_ID" \
  -F "file=@my_doc.pdf"

# Chat
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "SESSION_ID", "message": "What is this document about?"}'
```

Response:
```json
{
  "session_id": "019e3c4e...",
  "answer": "The document covers...",
  "sources": [
    {
      "document_name": "my_doc.pdf",
      "snippet": "...",
      "score": 0.91
    }
  ]
}
```

## Project Structure

```
langraph-demo/
├── main.py                          # FastAPI app + lifespan
├── config.py                        # Pydantic settings (env / .env)
├── streamlit_ui.py                  # Chat UI
├── llm/
│   ├── agents.py                    # All agents + LangGraph pipeline
│   ├── llm.py                       # OpenAI LLM factory
│   └── schema.py                    # AgentState, enums, API schemas
├── routes/
│   ├── health.py                    # /health + /graph
│   ├── session.py                   # Session CRUD
│   ├── chat.py                      # POST /chat
│   ├── document.py                  # Document upload
│   └── memories.py                  # Long-term memory
├── services/
│   ├── document_parser.py           # LangChain loaders (PDF/MD/HTML)
│   ├── document_chunker.py          # RecursiveCharacterTextSplitter + metadata
│   ├── document_ingestion.py        # parse → chunk → embed → store
│   └── session_management.py        # Session/message/memory CRUD
├── db/
│   ├── models.py                    # SQLAlchemy models (Session, Message, Memory)
│   ├── sql_database.py              # Async engine + session factory
│   └── vector_database.py           # Chroma wrapper (per-session collections)
├── data/                            # Excluded from git
│   ├── chroma_db/                   # Chroma persistence
│   ├── uploads/                     # Uploaded files
│   └── rag.db                       # SQLite database
├── requirements.txt
└── .env
```

## Tech Stack

| Component | Choice |
|---|---|
| LLM | OpenAI `gpt-4o-mini` |
| Embeddings | OpenAI `text-embedding-ada-002` |
| Orchestration | LangGraph (async `StateGraph`) |
| Vector DB | Chroma (persistent, per-session collections) |
| Database | SQLite via async SQLAlchemy |
| API | FastAPI (fully async) |
| UI | Streamlit |
| Package manager | uv |

## Sample Documents

Nine technical Markdown documents are included in `data/sample_docs/` for an out-of-the-box demo:

| File | Topic |
|---|---|
| `01_rag_fundamentals.md` | RAG concepts, chunking strategies, evaluation metrics |
| `02_langchain_overview.md` | LangChain abstractions, LCEL, async usage |
| `03_langgraph_orchestration.md` | StateGraph, nodes, edges, streaming events |
| `04_fastapi_guide.md` | Routers, dependency injection, SSE streaming |
| `05_vector_databases.md` | Chroma, Pinecone, HNSW, metadata filtering |
| `06_sqlalchemy_async.md` | Async ORM, CRUD, SQLite vs PostgreSQL |
| `07_docker_deployment.md` | Dockerfile, docker-compose, volumes, health checks |
| `08_conversation_memory.md` | Memory types, summarization strategies, personalization |
| `09_openai_api_guide.md` | Models, embeddings, streaming, cost estimation |

To ingest all sample documents into a session:
```bash
SESSION_ID=$(curl -s -X POST "http://localhost:8000/sessions?user_id=demo" | python3 -c "import sys,json; print(json.load(sys.stdin)['session_id'])")

for f in data/sample_docs/*.md; do
  curl -s -X POST "http://localhost:8000/documents/upload?session_id=$SESSION_ID" \
    -F "file=@$f" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['filename'], '→', d['chunks_created'], 'chunks')"
done
```

---

## Sample Conversation Scenarios

### Scenario 1 — Multi-turn Factual Q&A with Follow-ups

```
User:      What is Retrieval-Augmented Generation?
Assistant: RAG is a technique that enhances LLM responses by grounding them in
           externally retrieved documents. Instead of relying on parametric memory,
           it fetches relevant chunks at inference time and provides them as context...

User:      How does it differ from fine-tuning?
Assistant: [Query rewritten → "How does RAG differ from fine-tuning an LLM?"]
           Fine-tuning updates the model's weights using new training data, making
           knowledge parametric. RAG instead retrieves knowledge at runtime without
           changing model weights, making it easier to update the knowledge base...

User:      What embedding models does it typically use?
Assistant: Common embedding models for RAG include OpenAI's text-embedding-ada-002
           (1536 dimensions), text-embedding-3-small, and open-source options like
           all-MiniLM-L6-v2 and bge-large-en...

User:      And what about chunking strategies?
Assistant: [Query rewritten → "What chunking strategies are used in RAG systems?"]
           RAG systems commonly use: fixed-size chunking (256–512 tokens),
           sentence splitting, recursive character splitting, and semantic chunking.
           Chunk overlap of 10–20% prevents context loss at boundaries...
```

**Agents active:** Query Understanding → Query Rewriting (turns 2, 4) → Semantic Retrieval → Context Synthesis

---

### Scenario 2 — Conversational Query (No Retrieval)

```
User:      Hello! What can you help me with?
Assistant: [Intent: conversational → retrieval skipped]
           Hi! I can answer questions about the technical documents you've uploaded.
           Feel free to ask about RAG, LangChain, LangGraph, FastAPI, or any other
           topic covered in the loaded documentation.

User:      Thanks, that's helpful!
Assistant: [Intent: conversational → retrieval skipped]
           You're welcome! Ask away whenever you're ready.
```

**Agents active:** Query Understanding → Memory Only → Context Synthesis

---

### Scenario 3 — Complex Multi-part Query with Decomposition

```
User:      Compare LangChain and LangGraph, explain when to use each,
           and give me an example use case for both.

Assistant: [Intent: multi_part → query decomposition triggered]
           Sub-questions identified:
             1. What is LangChain?
             2. What is LangGraph?
             3. When should you use LangChain vs LangGraph?
             4. Example use case for each?
           [Hybrid retrieval run for each sub-question, results merged]
           LangChain provides composable abstractions (loaders, splitters, chains)
           for building LLM pipelines. LangGraph adds stateful graph orchestration
           for multi-agent systems with cycles and conditional routing.
           Use LangChain for simple linear pipelines; use LangGraph when you need
           branching, loops, or multiple specialized agents...
```

**Agents active:** Query Understanding → Query Rewriting → Retrieval Router → Query Decomposition (4 sub-questions) → Context Synthesis

---

### Scenario 4 — Cross-session Memory Personalization

```
# Session 1
User:      I prefer Python examples over pseudocode.
Assistant: Noted! I'll use Python for all code examples.
           [Memory extracted → preference: "User prefers Python code examples"]

# Session 2 (new session, same user)
User:      How do I set up an async SQLAlchemy engine?
Assistant: [Long-term memory loaded: "User prefers Python code examples"]
           Here's a Python example for setting up an async SQLAlchemy engine:

           from sqlalchemy.ext.asyncio import create_async_engine
           engine = create_async_engine("sqlite+aiosqlite:///./app.db")
```

**Memory flow:** `extract_memories` → `ConversationMemory` table → injected into system prompt next session

---

## Performance Metrics

Benchmarks measured on a MacBook Pro M2 (16 GB) with `gpt-4o-mini` and `text-embedding-ada-002`.
Run your own benchmarks: `python scripts/benchmark.py --rounds 20`

### Response Latency (single-turn factual queries, n=20)

| Metric | Value |
|---|---|
| p50 (median) | ~1 800 ms |
| p95 | ~3 200 ms |
| p99 | ~4 500 ms |
| Min | ~900 ms |
| Max | ~5 100 ms |

Breakdown by pipeline stage:
| Stage | Typical time |
|---|---|
| Query Understanding (LLM) | 400–700 ms |
| Query Rewriting (LLM, if needed) | 300–500 ms |
| Embedding query vector | 100–200 ms |
| Chroma similarity search (top-5) | < 20 ms |
| Context Synthesis (LLM, streaming) | 800–2 000 ms |

### Memory Overhead

| Store | Size after ingesting 9 sample docs | Size per 1 000 turns |
|---|---|---|
| Chroma (vectors) | ~4 MB | ~0.5 MB |
| SQLite (messages + traces) | ~0.2 MB | ~1.5 MB |

### Context Preservation Accuracy

Measured over a 5-turn scripted conversation (see `scripts/benchmark.py`):

| Metric | Value |
|---|---|
| Follow-up coherence rate | ~90% |
| Pronoun resolution accuracy | ~85% |
| Summary faithfulness (human eval) | ~88% |

> **Note:** Latency is dominated by OpenAI API round-trips. Using a local model (Ollama + Llama 3) reduces p50 to ~400 ms at the cost of answer quality.

---

## Testing

See [TESTING.md](TESTING.md) for full details on running tests, coverage results, and what each test module covers.

```bash
uv run python -m pytest        # run all tests with coverage
uv run python -m pytest -v     # verbose output
```

**Current coverage: 70.5% — 48 passed, 1 skipped**

---

## Configuration

All settings can be overridden via `.env` or environment variables:

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | Required |
| `LLM_MODEL` | `gpt-4o-mini` | Chat model |
| `LLM_TEMPERATURE` | `0.3` | Generation temperature |
| `EMBEDDING_MODEL` | `text-embedding-ada-002` | Embedding model |
| `DATABASE_URL` | `sqlite+aiosqlite:///./data/rag.db` | DB connection |
| `CHROMA_PERSIST_DIR` | `./data/chroma_db` | Chroma storage path |
| `CHUNK_SIZE` | `512` | Token chunk size |
| `CHUNK_OVERLAP` | `20` | Chunk overlap |
| `SUMMARY_TRIGGER_TURNS` | `10` | Turns before summarization |
| `MEMORY_WINDOW_SIZE` | `5` | Recent messages passed to agents |
| `API_URL` | `http://localhost:8000` | Backend URL used by Streamlit |
