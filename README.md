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
