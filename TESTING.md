# Testing Guide

## Requirements

`pytest`, `pytest-asyncio`, and `pytest-cov` are included in `requirements.txt`.

```bash
uv pip install -r requirements.txt
```

---

## Running Tests

```bash
# Run all tests with coverage (enforced at ≥70%)
uv run python -m pytest

# Short summary — no per-test output
uv run python -m pytest --tb=no -q

# Verbose — show each test name
uv run python -m pytest -v

# Run a specific module
uv run python -m pytest test/module_test_1.py -v

# Run a specific test by name
uv run python -m pytest -k "test_memory" -v

# Generate HTML coverage report
uv run python -m pytest --cov=. --cov-report=html
# Then open: htmlcov/index.html
```

---

## Coverage Results

```
Name                             Stmts   Miss  Cover
----------------------------------------------------
config.py                           22      0   100%
db/models.py                        41      0   100%
db/sql_database.py                  12      0   100%
db/vector_database.py               51      0   100%
llm/schema.py                       81      0   100%
services/document_chunker.py        25      0   100%
services/document_ingestion.py      21      0   100%
services/session_management.py      70      0   100%
services/document_parser.py         16      1    94%
routes/memories.py                  10      2    80%
main.py                             20      2    90%
routes/session.py                   23      6    74%
routes/document.py                  18      6    67%
routes/chat.py                      78     57    27%
llm/agents.py                      205    132    36%
----------------------------------------------------
TOTAL                              703    208    70%
```

**70.5% total — 48 passed, 1 skipped**

> `llm/agents.py` is at 36% because all agent functions make live OpenAI API calls. The integration test (`test_integration_full_pipeline`) covers this path but is skipped unless `OPENAI_API_KEY` is set.

---

## Test Modules

### `module_test_1.py` — Session & Memory CRUD
Tests the core database layer without any external dependencies.

| Test | What it verifies |
|---|---|
| `test_session_and_messages` | Create session, add messages, verify turn count increments |
| `test_memory` | Save preference and fact memories, retrieve them |

---

### `module_test_2.py` — Document Processing
Tests the document ingestion pipeline (parse → chunk → metadata).

| Test | What it verifies |
|---|---|
| `test_parse_markdown` | Markdown file parsed into LangChain `Document` objects |
| `test_parse_html` | HTML file parsed correctly |
| `test_unsupported_extension_raises` | `ValueError` raised for unsupported file types |
| `test_chunk_document` | Document split into chunks, `document_id` preserved |
| `test_chunk_preserves_metadata` | `source_file`, `page_number` carried through |
| `test_metadata_extraction` | Section headers and code block detection |
| `test_metadata_no_header` | Graceful handling of documents without headers |

---

### `module_test_3.py` — Agent Routing Logic
Tests the LangGraph routing decisions without making LLM calls (except one mocked test).

| Test | What it verifies |
|---|---|
| `test_router_factual` | `FACTUAL` intent → `SEMANTIC` strategy |
| `test_router_conversational` | `CONVERSATIONAL` intent → `MEMORY_ONLY` strategy |
| `test_router_multi_part` | `MULTI_PART` intent → `HYBRID` strategy |
| `test_router_follow_up` | `FOLLOW_UP` intent → `SEMANTIC` strategy |
| `test_route_retrieval_memory_only` | Router returns `"memory_only"` edge |
| `test_route_retrieval_hybrid` | Router returns `"query_decomposition"` edge |
| `test_route_retrieval_semantic` | Router returns `"semantic_retrieval"` edge |
| `test_query_understanding_factual` | Mocked LLM — verifies intent parsing |
| `test_query_rewriting_skips_without_history` | Rewriter short-circuits when `needs_history=False` |
| `test_integration_full_pipeline` *(skipped)* | Full end-to-end pipeline — requires `OPENAI_API_KEY` |

---

### `module_test_4.py` — API Endpoints
Tests all FastAPI routes using an in-memory SQLite database and mocked pipeline.

| Test | What it verifies |
|---|---|
| `test_health` | `GET /health` returns `{"status": "ok"}` |
| `test_create_session` | `POST /sessions` creates session and returns `session_id` |
| `test_get_session` | `GET /sessions/{id}` returns correct session |
| `test_get_session_not_found` | `GET /sessions/bad_id` returns 404 |
| `test_list_sessions` | `GET /sessions` returns all sessions for a user |
| `test_chat` | `POST /chat` with mocked pipeline returns answer |
| `test_chat_invalid_session` | `POST /chat` with bad session_id returns 404 |
| `test_upload_unsupported_file` | `POST /documents/upload` with `.txt` returns 400 |

---

### `module_test_5.py` — Session Management Extended
Tests the full `session_management.py` including memory pruning and agent traces.

| Test | What it verifies |
|---|---|
| `test_list_sessions_empty` | Empty list returned for user with no sessions |
| `test_list_sessions_with_preview` | Preview taken from first user message |
| `test_list_sessions_no_messages` | Defaults to `"New conversation"` |
| `test_update_session_summary` | Summary saved and persisted to DB |
| `test_save_memory_deduplication` | Duplicate memory skipped, only 1 entry stored |
| `test_save_memory_pruning` | Oldest entries deleted when count exceeds `MEMORY_MAX_ENTRIES` |
| `test_save_and_get_agent_trace` | Trace saved and retrieved with correct fields |
| `test_get_session_traces_empty` | Empty list for session with no traces |
| `test_ingest_document` | `ingest_document` calls `add_chunks` with correct args (mocked) |

---

### `module_test_6.py` — Vector Database & Infrastructure
Tests `db/vector_database.py` fully with mocked Chroma client and embeddings.

| Test | What it verifies |
|---|---|
| `test_get_embeddings_singleton` | `get_embeddings()` returns same instance on repeated calls |
| `test_get_client_persistent` | Uses `PersistentClient` when `CHROMA_HOST` is empty |
| `test_get_client_http` | Uses `HttpClient` when `CHROMA_HOST` is set |
| `test_get_collection` | Collection created once, cached on repeated calls |
| `test_embed_texts` | `embed_texts()` calls `aembed_documents` and returns vectors |
| `test_add_chunks_empty` | Returns 0 immediately for empty chunk list |
| `test_add_chunks` | Embeds and stores chunks, calls `collection.add` |
| `test_keyword_search_no_keywords` | Returns `[]` when all words are ≤3 characters |
| `test_keyword_search` | Queries Chroma with keyword filter, maps results correctly |
| `test_search` | Embeds query, queries Chroma, maps distances to scores |
| `test_init_db` | `init_db()` creates tables on a test engine |
| `test_get_db` | `get_db()` yields a valid async session |
| `test_ingest_directory` | Scans directory, ingests supported files, skips unsupported |

---

## Configuration

Coverage is configured in `.coveragerc` and pytest in `pytest.ini`:

```ini
# pytest.ini
[pytest]
asyncio_mode = auto
testpaths = test
python_files = module_test_*.py test_*.py
addopts = --cov=. --cov-report=term-missing --cov-fail-under=70
```

```ini
# .coveragerc
[run]
source = .
omit =
    test/*
    scripts/*
    streamlit_ui.py
    .venv/*
    */site-packages/*

[report]
show_missing = true
fail_under = 70
```
