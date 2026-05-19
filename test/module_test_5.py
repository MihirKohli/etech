"""
Module 5 tests — session_management extended + document_ingestion.

Covers: list_sessions, update_session_summary, save_agent_trace,
get_session_traces, memory deduplication, memory pruning,
and ingest_document (with mocked vector store).
"""

import pytest
import pytest_asyncio
from unittest.mock import patch, AsyncMock
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from db.models import Base
from services.session_management import (
    create_session, add_message, list_sessions,
    update_session_summary, save_memory, get_user_memories,
    save_agent_trace, get_session_traces,
)


@pytest_asyncio.fixture
async def db():
    engine = create_async_engine("sqlite+aiosqlite://")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as session:
        yield session
    await engine.dispose()


# ── list_sessions ────────────────────────────────────

@pytest.mark.asyncio
async def test_list_sessions_empty(db):
    rows = await list_sessions(db, "no_user")
    assert rows == []


@pytest.mark.asyncio
async def test_list_sessions_with_preview(db):
    s = await create_session(db, "u1")
    await add_message(db, s.id, "user", "Hello world from test")
    rows = await list_sessions(db, "u1")
    assert len(rows) == 1
    assert rows[0]["preview"] == "Hello world from test"


@pytest.mark.asyncio
async def test_list_sessions_no_messages(db):
    await create_session(db, "u2")
    rows = await list_sessions(db, "u2")
    assert rows[0]["preview"] == "New conversation"


# ── update_session_summary ───────────────────────────

@pytest.mark.asyncio
async def test_update_session_summary(db):
    s = await create_session(db, "u1")
    await update_session_summary(db, s.id, "User asked about Python.")
    refreshed = await db.get(type(s), s.id)
    assert refreshed.summary == "User asked about Python."


# ── memory deduplication & pruning ──────────────────

@pytest.mark.asyncio
async def test_save_memory_deduplication(db):
    await save_memory(db, "u1", "preference", "Likes Python")
    result = await save_memory(db, "u1", "preference", "Likes Python")
    assert result is None
    mems = await get_user_memories(db, "u1")
    assert len(mems) == 1


@pytest.mark.asyncio
async def test_save_memory_pruning(db):
    from config import get_settings
    cap = get_settings().MEMORY_MAX_ENTRIES
    for i in range(cap + 5):
        await save_memory(db, "prune_user", "fact", f"Unique fact number {i}")
    mems = await get_user_memories(db, "prune_user", limit=cap + 10)
    assert len(mems) <= cap


# ── save_agent_trace / get_session_traces ────────────

@pytest.mark.asyncio
async def test_save_and_get_agent_trace(db):
    s = await create_session(db, "u1")
    trace = await save_agent_trace(
        db,
        session_id=s.id,
        query_intent="factual",
        retrieval_strategy="hybrid",
        rewritten_query="What is RAG?",
        sub_questions=["What is retrieval?", "What is generation?"],
        nodes_visited=None,
        response_time_ms=1234.5,
    )
    assert trace.id is not None
    assert trace.query_intent == "factual"

    traces = await get_session_traces(db, s.id)
    assert len(traces) == 1
    assert traces[0].response_time_ms == 1234.5


@pytest.mark.asyncio
async def test_get_session_traces_empty(db):
    s = await create_session(db, "u1")
    traces = await get_session_traces(db, s.id)
    assert traces == []


# ── ingest_document (mocked) ─────────────────────────

@pytest.mark.asyncio
async def test_ingest_document(tmp_path):
    md_file = tmp_path / "test.md"
    md_file.write_text("# Test\nThis is a test document for ingestion.")

    with patch("services.document_ingestion.add_chunks", new=AsyncMock(return_value=3)):
        from services.document_ingestion import ingest_document
        result = await ingest_document(str(md_file), session_id="sess1")

    assert result["filename"] == "test.md"
    assert result["chunks_created"] == 3
    assert "document_id" in result
