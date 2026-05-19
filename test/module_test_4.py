"""
Module 4 tests — API endpoints.

Tests mock the LangGraph pipeline so no API key is needed.

Run:  pytest tests/test_module4.py -v
"""

import pytest
import pytest_asyncio
from unittest.mock import patch
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from db.sql_database import Base, get_db
from main import app


# ── Test DB fixture ──────────────────────────────────

@pytest_asyncio.fixture
async def db_session():
    engine = create_async_engine("sqlite+aiosqlite://")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as session:
        yield session
    await engine.dispose()


@pytest_asyncio.fixture
async def client(db_session):
    """Test client with overridden DB dependency."""
    async def override_db():
        yield db_session

    app.dependency_overrides[get_db] = override_db
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
    app.dependency_overrides.clear()


# ── Health ───────────────────────────────────────────

@pytest.mark.asyncio
async def test_health(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


# ── Sessions ─────────────────────────────────────────

@pytest.mark.asyncio
async def test_create_session(client):
    resp = await client.post("/api/sessions?user_id=user1")
    assert resp.status_code == 200
    data = resp.json()
    assert "session_id" in data
    assert data["user_id"] == "user1"


@pytest.mark.asyncio
async def test_get_session(client):
    # Create first
    resp = await client.post("/api/sessions?user_id=user1")
    sid = resp.json()["session_id"]

    # Get
    resp = await client.get(f"/api/sessions/{sid}")
    assert resp.status_code == 200
    assert resp.json()["session_id"] == sid


@pytest.mark.asyncio
async def test_get_session_not_found(client):
    resp = await client.get("/api/sessions/nonexistent")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_list_sessions(client):
    await client.post("/api/sessions?user_id=user1")
    await client.post("/api/sessions?user_id=user1")

    resp = await client.get("/api/sessions?user_id=user1")
    assert resp.status_code == 200
    assert len(resp.json()) == 2


# ── Chat (mocked pipeline) ──────────────────────────

@pytest.mark.asyncio
async def test_chat(client):
    # Create session
    resp = await client.post("/api/sessions?user_id=user1")
    sid = resp.json()["session_id"]

    # Mock the pipeline + memory extraction
    mock_result = {
        "answer": "FastAPI is a modern Python web framework.",
        "sources": [],
    }

    with patch("app.api.routes.run_pipeline", return_value=mock_result):
        with patch("app.api.routes.extract_memories", return_value=[]):
            resp = await client.post("/api/chat", json={
                "session_id": sid,
                "message": "What is FastAPI?",
            })

    assert resp.status_code == 200
    data = resp.json()
    assert data["session_id"] == sid
    assert "FastAPI" in data["answer"]


@pytest.mark.asyncio
async def test_chat_invalid_session(client):
    with patch("app.api.routes.run_pipeline", return_value={"answer": "x", "sources": []}):
        resp = await client.post("/api/chat", json={
            "session_id": "bad_id",
            "message": "hello",
        })
    assert resp.status_code == 404


# ── Document Upload ──────────────────────────────────

@pytest.mark.asyncio
async def test_upload_unsupported_file(client):
    resp = await client.post(
        "/api/documents/upload",
        files={"file": ("test.txt", b"hello", "text/plain")},
    )
    assert resp.status_code == 400