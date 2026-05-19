"""
Module 6 tests — vector_database and document_ingestion (mocked chromadb/embeddings).
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch


# ── Helpers ──────────────────────────────────────────

def _make_chroma_result(ids, docs, metas, distances):
    return {
        "ids": [ids],
        "documents": [docs],
        "metadatas": [metas],
        "distances": [distances],
    }


# ── vector_database ──────────────────────────────────

@pytest.mark.asyncio
async def test_get_embeddings_singleton():
    import db.vector_database as vdb
    vdb._embeddings = None
    with patch("db.vector_database.OpenAIEmbeddings") as mock_cls:
        mock_cls.return_value = MagicMock()
        e1 = vdb.get_embeddings()
        e2 = vdb.get_embeddings()
        assert e1 is e2
        mock_cls.assert_called_once()
    vdb._embeddings = None


@pytest.mark.asyncio
async def test_get_client_persistent(tmp_path):
    import db.vector_database as vdb
    vdb._client = None
    with patch("db.vector_database.settings") as mock_settings, \
         patch("db.vector_database.chromadb.PersistentClient") as mock_pc:
        mock_settings.CHROMA_HOST = ""
        mock_settings.CHROMA_PERSIST_DIR = str(tmp_path)
        mock_pc.return_value = MagicMock()
        client = vdb._get_client()
        assert client is mock_pc.return_value
    vdb._client = None


@pytest.mark.asyncio
async def test_get_client_http():
    import db.vector_database as vdb
    vdb._client = None
    with patch("db.vector_database.settings") as mock_settings, \
         patch("db.vector_database.chromadb.HttpClient") as mock_hc:
        mock_settings.CHROMA_HOST = "chroma"
        mock_settings.CHROMA_PORT = 8000
        mock_hc.return_value = MagicMock()
        client = vdb._get_client()
        assert client is mock_hc.return_value
        mock_hc.assert_called_once_with(host="chroma", port=8000)
    vdb._client = None


@pytest.mark.asyncio
async def test_get_collection():
    import db.vector_database as vdb
    vdb._client = None
    vdb._collections = {}
    mock_collection = MagicMock()
    mock_client = MagicMock()
    mock_client.get_or_create_collection.return_value = mock_collection

    with patch("db.vector_database._get_client", return_value=mock_client):
        col = vdb.get_collection("sess_abc")
        assert col is mock_collection
        col2 = vdb.get_collection("sess_abc")
        assert col2 is mock_collection
        mock_client.get_or_create_collection.assert_called_once()
    vdb._collections = {}


@pytest.mark.asyncio
async def test_embed_texts():
    import db.vector_database as vdb
    mock_embeddings = MagicMock()
    mock_embeddings.aembed_documents = AsyncMock(return_value=[[0.1, 0.2], [0.3, 0.4]])

    with patch("db.vector_database.get_embeddings", return_value=mock_embeddings):
        result = await vdb.embed_texts(["hello", "world"])
    assert result == [[0.1, 0.2], [0.3, 0.4]]


@pytest.mark.asyncio
async def test_add_chunks_empty():
    import db.vector_database as vdb
    count = await vdb.add_chunks([], "sess1")
    assert count == 0


@pytest.mark.asyncio
async def test_add_chunks():
    import db.vector_database as vdb
    chunks = [
        {"chunk_id": "c1", "content": "hello", "metadata": {"source": "f.md"}},
        {"chunk_id": "c2", "content": "world", "metadata": {"source": "f.md"}},
    ]
    mock_col = MagicMock()

    with patch("db.vector_database.get_collection", return_value=mock_col), \
         patch("db.vector_database.embed_texts", new=AsyncMock(return_value=[[0.1], [0.2]])):
        count = await vdb.add_chunks(chunks, "sess1")

    assert count == 2
    mock_col.add.assert_called_once()


@pytest.mark.asyncio
async def test_keyword_search_no_keywords():
    import db.vector_database as vdb
    result = await vdb.keyword_search("to a", "sess1")
    assert result == []


@pytest.mark.asyncio
async def test_keyword_search():
    import db.vector_database as vdb
    chroma_result = _make_chroma_result(
        ["c1"], ["some content"], [{"source": "f.md"}], [0.5]
    )
    mock_col = MagicMock()
    mock_col.query.return_value = chroma_result

    with patch("db.vector_database.get_collection", return_value=mock_col):
        hits = await vdb.keyword_search("python async programming", "sess1", top_k=3)

    assert len(hits) == 1
    assert hits[0]["chunk_id"] == "c1"
    assert hits[0]["score"] == 0.5


@pytest.mark.asyncio
async def test_search():
    import db.vector_database as vdb
    chroma_result = _make_chroma_result(
        ["c1"], ["content"], [{"source": "f.md"}], [0.2]
    )
    mock_col = MagicMock()
    mock_col.query.return_value = chroma_result
    mock_embeddings = MagicMock()
    mock_embeddings.aembed_query = AsyncMock(return_value=[0.1, 0.2])

    with patch("db.vector_database.get_collection", return_value=mock_col), \
         patch("db.vector_database.get_embeddings", return_value=mock_embeddings):
        hits = await vdb.search("what is async", "sess1", top_k=3)

    assert len(hits) == 1
    assert hits[0]["score"] == pytest.approx(0.8)


# ── sql_database ─────────────────────────────────────

@pytest.mark.asyncio
async def test_init_db():
    from sqlalchemy.ext.asyncio import create_async_engine
    from db.sql_database import init_db
    test_engine = create_async_engine("sqlite+aiosqlite://")
    with patch("db.sql_database.engine", test_engine):
        await init_db()


@pytest.mark.asyncio
async def test_get_db():
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
    from db.models import Base
    from db.sql_database import get_db
    test_engine = create_async_engine("sqlite+aiosqlite://")
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(test_engine, class_=AsyncSession)
    with patch("db.sql_database.async_session_factory", factory):
        async for session in get_db():
            assert session is not None


# ── ingest_directory ─────────────────────────────────

@pytest.mark.asyncio
async def test_ingest_directory(tmp_path):
    (tmp_path / "doc1.md").write_text("# Doc1\nContent one.")
    (tmp_path / "doc2.md").write_text("# Doc2\nContent two.")
    (tmp_path / "skip.txt").write_text("not supported")

    with patch("services.document_ingestion.add_chunks", new=AsyncMock(return_value=2)):
        from services.document_ingestion import ingest_directory
        results = await ingest_directory(str(tmp_path), session_id="sess1")

    assert len(results) == 2
    assert all(r["chunks_created"] == 2 for r in results)
