import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from db.models import Base
from services.session_management import (
    create_session, get_session, add_message,
    get_recent_messages, save_memory, get_user_memories,
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


@pytest.mark.asyncio
async def test_session_and_messages(db):
    session = await create_session(db, user_id="u1")
    assert session.id is not None

    await add_message(db, session.id, "user", "Hello")
    await add_message(db, session.id, "assistant", "Hi!")

    msgs = await get_recent_messages(db, session.id)
    assert len(msgs) == 2
    assert msgs[0].content == "Hello"

    updated = await get_session(db, session.id)
    assert updated.turn_count == 2


@pytest.mark.asyncio
async def test_memory(db):
    await save_memory(db, "u1", "preference", "Likes Python examples", importance=8)
    await save_memory(db, "u1", "fact", "Works on FastAPI", importance=5)

    mems = await get_user_memories(db, "u1")
    assert len(mems) == 2
    assert mems[0].importance == 8