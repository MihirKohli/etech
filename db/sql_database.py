
from config import get_settings
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from db.models import Base
# ── Engine setup ─────────────────────────────────────

settings = get_settings()

engine = create_async_engine(
    settings.DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in settings.DATABASE_URL else {},
)

async_session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db():
    async with async_session_factory() as session:
        yield session