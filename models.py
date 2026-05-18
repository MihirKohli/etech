from datetime import datetime
from uuid6 import uuid7

from sqlalchemy import Column, String, Text, DateTime, Integer, ForeignKey
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, relationship

from config import get_settings


class Base(DeclarativeBase):
    pass


class Session(Base):
    __tablename__ = "sessions"

    id = Column(String, primary_key=True, default= uuid7().hex)
    user_id = Column(String, nullable=False, index=True)
    created_at = Column(DateTime, default= datetime.now())
    summary = Column(Text, default="")
    turn_count = Column(Integer, default=0)
    messages = relationship("Message", back_populates="session", order_by="Message.created_at")


class Message(Base):
    __tablename__ = "messages"

    id = Column(String, primary_key=True, default= uuid7().hex)
    session_id = Column(String, ForeignKey("sessions.id"), nullable=False, index=True)
    role = Column(String, nullable=False)       # "user" | "assistant"
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default= datetime.now())
    session = relationship("Session", back_populates="messages")


class ConversationMemory(Base):
    """Cross-session long-term memory per user."""
    __tablename__ = "conversation_memory"

    id = Column(String, primary_key=True, default= uuid7().hex)
    user_id = Column(String, nullable=False, index=True)
    memory_type = Column(String, nullable=False)   # "preference" | "fact" | "summary"
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default= datetime.now())


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