# SQLAlchemy Async — Persistent Storage for AI Applications

## Overview

SQLAlchemy is Python's most popular ORM. Its async extension (`sqlalchemy.ext.asyncio`) enables non-blocking database operations, essential for high-throughput async web applications like FastAPI backends.

## Setup

```python
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

# SQLite (development)
engine = create_async_engine("sqlite+aiosqlite:///./app.db")

# PostgreSQL (production)
# engine = create_async_engine("postgresql+asyncpg://user:pass@host/db")

async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
```

## Model Definition

```python
from datetime import datetime
from sqlalchemy import Column, String, Text, DateTime, Integer, ForeignKey
from sqlalchemy.orm import relationship

class Base(DeclarativeBase):
    pass

class Session(Base):
    __tablename__ = "sessions"

    id = Column(String, primary_key=True)
    user_id = Column(String, nullable=False, index=True)
    summary = Column(Text, default="")
    turn_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.now)
    messages = relationship("Message", back_populates="session")

class Message(Base):
    __tablename__ = "messages"

    id = Column(String, primary_key=True)
    session_id = Column(String, ForeignKey("sessions.id"), nullable=False)
    role = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.now)
    session = relationship("Session", back_populates="messages")
```

## Table Initialization

```python
async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
```

## CRUD Operations

```python
# Create
async def create_session(db: AsyncSession, user_id: str) -> Session:
    session = Session(id=uuid4().hex, user_id=user_id)
    db.add(session)
    await db.commit()
    await db.refresh(session)
    return session

# Read
async def get_session(db: AsyncSession, session_id: str) -> Session | None:
    return await db.get(Session, session_id)

# Query
from sqlalchemy import select

async def get_messages(db: AsyncSession, session_id: str) -> list[Message]:
    result = await db.execute(
        select(Message)
        .where(Message.session_id == session_id)
        .order_by(Message.created_at)
        .limit(20)
    )
    return list(result.scalars().all())

# Update
async def update_summary(db: AsyncSession, session_id: str, summary: str):
    session = await db.get(Session, session_id)
    if session:
        session.summary = summary
        await db.commit()
```

## Dependency Injection with FastAPI

```python
from fastapi import Depends

async def get_db():
    async with async_session() as session:
        yield session

@router.get("/sessions/{id}")
async def read_session(id: str, db: AsyncSession = Depends(get_db)):
    return await get_session(db, id)
```

## SQLite vs PostgreSQL

| Feature | SQLite | PostgreSQL |
|---|---|---|
| Setup | Zero config | Requires server |
| Concurrent writes | Limited (WAL mode) | Excellent |
| Production ready | Small scale only | Yes |
| Async driver | `aiosqlite` | `asyncpg` |
| Connection string | `sqlite+aiosqlite:///./app.db` | `postgresql+asyncpg://...` |

## Switching to PostgreSQL

Change the `DATABASE_URL` environment variable:

```bash
DATABASE_URL=postgresql+asyncpg://rag:rag@postgres:5432/ragdb
```

Add to `requirements.txt`:
```
asyncpg>=0.29.0
```

The SQLAlchemy models and queries remain identical — only the driver changes.
