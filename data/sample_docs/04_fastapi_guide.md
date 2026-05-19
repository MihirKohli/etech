# FastAPI — Production API Development Guide

## Overview

FastAPI is a modern, high-performance Python web framework built on Starlette and Pydantic. It provides automatic OpenAPI documentation, async support, and type-safe request/response handling.

## Key Features

- **Async-first** — built on ASGI, supports `async def` handlers natively
- **Auto-generated docs** — Swagger UI at `/docs`, ReDoc at `/redoc`
- **Type validation** — Pydantic models validate request bodies automatically
- **Dependency injection** — clean way to share DB sessions, settings, auth

## Application Setup

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await init_database()
    yield
    # Shutdown
    await cleanup()

app = FastAPI(title="My API", version="1.0.0", lifespan=lifespan)
```

## Routers and Modular Structure

```python
# routes/chat.py
from fastapi import APIRouter

router = APIRouter(prefix="/chat", tags=["chat"])

@router.post("/")
async def send_message(req: ChatRequest) -> ChatResponse:
    ...

# main.py
app.include_router(chat_router)
app.include_router(session_router)
```

## Request and Response Models

```python
from pydantic import BaseModel, Field
from typing import Optional

class ChatRequest(BaseModel):
    session_id: str
    message: str = Field(..., min_length=1, max_length=2000)

class ChatResponse(BaseModel):
    session_id: str
    answer: str
    sources: list[SourceInfo] = []
```

## Dependency Injection

```python
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

async def get_db() -> AsyncSession:
    async with async_session_factory() as session:
        yield session

@router.get("/sessions/{id}")
async def get_session(id: str, db: AsyncSession = Depends(get_db)):
    return await db.get(Session, id)
```

## File Upload

```python
from fastapi import UploadFile, File
import shutil

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    dest = f"./uploads/{file.filename}"
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"filename": file.filename}
```

## Streaming Responses (Server-Sent Events)

```python
from fastapi.responses import StreamingResponse

@router.post("/stream")
async def stream_response(req: ChatRequest):
    async def generator():
        async for token in llm.astream(req.message):
            yield f"data: {token}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generator(), media_type="text/event-stream")
```

## Error Handling

```python
from fastapi import HTTPException

@router.get("/sessions/{id}")
async def get_session(id: str, db=Depends(get_db)):
    session = await db.get(Session, id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session
```

## Settings Management

```python
from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    DATABASE_URL: str = "sqlite+aiosqlite:///./app.db"
    OPENAI_API_KEY: str
    model_config = {"env_file": ".env"}

@lru_cache
def get_settings() -> Settings:
    return Settings()
```

## Running with Uvicorn

```bash
# Development
uvicorn main:app --reload --port 8000

# Production
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```
