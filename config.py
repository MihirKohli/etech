"""
Application configuration loaded from environment variables.

Uses pydantic-settings so every value can be overridden via .env or env vars.
Defaults are set for local SQLite + Chroma development — swap DATABASE_URL
to a PostgreSQL DSN for production.
"""

from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # ── App ──────────────────────────────────────────────
    APP_NAME: str = "Conversational RAG System"
    DEBUG: bool = True

    # ── LLM ──────────────────────────────────────────────
    LLM_PROVIDER: str = "groq"  # "openai" | "groq"
    OPENAI_API_KEY: str = ""
    GROQ_API_KEY: str = ""
    LLM_MODEL: str = "llama-3.3-70b-versatile"  # or "gpt-4o-mini"
    LLM_TEMPERATURE: float = 0.1

    # ── Embeddings ───────────────────────────────────────
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"  # sentence-transformers model
    EMBEDDING_DIMENSION: int = 384

    # ── Vector Store (Chroma) ────────────────────────────
    CHROMA_PERSIST_DIR: str = "./data/chroma_db"
    CHROMA_COLLECTION: str = "rag_documents"

    # ── Database (sessions & memory) ─────────────────────
    # SQLite for dev, PostgreSQL for prod:
    #   postgresql+asyncpg://user:pass@localhost:5432/ragdb
    DATABASE_URL: str = "sqlite+aiosqlite:///./data/rag.db"

    # ── Document Processing ──────────────────────────────
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 20

    # ── Memory Management ────────────────────────────────
    MAX_CONVERSATION_TURNS: int = 50  # triggers summarization
    SUMMARY_TRIGGER_TURNS: int = 10   # summarize every N turns
    MEMORY_WINDOW_SIZE: int = 5       # recent turns kept in full

    # ── API ──────────────────────────────────────────────
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache
def get_settings() -> Settings:
    """Singleton settings instance (cached)."""
    return Settings()