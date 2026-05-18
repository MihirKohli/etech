"""
Application configuration loaded from environment variables.

Uses pydantic-settings so every value can be overridden via .env or env vars.
Defaults are set for local SQLite + Chroma development — swap DATABASE_URL
to a PostgreSQL DSN for production.
"""

from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    GROQ_API_KEY: str = ""
    LLM_MODEL: str = "llama-3.3-70b-versatile"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    CHROMA_PERSIST_DIR: str = "./data/chroma_db"
    CHROMA_COLLECTION: str = "rag_documents"
    DATABASE_URL: str = "sqlite+aiosqlite:///./data/rag.db"
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 20
    SUMMARY_TRIGGER_TURNS: int = 10
    MEMORY_WINDOW_SIZE: int = 5

    model_config = {"env_file": ".env"}


@lru_cache
def get_settings() -> Settings:
    return Settings()