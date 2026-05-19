"""
Application configuration loaded from environment variables.

Uses pydantic-settings so every value can be overridden via .env or env vars.
Defaults are set for local SQLite + Chroma development — swap DATABASE_URL
to a PostgreSQL DSN for production.
"""

from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    OPENAI_API_KEY: str
    LLM_MODEL: str = "gpt-4o-mini"
    LLM_TEMPERATURE: float = 0.3

    EMBEDDING_MODEL: str = "text-embedding-ada-002"
    CHROMA_PERSIST_DIR: str = "./data/chroma_db"
    CHROMA_COLLECTION: str = "rag_documents"
    CHROMA_HOST: str = ""        # set to "chroma" in Docker; empty = use local PersistentClient
    CHROMA_PORT: int = 8001
    DATABASE_URL: str = "sqlite+aiosqlite:///./data/rag.db"
    API_URL: str = "http://localhost:8080"

    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 20
    SUMMARY_TRIGGER_TURNS: int = 10
    MEMORY_WINDOW_SIZE: int = 5
    MEMORY_MAX_ENTRIES: int = 50   # max long-term memories kept per user; oldest pruned beyond this

    model_config = {"env_file": ".env"}


@lru_cache
def get_settings() -> Settings:
    return Settings()