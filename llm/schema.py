"""
Pydantic schemas used across the system.

Three categories:
  1. API schemas    — request/response models for FastAPI endpoints
  2. Internal DTOs  — data passed between services (ingestion, search, etc.)
  3. Agent state    — the TypedDict that flows through the LangGraph pipeline
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, TypedDict
from uuid import uuid4

from pydantic import BaseModel, Field


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Enums
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class DocumentType(str, Enum):
    PDF = "pdf"
    MARKDOWN = "markdown"
    HTML = "html"


class QueryIntent(str, Enum):
    """Classified by the Query Understanding Agent."""
    FACTUAL = "factual"           # direct lookup from docs
    FOLLOW_UP = "follow_up"       # needs conversation context
    CLARIFICATION = "clarification"
    MULTI_PART = "multi_part"     # decomposable question
    CONVERSATIONAL = "conversational"  # greetings, chitchat


class RetrievalStrategy(str, Enum):
    """Chosen by the Retrieval Router Agent."""
    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    HYBRID = "hybrid"
    MEMORY_ONLY = "memory_only"   # answer from conversation history alone


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  API Schemas
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class SessionCreate(BaseModel):
    """POST /sessions — start a new conversation."""
    user_id: str = Field(..., description="Unique user identifier")
    metadata: dict[str, Any] = Field(default_factory=dict)


class SessionResponse(BaseModel):
    session_id: str
    user_id: str
    created_at: datetime


class ChatRequest(BaseModel):
    """POST /chat — send a message within a session."""
    session_id: str
    message: str


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    sources: list[SourceInfo] = []
    agent_trace: list[str] = Field(
        default_factory=list,
        description="Ordered list of agents that processed this query",
    )


class SourceInfo(BaseModel):
    """A single retrieved chunk shown as a citation."""
    document_name: str
    page: int | None = None
    section: str | None = None
    score: float
    snippet: str


class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    chunks_created: int
    metadata: dict[str, Any] = {}


# Fix forward reference (ChatResponse uses SourceInfo)
ChatResponse.model_rebuild()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Internal DTOs
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class DocumentChunk(BaseModel):
    """A single chunk ready for embedding and vector storage."""
    chunk_id: str = Field(default_factory=lambda: uuid4().hex)
    document_id: str
    content: str
    metadata: ChunkMetadata


class ChunkMetadata(BaseModel):
    source_file: str
    document_type: DocumentType
    page_number: int | None = None
    section_header: str | None = None
    has_code_block: bool = False
    chunk_index: int = 0


# Rebuild DocumentChunk now that ChunkMetadata is defined
DocumentChunk.model_rebuild()


class SearchResult(BaseModel):
    """Single result from the vector store."""
    chunk_id: str
    content: str
    metadata: dict[str, Any]
    score: float


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  LangGraph Agent State
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class AgentState(TypedDict, total=False):
    """
    The shared state dict that flows through every node in the
    LangGraph pipeline.  Each agent reads what it needs and writes
    its output keys.

    Keys are added incrementally — `total=False` lets each node
    only set the keys it owns.
    """

    # ── Input ────────────────────────────────────────────
    session_id: str
    user_id: str
    original_query: str

    # ── Query Understanding Agent output ─────────────────
    query_intent: QueryIntent
    needs_history: bool
    needs_retrieval: bool

    # ── Query Rewriting Agent output ─────────────────────
    rewritten_query: str

    # ── Retrieval Router Agent output ────────────────────
    retrieval_strategy: RetrievalStrategy

    # ── Retrieved context ────────────────────────────────
    retrieved_chunks: list[SearchResult]

    # ── Conversation history (loaded by memory agent) ────
    conversation_history: list[dict[str, str]]
    conversation_summary: str

    # ── Context Synthesis Agent output ───────────────────
    synthesized_context: str

    # ── Final answer ─────────────────────────────────────
    answer: str
    sources: list[SourceInfo]

    # ── Query decomposition output ───────────────────────
    sub_questions: list[str]

    # ── Trace / observability ────────────────────────────
    agent_trace: list[str]