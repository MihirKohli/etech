"""
Module 3 tests — agent routing logic and graph structure.

Unit tests mock the LLM. Integration test needs GROQ_API_KEY.

Run unit tests:    pytest tests/test_module3.py -v -k "not integration"
Run all:           pytest tests/test_module3.py -v
"""

import pytest
from unittest.mock import patch, MagicMock

from schema import AgentState, QueryIntent, RetrievalStrategy
from agents import retrieval_router_agent
from agents import route_after_understanding, route_after_router


# ── Retrieval Router (no LLM, pure logic) ────────────

def test_router_factual():
    state: AgentState = {"query_intent": QueryIntent.FACTUAL, "needs_retrieval": True}
    result = retrieval_router_agent(state)
    assert result["retrieval_strategy"] == RetrievalStrategy.SEMANTIC


def test_router_conversational():
    state: AgentState = {"query_intent": QueryIntent.CONVERSATIONAL, "needs_retrieval": False}
    result = retrieval_router_agent(state)
    assert result["retrieval_strategy"] == RetrievalStrategy.MEMORY_ONLY


def test_router_multi_part():
    state: AgentState = {"query_intent": QueryIntent.MULTI_PART, "needs_retrieval": True}
    result = retrieval_router_agent(state)
    assert result["retrieval_strategy"] == RetrievalStrategy.HYBRID


# ── Graph routing functions ──────────────────────────

def test_route_conversational_skips_retrieval():
    state: AgentState = {"query_intent": QueryIntent.CONVERSATIONAL}
    assert route_after_understanding(state) == "context_synthesis"


def test_route_factual_continues():
    state: AgentState = {"query_intent": QueryIntent.FACTUAL}
    assert route_after_understanding(state) == "query_rewriting"


def test_route_memory_only_skips_retrieval():
    state: AgentState = {"retrieval_strategy": RetrievalStrategy.MEMORY_ONLY}
    assert route_after_router(state) == "context_synthesis"


def test_route_semantic_does_retrieval():
    state: AgentState = {"retrieval_strategy": RetrievalStrategy.SEMANTIC}
    assert route_after_router(state) == "retrieval"


# ── Query Understanding (mocked LLM) ────────────────

def test_query_understanding_mocked():
    mock_response = MagicMock()
    mock_response.content = "intent: factual\nneeds_history: false\nneeds_retrieval: true"

    mock_llm = MagicMock()
    mock_llm.__or__ = lambda self, other: other  # for | operator
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = mock_response

    with patch("app.agents.query_understanding.get_llm", return_value=mock_llm):
        with patch("app.agents.query_understanding.PROMPT.__or__", return_value=mock_chain):
            from agents import query_understanding_agent
            state: AgentState = {
                "original_query": "What is FastAPI?",
                "conversation_summary": "",
            }
            result = query_understanding_agent(state)
            assert result["query_intent"] == QueryIntent.FACTUAL
            assert result["needs_retrieval"] is True


# ── Integration test (needs API key + ingested docs) ─

@pytest.mark.skipif(
    not __import__("app.core.config", fromlist=["get_settings"]).get_settings().GROQ_API_KEY,
    reason="GROQ_API_KEY not set",
)
def test_integration_full_pipeline():
    """Run the full pipeline with real LLM. Needs ingested docs."""
    from orchestrator import run_pipeline

    result = run_pipeline(
        session_id="test_session",
        user_id="test_user",
        query="What is RAG?",
    )

    assert "answer" in result
    assert len(result["answer"]) > 0
    print(f"\nAnswer: {result['answer'][:200]}")