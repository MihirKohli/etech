"""
Module 3 tests — agent routing logic and graph structure.

Unit tests mock the LLM. Integration test needs OPENAI_API_KEY + ingested docs.

Run unit tests:    pytest test/module_test_3.py -v -k "not integration"
Run all:           pytest test/module_test_3.py -v
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from llm.schema import AgentState, QueryIntent, RetrievalStrategy
from llm.agents import retrieval_router_agent, route_retrieval


# ── Retrieval Router (pure logic, no LLM) ────────────

@pytest.mark.asyncio
async def test_router_factual():
    state: AgentState = {"query_intent": QueryIntent.FACTUAL, "needs_retrieval": True}
    result = await retrieval_router_agent(state)
    assert result["retrieval_strategy"] == RetrievalStrategy.SEMANTIC


@pytest.mark.asyncio
async def test_router_conversational():
    state: AgentState = {"query_intent": QueryIntent.CONVERSATIONAL, "needs_retrieval": False}
    result = await retrieval_router_agent(state)
    assert result["retrieval_strategy"] == RetrievalStrategy.MEMORY_ONLY


@pytest.mark.asyncio
async def test_router_multi_part():
    state: AgentState = {"query_intent": QueryIntent.MULTI_PART, "needs_retrieval": True}
    result = await retrieval_router_agent(state)
    assert result["retrieval_strategy"] == RetrievalStrategy.HYBRID


@pytest.mark.asyncio
async def test_router_follow_up():
    state: AgentState = {"query_intent": QueryIntent.FOLLOW_UP, "needs_retrieval": True}
    result = await retrieval_router_agent(state)
    assert result["retrieval_strategy"] == RetrievalStrategy.SEMANTIC


# ── Graph edge routing ────────────────────────────────

def test_route_retrieval_memory_only():
    state: AgentState = {"retrieval_strategy": RetrievalStrategy.MEMORY_ONLY}
    assert route_retrieval(state) == "memory_only"


def test_route_retrieval_hybrid():
    state: AgentState = {"retrieval_strategy": RetrievalStrategy.HYBRID}
    assert route_retrieval(state) == "query_decomposition"


def test_route_retrieval_semantic():
    state: AgentState = {"retrieval_strategy": RetrievalStrategy.SEMANTIC}
    assert route_retrieval(state) == "semantic_retrieval"


# ── Query Understanding (mocked LLM) ─────────────────

@pytest.mark.asyncio
async def test_query_understanding_factual():
    mock_response = MagicMock()
    mock_response.content = "intent: factual\nneeds_history: false\nneeds_retrieval: true"

    mock_chain = MagicMock()
    mock_chain.ainvoke = AsyncMock(return_value=mock_response)

    with patch("llm.agents.get_openai_llm") as mock_llm_fn:
        mock_llm = MagicMock()
        mock_llm_fn.return_value = mock_llm

        with patch("llm.agents.ChatPromptTemplate.from_messages") as mock_prompt:
            mock_prompt.return_value.__or__ = MagicMock(return_value=mock_chain)
            # Patch at the chain level directly
            pass

    # Simpler: patch ainvoke at the chain level via the prompt | llm pipe
    from llm.agents import query_understanding_agent
    state: AgentState = {
        "original_query": "What is FastAPI?",
        "conversation_summary": "",
        "session_id": "s1",
        "user_id": "u1",
        "conversation_history": [],
    }

    fake_response = MagicMock()
    fake_response.content = "intent: factual\nneeds_history: false\nneeds_retrieval: true"

    with patch("llm.agents.get_openai_llm") as mock_llm_fn:
        mock_llm = MagicMock()
        mock_llm_fn.return_value = mock_llm
        pipe = MagicMock()
        pipe.ainvoke = AsyncMock(return_value=fake_response)
        mock_llm.__ror__ = MagicMock(return_value=pipe)

        result = await query_understanding_agent(state)

    assert result["query_intent"] == QueryIntent.FACTUAL
    assert result["needs_retrieval"] is True


# ── Query Rewriting — skip if no history ─────────────

@pytest.mark.asyncio
async def test_query_rewriting_skips_without_history():
    from llm.agents import query_rewriting_agent
    state: AgentState = {
        "original_query": "What is RAG?",
        "needs_history": False,
        "conversation_history": [],
        "session_id": "s1",
        "user_id": "u1",
    }
    result = await query_rewriting_agent(state)
    assert result["rewritten_query"] == "What is RAG?"


# ── Integration test (needs API key + ingested docs) ─

@pytest.mark.asyncio
@pytest.mark.skipif(
    not __import__("os").environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)
async def test_integration_full_pipeline():
    from llm.agents import run_pipeline

    result = await run_pipeline(
        session_id="test_session",
        user_id="test_user",
        query="What is RAG?",
        conversation_history=[],
    )

    assert "answer" in result
    assert len(result["answer"]) > 0
