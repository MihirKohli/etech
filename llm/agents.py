"""
LangGraph RAG pipeline agents.
"""

import logging
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langgraph")
warnings.filterwarnings("ignore", message=".*allowed_objects.*")

from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from llm.llm import get_openai_llm
from llm.schema import AgentState, QueryIntent, RetrievalStrategy, SourceInfo
from db.vector_database import search, keyword_search

logger = logging.getLogger(__name__)


# ── Helpers ──────────────────────────────────────────────────────────────────

def format_history(messages: list[dict], limit: int = 6) -> str:
    if not messages:
        return "No prior conversation."
    return "\n".join(
        f"{m.get('role', 'user')}: {m.get('content', '')}"
        for m in messages[-limit:]
    )


def format_context(chunks: list[dict]) -> str:
    if not chunks:
        return "No documents retrieved."
    parts = []
    for i, c in enumerate(chunks, 1):
        src = c.get("metadata", {}).get("source_file", "unknown")
        parts.append(f"[{i}] (from {src}):\n{c['content']}")
    return "\n\n".join(parts)


# ── Agents ───────────────────────────────────────────────────────────────────

# def query_understanding_agent(state: AgentState) -> dict:
async def query_understanding_agent(state: AgentState) -> dict:
    PROMPT = ChatPromptTemplate.from_messages([
        ("system", """Classify the user query into exactly one intent and decide what's needed.

Respond in this exact format (3 lines, nothing else):
intent: <factual|follow_up|multi_part|conversational>
needs_history: <true|false>
needs_retrieval: <true|false>

Rules:
- factual: direct question answerable from documents
- follow_up: references previous conversation ("what about...", "and how...", "you mentioned...")
- multi_part: contains multiple sub-questions
- conversational: greetings, thanks, chitchat

Conversation summary so far: {summary}"""),
        ("human", "{query}"),
    ])
    llm = get_openai_llm()
    # response = (PROMPT | llm).invoke({...})
    response = await (PROMPT | llm).ainvoke({
        "query": state["original_query"],
        "summary": state.get("conversation_summary", "No prior conversation."),
    })

    lines = response.content.strip().split("\n")
    parsed = {}
    for line in lines:
        if ":" in line:
            key, val = line.split(":", 1)
            parsed[key.strip()] = val.strip()

    intent_map = {
        "factual": QueryIntent.FACTUAL,
        "follow_up": QueryIntent.FOLLOW_UP,
        "multi_part": QueryIntent.MULTI_PART,
        "conversational": QueryIntent.CONVERSATIONAL,
    }
    intent = intent_map.get(parsed.get("intent", ""), QueryIntent.FACTUAL)
    needs_history = parsed.get("needs_history", "false") == "true"
    needs_retrieval = parsed.get("needs_retrieval", "true") == "true"

    if intent == QueryIntent.FOLLOW_UP:
        needs_history = True
    if intent == QueryIntent.CONVERSATIONAL:
        needs_retrieval = False

    return {
        "query_intent": intent,
        "needs_history": needs_history,
        "needs_retrieval": needs_retrieval,
    }


# def query_rewriting_agent(state: AgentState) -> dict:
async def query_rewriting_agent(state: AgentState) -> dict:
    if not state.get("needs_history", False):
        return {"rewritten_query": state["original_query"]}

    PROMPT = ChatPromptTemplate.from_messages([
        ("system", """Rewrite the user's query to be self-contained using the conversation history.

Rules:
- Resolve all pronouns and references ("it", "that", "the above")
- Keep the rewritten query concise
- If the query is already clear, return it unchanged
- Output ONLY the rewritten query, nothing else

Conversation history:
{history}"""),
        ("human", "{query}"),
    ])
    llm = get_openai_llm()
    # response = (PROMPT | llm).invoke({...})
    response = await (PROMPT | llm).ainvoke({
        "query": state["original_query"],
        "history": format_history(state.get("conversation_history", [])),
    })
    rewritten = response.content.strip()
    return {"rewritten_query": rewritten or state["original_query"]}


# def retrieval_router_agent(state: AgentState) -> dict:
async def retrieval_router_agent(state: AgentState) -> dict:
    intent = state.get("query_intent", QueryIntent.FACTUAL)

    if not state.get("needs_retrieval", True):
        return {"retrieval_strategy": RetrievalStrategy.MEMORY_ONLY}
    if intent == QueryIntent.MULTI_PART:
        return {"retrieval_strategy": RetrievalStrategy.HYBRID}
    if intent == QueryIntent.FOLLOW_UP:
        return {"retrieval_strategy": RetrievalStrategy.SEMANTIC}
    return {"retrieval_strategy": RetrievalStrategy.SEMANTIC}


async def semantic_retrieval_node(state: AgentState) -> dict:
    query = state.get("rewritten_query", state["original_query"])
    results = await search(query, session_id=state["session_id"], top_k=5)
    return {"retrieved_chunks": [
        {"chunk_id": r["chunk_id"], "content": r["content"], "metadata": r["metadata"], "score": r["score"]}
        for r in results
    ]}


async def hybrid_retrieval_node(state: AgentState) -> dict:
    query = state.get("rewritten_query", state["original_query"])
    session_id = state["session_id"]

    semantic_results = await search(query, session_id=session_id, top_k=7)
    keyword_results = keyword_search(query, session_id=session_id, top_k=7)

    # Merge by chunk_id, semantic score wins on conflict
    merged: dict[str, dict] = {}
    for r in keyword_results:
        merged[r["chunk_id"]] = {"chunk_id": r["chunk_id"], "content": r["content"], "metadata": r["metadata"], "score": r["score"]}
    for r in semantic_results:
        merged[r["chunk_id"]] = {"chunk_id": r["chunk_id"], "content": r["content"], "metadata": r["metadata"], "score": r["score"]}

    chunks = sorted(merged.values(), key=lambda x: x["score"], reverse=True)[:7]
    return {"retrieved_chunks": chunks}


async def memory_only_node(_state: AgentState) -> dict:
    return {"retrieved_chunks": []}


async def query_decomposition_node(state: AgentState) -> dict:
    """
    For MULTI_PART queries: split the question into sub-questions, run hybrid
    retrieval for each, then merge results (deduped by chunk_id).
    Falls back to a single hybrid retrieval if decomposition yields nothing.
    """
    PROMPT = ChatPromptTemplate.from_messages([
        ("system", """Break the user question into independent sub-questions.
Output ONLY a numbered list, one sub-question per line.
If the question is not multi-part, output it unchanged as a single item.
Maximum 4 sub-questions.

Example:
1. What is LangChain?
2. How does it compare to LlamaIndex?"""),
        ("human", "{query}"),
    ])
    llm = get_openai_llm()
    response = await (PROMPT | llm).ainvoke({
        "query": state.get("rewritten_query", state["original_query"]),
    })

    lines = [l.strip() for l in response.content.strip().splitlines() if l.strip()]
    sub_questions = []
    for line in lines:
        # strip leading "1. " / "- " markers
        clean = line.lstrip("0123456789.-) ").strip()
        if clean:
            sub_questions.append(clean)

    if not sub_questions:
        sub_questions = [state.get("rewritten_query", state["original_query"])]

    session_id = state["session_id"]
    merged: dict[str, dict] = {}
    for sq in sub_questions:
        sem = await search(sq, session_id=session_id, top_k=5)
        kw = keyword_search(sq, session_id=session_id, top_k=5)
        for r in kw:
            merged.setdefault(r["chunk_id"], r)
        for r in sem:
            merged[r["chunk_id"]] = r  # semantic score wins

    chunks = sorted(merged.values(), key=lambda x: x["score"], reverse=True)[:8]
    return {
        "retrieved_chunks": chunks,
        "sub_questions": sub_questions,
    }


# def context_synthesis_agent(state: AgentState) -> dict:
async def context_synthesis_agent(state: AgentState) -> dict:
    PROMPT = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful technical assistant. Answer the user's question
using ONLY the provided context. If the context doesn't contain the answer, say so.

Rules:
- Be concise and accurate
- Reference specific documents when possible
- If code is relevant, include code examples
- Don't make up information not in the context

Retrieved documents:
{context}

Conversation history:
{history}"""),
        ("human", "{query}"),
    ])
    llm = get_openai_llm()
    chunks = state.get("retrieved_chunks", [])
    # response = (PROMPT | llm).invoke({...})
    response = await (PROMPT | llm).ainvoke({
        "query": state.get("rewritten_query", state["original_query"]),
        "context": format_context(chunks),
        "history": format_history(state.get("conversation_history", []), limit=4),
    })

    sources = []
    for c in chunks[:3]:
        meta = c.get("metadata", {})
        sources.append(SourceInfo(
            document_name=meta.get("source_file", "unknown"),
            snippet=c["content"][:150],
            score=c.get("score", 0.0),
        ))

    return {
        "answer": response.content.strip(),
        "sources": sources,
    }


# def summarize_conversation(messages: list[dict], existing_summary: str = "") -> str:
async def summarize_conversation(messages: list[dict], existing_summary: str = "") -> str:
    """Takes message dicts and returns a summary string."""
    if not messages:
        return existing_summary

    PROMPT = ChatPromptTemplate.from_messages([
        ("system", """Summarize this conversation in 3-5 sentences.
Capture: key topics discussed, questions asked, important facts mentioned,
and any user preferences revealed. Be concise.

Existing summary (if any): {existing_summary}"""),
        ("human", "Conversation to summarize:\n{conversation}"),
    ])
    llm = get_openai_llm()
    conversation_text = "\n".join(
        f"{m.get('role', 'user')}: {m.get('content', '')}"
        for m in messages
    )
    # response = (PROMPT | llm).invoke({...})
    response = await (PROMPT | llm).ainvoke({
        "existing_summary": existing_summary or "None",
        "conversation": conversation_text,
    })
    return response.content.strip()


# def extract_memories(user_msg: str, assistant_msg: str) -> list[dict]:
async def extract_memories(user_msg: str, assistant_msg: str) -> list[dict]:
    """Returns list of {memory_type, content} dicts to store."""
    PROMPT = ChatPromptTemplate.from_messages([
        ("system", """Extract any important user preferences or facts from this exchange
that should be remembered for future conversations.

Respond in this format (one per line, or "none" if nothing worth remembering):
type: <preference|fact>
content: <what to remember>

Examples:
type: preference
content: User prefers Python over JavaScript

type: fact
content: User is building a FastAPI microservice

Only extract genuinely useful long-term information. Skip transient questions."""),
        ("human", "User said: {user_msg}\nAssistant replied: {assistant_msg}"),
    ])
    llm = get_openai_llm()
    # response = (PROMPT | llm).invoke({...})
    response = await (PROMPT | llm).ainvoke({
        "user_msg": user_msg,
        "assistant_msg": assistant_msg,
    })

    text = response.content.strip()
    if text.lower() == "none":
        return []

    memories = []
    lines = text.split("\n")
    current: dict = {}
    for line in lines:
        if ":" in line:
            key, val = line.split(":", 1)
            key = key.strip().lower()
            val = val.strip()
            if key == "type":
                current["memory_type"] = val
            elif key == "content":
                current["content"] = val
                if "memory_type" in current and "content" in current:
                    memories.append(current.copy())
                    current = {}

    return memories


# ── LangGraph Pipeline ───────────────────────────────────────────────────────

def route_retrieval(state: AgentState) -> str:
    strategy = state.get("retrieval_strategy", RetrievalStrategy.SEMANTIC)
    if strategy == RetrievalStrategy.MEMORY_ONLY:
        return "memory_only"
    if strategy == RetrievalStrategy.HYBRID:
        return "query_decomposition"
    return "semantic_retrieval"


def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("query_understanding", query_understanding_agent)
    graph.add_node("query_rewriting", query_rewriting_agent)
    graph.add_node("retrieval_router", retrieval_router_agent)
    graph.add_node("semantic_retrieval", semantic_retrieval_node)
    graph.add_node("query_decomposition", query_decomposition_node)
    graph.add_node("memory_only", memory_only_node)
    graph.add_node("context_synthesis", context_synthesis_agent)

    graph.set_entry_point("query_understanding")
    graph.add_edge("query_understanding", "query_rewriting")
    graph.add_edge("query_rewriting", "retrieval_router")
    graph.add_conditional_edges("retrieval_router", route_retrieval, {
        "semantic_retrieval": "semantic_retrieval",
        "query_decomposition": "query_decomposition",
        "memory_only": "memory_only",
    })
    graph.add_edge("semantic_retrieval", "context_synthesis")
    graph.add_edge("query_decomposition", "context_synthesis")
    graph.add_edge("memory_only", "context_synthesis")
    graph.add_edge("context_synthesis", END)

    return graph.compile()


pipeline = build_graph()


# def run_pipeline(...) -> dict:
#     return pipeline.invoke(initial_state)
async def run_pipeline(
    session_id: str,
    user_id: str,
    query: str,
    conversation_history: list[dict],
    conversation_summary: str = "",
) -> dict:
    initial_state: AgentState = {
        "session_id": session_id,
        "user_id": user_id,
        "original_query": query,
        "conversation_history": conversation_history,
        "conversation_summary": conversation_summary,
    }

    final_state = {}
    async for event in pipeline.astream_events(initial_state, version="v2"):
        kind = event["event"]
        name = event.get("name", "")

        if kind == "on_chain_start" and name in {
            "query_understanding", "query_rewriting", "retrieval_router",
            "semantic_retrieval", "hybrid_retrieval", "memory_only", "context_synthesis",
        }:
            logger.info(f"[node:start]  {name}")

        elif kind == "on_chain_end" and name in {
            "query_understanding", "query_rewriting", "retrieval_router",
            "semantic_retrieval", "hybrid_retrieval", "memory_only", "context_synthesis",
        }:
            logger.info(f"[node:end]    {name}")
            if name == "retrieval_router":
                strategy = event.get("data", {}).get("output", {}).get("retrieval_strategy", "")
                logger.info(f"[edge]        retrieval_router → {strategy}")

        elif kind == "on_chain_end" and name == "LangGraph":
            final_state = event.get("data", {}).get("output", {})

    return final_state


async def stream_pipeline(
    session_id: str,
    user_id: str,
    query: str,
    conversation_history: list[dict],
    conversation_summary: str = "",
):
    """
    Yields token strings from context_synthesis, then a final dict with
    {"done": True, "answer": ..., "sources": ...} for the caller to save.
    """
    initial_state: AgentState = {
        "session_id": session_id,
        "user_id": user_id,
        "original_query": query,
        "conversation_history": conversation_history,
        "conversation_summary": conversation_summary,
    }

    full_answer = []
    final_state = {}

    async for event in pipeline.astream_events(initial_state, version="v2"):
        kind = event["event"]
        name = event.get("name", "")
        meta = event.get("metadata", {})

        # Log nodes and edges (same as run_pipeline)
        if kind == "on_chain_start" and name in {
            "query_understanding", "query_rewriting", "retrieval_router",
            "semantic_retrieval", "hybrid_retrieval", "memory_only", "context_synthesis",
        }:
            logger.info(f"[node:start]  {name}")

        elif kind == "on_chain_end" and name in {
            "query_understanding", "query_rewriting", "retrieval_router",
            "semantic_retrieval", "hybrid_retrieval", "memory_only", "context_synthesis",
        }:
            logger.info(f"[node:end]    {name}")
            if name == "retrieval_router":
                strategy = event.get("data", {}).get("output", {}).get("retrieval_strategy", "")
                logger.info(f"[edge]        retrieval_router → {strategy}")

        # Stream tokens only from the context_synthesis node
        elif (
            kind == "on_chat_model_stream"
            and meta.get("langgraph_node") == "context_synthesis"
        ):
            token = event["data"]["chunk"].content
            if token:
                full_answer.append(token)
                yield token

        elif kind == "on_chain_end" and name == "LangGraph":
            final_state = event.get("data", {}).get("output", {})

    # Final metadata event — caller uses this to save to DB
    yield {
        "done": True,
        "answer": "".join(full_answer),
        "sources": final_state.get("sources", []),
    }
