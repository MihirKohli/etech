"""
Query Understanding Agent — classifies user intent and decides
whether conversation history or document retrieval is needed.
"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langgraph")
warnings.filterwarnings("ignore", message=".*allowed_objects.*")

from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from llm.llm import get_openai_llm
from llm.schema import AgentState, QueryIntent, RetrievalStrategy, SourceInfo
from db.vector_database import search


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


def query_understanding_agent(state: AgentState) -> dict:
    llm = get_openai_llm()
    chain = PROMPT | llm

    response = chain.invoke({
        "query": state["original_query"],
        "summary": state.get("conversation_summary", "No prior conversation."),
    })

    # Parse the structured response
    lines = response.content.strip().split("\n")
    parsed = {}
    for line in lines:
        if ":" in line:
            key, val = line.split(":", 1)
            parsed[key.strip()] = val.strip()

    # Map to enum with fallback
    intent_map = {
        "factual": QueryIntent.FACTUAL,
        "follow_up": QueryIntent.FOLLOW_UP,
        "multi_part": QueryIntent.MULTI_PART,
        "conversational": QueryIntent.CONVERSATIONAL,
    }

    intent = intent_map.get(parsed.get("intent", ""), QueryIntent.FACTUAL)
    needs_history = parsed.get("needs_history", "false") == "true"
    needs_retrieval = parsed.get("needs_retrieval", "true") == "true"

    # Follow-ups always need history
    if intent == QueryIntent.FOLLOW_UP:
        needs_history = True

    # Conversational queries don't need retrieval
    if intent == QueryIntent.CONVERSATIONAL:
        needs_retrieval = False

    return {
        "query_intent": intent,
        "needs_history": needs_history,
        "needs_retrieval": needs_retrieval,
    }




"""
Query Rewriting Agent — rewrites the query using conversation history
so it's self-contained and clear for retrieval.

Only runs when the query understanding agent says needs_history=True.
"""

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


def format_history(messages: list[dict]) -> str:
    if not messages:
        return "No prior conversation."
    lines = []
    for m in messages[-6:]:  # last 6 messages max
        role = m.get("role", "user")
        content = m.get("content", "")
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def query_rewriting_agent(state: AgentState) -> dict:
    # Skip if history not needed
    if not state.get("needs_history", False):
        return {"rewritten_query": state["original_query"]}

    llm = get_openai_llm()
    chain = PROMPT | llm

    history_text = format_history(state.get("conversation_history", []))

    response = chain.invoke({
        "query": state["original_query"],
        "history": history_text,
    })

    rewritten = response.content.strip()
    return {"rewritten_query": rewritten or state["original_query"]}




"""
Retrieval Router Agent — picks the best retrieval strategy.

Simple rule-based routing (no LLM call needed here).
"""

def retrieval_router_agent(state: AgentState) -> dict:
    intent = state.get("query_intent", QueryIntent.FACTUAL)

    if not state.get("needs_retrieval", True):
        return {"retrieval_strategy": RetrievalStrategy.MEMORY_ONLY}

    if intent == QueryIntent.MULTI_PART:
        return {"retrieval_strategy": RetrievalStrategy.HYBRID}

    if intent == QueryIntent.FOLLOW_UP:
        return {"retrieval_strategy": RetrievalStrategy.SEMANTIC}

    # Default: semantic search
    return {"retrieval_strategy": RetrievalStrategy.SEMANTIC}



"""
Retrieval node — runs vector search using the rewritten query.

Not an LLM agent, just a function that calls the vector store.
"""

def retrieval_node(state: AgentState) -> dict:
    strategy = state.get("retrieval_strategy", RetrievalStrategy.SEMANTIC)

    # If memory-only, skip retrieval
    if strategy == RetrievalStrategy.MEMORY_ONLY:
        return {"retrieved_chunks": []}

    query = state.get("rewritten_query", state["original_query"])

    # For hybrid/multi-part, get more results
    top_k = 7 if strategy == RetrievalStrategy.HYBRID else 5

    results = search(query, top_k=top_k)

    # Convert to SearchResult-compatible dicts
    chunks = [
        {
            "chunk_id": r["chunk_id"],
            "content": r["content"],
            "metadata": r["metadata"],
            "score": r["score"],
        }
        for r in results
    ]

    return {"retrieved_chunks": chunks}



"""
Context Synthesis Agent — combines retrieved documents with
conversation history and generates the final answer.
"""


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


def format_context(chunks: list[dict]) -> str:
    if not chunks:
        return "No documents retrieved."
    parts = []
    for i, c in enumerate(chunks, 1):
        src = c.get("metadata", {}).get("source_file", "unknown")
        parts.append(f"[{i}] (from {src}):\n{c['content']}")
    return "\n\n".join(parts)


def format_history(messages: list[dict]) -> str:
    if not messages:
        return "No prior conversation."
    lines = []
    for m in messages[-4:]:
        lines.append(f"{m.get('role', 'user')}: {m.get('content', '')}")
    return "\n".join(lines)


def context_synthesis_agent(state: AgentState) -> dict:
    llm = get_openai_llm()
    chain = PROMPT | llm

    chunks = state.get("retrieved_chunks", [])
    history = state.get("conversation_history", [])

    response = chain.invoke({
        "query": state.get("rewritten_query", state["original_query"]),
        "context": format_context(chunks),
        "history": format_history(history),
    })

    # Build source list from top chunks
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



"""
Conversation Summarization Agent — compresses long conversation
history into a summary. Triggered when turn count hits the threshold.
"""


PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Summarize this conversation in 3-5 sentences.
Capture: key topics discussed, questions asked, important facts mentioned,
and any user preferences revealed. Be concise.

Existing summary (if any): {existing_summary}"""),
    ("human", "Conversation to summarize:\n{conversation}"),
])


def summarize_conversation(messages: list[dict], existing_summary: str = "") -> str:
    """
    Takes message dicts [{role, content}, ...] and returns a summary string.
    Called by the orchestrator when turn count crosses the threshold.
    """
    if not messages:
        return existing_summary

    llm = get_openai_llm()
    chain = PROMPT | llm

    conversation_text = "\n".join(
        f"{m.get('role', 'user')}: {m.get('content', '')}"
        for m in messages
    )

    response = chain.invoke({
        "existing_summary": existing_summary or "None",
        "conversation": conversation_text,
    })

    return response.content.strip()



"""
Memory Management Agent — extracts important facts/preferences
from the conversation to store in long-term cross-session memory.
"""


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


def extract_memories(user_msg: str, assistant_msg: str) -> list[dict]:
    """
    Returns list of {{memory_type, content}} dicts to store.
    Returns empty list if nothing worth remembering.
    """
    llm = get_openai_llm()
    chain = PROMPT | llm

    response = chain.invoke({
        "user_msg": user_msg,
        "assistant_msg": assistant_msg,
    })

    text = response.content.strip()
    if text.lower() == "none":
        return []

    memories = []
    lines = text.split("\n")
    current = {}

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

def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("query_understanding", query_understanding_agent)
    graph.add_node("query_rewriting", query_rewriting_agent)
    graph.add_node("retrieval_router", retrieval_router_agent)
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("context_synthesis", context_synthesis_agent)

    graph.set_entry_point("query_understanding")
    graph.add_edge("query_understanding", "query_rewriting")
    graph.add_edge("query_rewriting", "retrieval_router")
    graph.add_edge("retrieval_router", "retrieval")
    graph.add_edge("retrieval", "context_synthesis")
    graph.add_edge("context_synthesis", END)

    return graph.compile()


pipeline = build_graph()


def run_pipeline(
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
    return pipeline.invoke(initial_state)