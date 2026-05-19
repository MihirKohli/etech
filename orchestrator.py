from langgraph.graph import StateGraph, END
from schema import AgentState, QueryIntent,RetrievalStrategy
from agents import query_understanding_agent, query_rewriting_agent, retrieval_router_agent, retrieval_node, context_synthesis_agent


def route_after_understanding(state: AgentState) -> str:
    """Skip to synthesis for conversational queries, else continue pipeline."""
    if state.get("query_intent") == QueryIntent.CONVERSATIONAL:
        return "context_synthesis"
    return "query_rewriting"
 
def route_after_router(state: AgentState) -> str:
    """Skip retrieval if memory-only."""
    if state.get("retrieval_strategy") == RetrievalStrategy.MEMORY_ONLY:
        return "context_synthesis"
    return "retrieval"
 

def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("query_understanding", query_understanding_agent)
    graph.add_node("query_rewriting", query_rewriting_agent)
    graph.add_node("retrieval_router", retrieval_router_agent)
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("context_synthesis", context_synthesis_agent)

    graph.set_entry_point("query_understanding")

    graph.add_conditional_edges(
        "query_understanding",
        route_after_understanding,
        {
            "query_rewriting": "query_rewriting",
            "context_synthesis": "context_synthesis",
        },
    )

    graph.add_edge("query_rewriting", "retrieval_router")

     # Conditional: router → retrieval OR synthesis
    graph.add_conditional_edges(
        "retrieval_router",
        route_after_router,
        {
            "retrieval": "retrieval",
            "context_synthesis": "context_synthesis",
        },
    )
 
    # Retrieval → synthesis
    graph.add_edge("retrieval", "context_synthesis")
 
    # Synthesis → END
    graph.add_edge("context_synthesis", END)
 
    return graph
 

rag_graph = build_graph().compile()


def run_pipeline(
    session_id: str,
    user_id: str,
    query: str,
    conversation_history: list[dict] | None = None,
    conversation_summary: str = "",
) -> dict:
    """
    Run a query through the full agent pipeline.
 
    Returns the final AgentState with answer + sources.
    """
    initial_state: AgentState = {
        "session_id": session_id,
        "user_id": user_id,
        "original_query": query,
        "conversation_history": conversation_history or [],
        "conversation_summary": conversation_summary,
    }
 
    result = rag_graph.invoke(initial_state)
    return result