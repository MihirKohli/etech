# LangGraph — Agentic Orchestration

## What is LangGraph?

LangGraph is a library built on top of LangChain for creating stateful, multi-actor applications with LLMs. It models agent workflows as directed graphs where nodes are functions (agents) and edges define control flow between them.

## Core Concepts

### State
A `TypedDict` that flows through every node. Each node reads what it needs and writes its outputs back to state.

```python
from typing import TypedDict, Optional

class AgentState(TypedDict, total=False):
    query: str
    intent: str
    retrieved_chunks: list[dict]
    answer: str
```

### Nodes
Any async or sync callable that takes `state` and returns a partial state update:

```python
async def query_understanding(state: AgentState) -> dict:
    # Classify the query intent using an LLM
    intent = await classify_intent(state["query"])
    return {"intent": intent}
```

### Edges
- **Static edges** — always go from node A to node B
- **Conditional edges** — routing function decides next node at runtime

```python
def route(state: AgentState) -> str:
    if state["intent"] == "conversational":
        return "synthesis"
    return "retrieval"
```

### Building a Graph

```python
from langgraph.graph import StateGraph, END

graph = StateGraph(AgentState)

graph.add_node("understand", query_understanding)
graph.add_node("retrieve", retrieval_node)
graph.add_node("synthesize", synthesis_node)

graph.set_entry_point("understand")
graph.add_conditional_edges("understand", route, {
    "retrieval": "retrieve",
    "synthesis": "synthesize",
})
graph.add_edge("retrieve", "synthesize")
graph.add_edge("synthesize", END)

pipeline = graph.compile()
```

## Streaming Events

LangGraph supports granular event streaming via `astream_events`:

```python
async for event in pipeline.astream_events(state, version="v2"):
    kind = event["event"]
    if kind == "on_chat_model_stream":
        token = event["data"]["chunk"].content
        print(token, end="")
```

Event types:
| Event | Description |
|---|---|
| `on_chain_start` | Node begins execution |
| `on_chain_end` | Node completes |
| `on_chat_model_stream` | LLM token emitted |
| `on_tool_start/end` | Tool called |

## Checkpointing

LangGraph supports persistent checkpointing to resume interrupted workflows:

```python
from langgraph.checkpoint.sqlite import SqliteSaver

saver = SqliteSaver.from_conn_string("checkpoints.db")
pipeline = graph.compile(checkpointer=saver)

# Resume from a thread
config = {"configurable": {"thread_id": "session_123"}}
result = await pipeline.ainvoke(state, config=config)
```

## Multi-Agent Patterns

### Supervisor Pattern
One orchestrator node routes between specialist agents:
```
Supervisor → [Agent A | Agent B | Agent C] → Supervisor → END
```

### Pipeline Pattern
Sequential nodes, each enriching state:
```
Understand → Rewrite → Route → Retrieve → Synthesize → END
```

### Parallel Fan-Out
Multiple retrievers run concurrently, results merged:
```
Router → [Semantic Search ‖ Keyword Search] → Merge → Synthesize
```

## Conditional Routing Best Practices

- Keep routing functions pure (no side effects)
- Return string keys that match the `path_map` exactly
- Use `END` as a terminal key for early exits

```python
graph.add_conditional_edges(
    "router",
    route_fn,
    {
        "semantic": "semantic_node",
        "hybrid": "hybrid_node",
        "memory_only": END,
    }
)
```
