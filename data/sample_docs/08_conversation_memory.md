# Conversation Memory Management in LLM Applications

## The Problem

LLMs are stateless — each API call is independent. To build coherent multi-turn conversations, the application must explicitly manage and inject conversation history into every request.

## Memory Types

### 1. In-Context Memory (Short-term)
Raw message history passed directly in the prompt. Simplest approach, but limited by the context window.

```python
messages = [
    {"role": "user", "content": "What is RAG?"},
    {"role": "assistant", "content": "RAG stands for Retrieval-Augmented Generation..."},
    {"role": "user", "content": "How does it compare to fine-tuning?"},  # current query
]
response = llm.invoke(messages)
```

**Pros:** Perfect recall, no information loss
**Cons:** Grows unboundedly; expensive at scale

### 2. Summary Memory
Periodically compress history into a running summary using an LLM call:

```python
summary_prompt = """Summarize this conversation in 3-5 sentences.
Capture: topics discussed, questions asked, key facts, user preferences.

Previous summary: {existing_summary}
New messages: {new_messages}"""
```

**Pros:** Bounded token usage; retains key information
**Cons:** Lossy compression; subtle details may be dropped

### 3. Sliding Window Memory
Keep only the last N turns in context:

```python
WINDOW_SIZE = 5
recent_messages = all_messages[-WINDOW_SIZE * 2:]  # N user + N assistant turns
```

**Pros:** Simple, predictable cost
**Cons:** Loses older context entirely

### 4. Long-term Memory (Cross-session)
Extract durable facts and preferences from conversations and store them in a database, retrieving them in future sessions:

```python
# After each exchange, extract memorable facts
memories = extract_memories(user_message, assistant_response)
# → [{"type": "preference", "content": "User prefers Python over JavaScript"}]

# In future sessions, load user memories and inject into system prompt
user_memories = get_user_memories(user_id)
system_prompt = f"User preferences: {format_memories(user_memories)}"
```

## Memory Layers in Practice

| Layer | Scope | Storage | Trigger |
|---|---|---|---|
| Message history | Per session | DB `messages` table | Every turn |
| Conversation summary | Per session | DB `sessions.summary` | Every N turns |
| Long-term memory | Per user | DB `conversation_memory` | Every turn (LLM extracts) |

## Query Rewriting for Context-Aware Retrieval

When a user asks a follow-up like "What about its limitations?", the retriever needs the full context to find relevant documents. The query must be rewritten:

```
Original:   "What about its limitations?"
Rewritten:  "What are the limitations of Retrieval-Augmented Generation (RAG)?"
```

Query rewriting resolves pronouns and references using conversation history before retrieval.

## Summarization Strategy

### When to Summarize
Trigger summarization every N turns (configurable):
```python
if session.turn_count % SUMMARY_TRIGGER_TURNS == 0:
    summary = await summarize_conversation(messages, existing_summary)
    await save_summary(session_id, summary)
```

### Incremental Summarization
Pass the existing summary along with new messages to avoid reprocessing old turns:
```
Existing summary + [new messages] → Updated summary
```

This is O(new_messages) rather than O(all_messages).

## Cross-Session Personalization

Extract persistent user facts across sessions:

```python
# Things worth remembering long-term:
# - "User is building a FastAPI microservice"
# - "User prefers concise code examples"
# - "User works in Python 3.11"

# Things not worth remembering:
# - "User asked about list comprehensions" (transient)
# - "User said thanks" (conversational)
```

In future sessions, inject long-term memories into the system prompt for personalized responses.
