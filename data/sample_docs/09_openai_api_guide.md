# OpenAI API — Developer Reference

## Authentication

```python
from openai import OpenAI

client = OpenAI(api_key="sk-...")  # or set OPENAI_API_KEY env var
```

## Chat Completions

```python
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain vector embeddings."},
    ],
    temperature=0.3,
    max_tokens=1024,
)
answer = response.choices[0].message.content
```

## Streaming

```python
stream = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Tell me about RAG"}],
    stream=True,
)
for chunk in stream:
    token = chunk.choices[0].delta.content or ""
    print(token, end="", flush=True)
```

## Embeddings

```python
response = client.embeddings.create(
    model="text-embedding-ada-002",
    input=["Hello world", "How does RAG work?"],
)
vectors = [item.embedding for item in response.data]
# Each vector is a list of 1536 floats
```

## Models Comparison

| Model | Context | Speed | Best For |
|---|---|---|---|
| `gpt-4o` | 128k | Moderate | Complex reasoning |
| `gpt-4o-mini` | 128k | Fast | Most tasks, cost-efficient |
| `gpt-4-turbo` | 128k | Moderate | Long documents |
| `gpt-3.5-turbo` | 16k | Very fast | Simple Q&A |

## Token Counting

```python
import tiktoken

enc = tiktoken.encoding_for_model("gpt-4o-mini")
tokens = enc.encode("Hello, how does RAG work?")
print(f"Token count: {len(tokens)}")
```

## Rate Limits and Retries

```python
from openai import RateLimitError
import time

def call_with_retry(messages, max_retries=3):
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
            )
        except RateLimitError:
            wait = 2 ** attempt
            time.sleep(wait)
    raise Exception("Max retries exceeded")
```

## Async Usage with LangChain

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Async chat model
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
response = await llm.ainvoke("What is LangGraph?")

# Async embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
vectors = await embeddings.aembed_documents(["text one", "text two"])
query_vec = await embeddings.aembed_query("search query")
```

## Cost Estimation

| Model | Input (per 1M tokens) | Output (per 1M tokens) |
|---|---|---|
| `gpt-4o` | $2.50 | $10.00 |
| `gpt-4o-mini` | $0.15 | $0.60 |
| `text-embedding-ada-002` | $0.10 | — |
| `text-embedding-3-small` | $0.02 | — |

For a RAG system processing 1000 queries/day at ~2000 tokens each:
- `gpt-4o-mini`: ~$0.30/day for generation
- `text-embedding-ada-002`: ~$0.10/day for query embeddings

## Best Practices

1. **Set temperature low (0.0–0.3)** for factual Q&A to reduce hallucination
2. **Use system prompts** to constrain the model to provided context only
3. **Stream responses** for better perceived latency in user-facing apps
4. **Cache embeddings** — document embeddings are deterministic; don't re-embed unchanged chunks
5. **Monitor token usage** — log `response.usage` to track costs per request
