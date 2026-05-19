# LangChain — Developer Overview

## What is LangChain?

LangChain is an open-source framework for building applications powered by Large Language Models. It provides composable abstractions for chaining together prompts, models, retrievers, memory, and tools into end-to-end pipelines.

## Core Abstractions

### LLMs and Chat Models
```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
response = llm.invoke("What is LangChain?")
```

### Prompt Templates
```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Context: {context}"),
    ("human", "{question}"),
])
chain = prompt | llm
```

### Document Loaders
LangChain provides loaders for many formats:

| Loader | Format |
|---|---|
| `PyPDFLoader` | PDF files |
| `TextLoader` | Plain text / Markdown |
| `BSHTMLLoader` | HTML pages |
| `WebBaseLoader` | Web URLs |
| `CSVLoader` | CSV files |

```python
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("document.pdf")
docs = loader.load()  # returns list[Document]
```

### Text Splitters
```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""],
)
chunks = splitter.split_documents(docs)
```

### Vector Stores
```python
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=OpenAIEmbeddings(),
    persist_directory="./chroma_db",
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
```

### LCEL — LangChain Expression Language

LCEL uses the `|` operator to compose chains:

```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
answer = chain.invoke("What is RAG?")
```

### Memory
```python
from langchain.memory import ConversationSummaryBufferMemory

memory = ConversationSummaryBufferMemory(
    llm=llm,
    max_token_limit=1000,
    return_messages=True,
)
```

## LangChain vs LangGraph

| Feature | LangChain | LangGraph |
|---|---|---|
| Flow type | Linear chains | Directed graphs with cycles |
| State | Passed through chain | Shared `TypedDict` state |
| Branching | Limited | Full conditional edges |
| Best for | Simple pipelines | Complex multi-agent systems |

## Async Support

All major LangChain components support async via `ainvoke`, `astream`:

```python
# Async invocation
response = await chain.ainvoke({"question": "What is LangChain?"})

# Async streaming
async for token in chain.astream({"question": "Explain RAG"}):
    print(token, end="", flush=True)
```
