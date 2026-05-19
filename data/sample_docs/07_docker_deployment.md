# Docker Deployment Guide

## Overview

Docker packages applications and their dependencies into portable containers. Docker Compose orchestrates multiple containers (API, UI, database) as a single service.

## Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Multi-stage Build (smaller image)

```dockerfile
FROM python:3.11-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## docker-compose.yml

```yaml
services:
  api:
    build: .
    ports:
      - "8000:8000"
    env_file:
      - .env
    volumes:
      - chroma_data:/app/data/chroma_db
      - sqlite_data:/app/data
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 10s
      retries: 3

  streamlit:
    build: .
    ports:
      - "8501:8501"
    depends_on:
      - api
    command: streamlit run streamlit_ui.py --server.port 8501
    environment:
      - API_URL=http://api:8000

  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_USER: rag
      POSTGRES_PASSWORD: rag
      POSTGRES_DB: ragdb
    volumes:
      - pg_data:/var/lib/postgresql/data

volumes:
  chroma_data:
  sqlite_data:
  pg_data:
```

## Common Docker Commands

```bash
# Build and start all services
docker compose up --build

# Run in background
docker compose up -d

# View logs
docker compose logs -f api

# Stop all services
docker compose down

# Remove volumes (wipes data)
docker compose down -v
```

## Environment Variables

Pass secrets via `.env` file (never bake into the image):

```bash
# .env
OPENAI_API_KEY=sk-...
DATABASE_URL=sqlite+aiosqlite:///./data/rag.db
```

```yaml
# docker-compose.yml
services:
  api:
    env_file:
      - .env
```

## Volumes and Persistence

| Volume | Purpose |
|---|---|
| `chroma_data` | Chroma vector store (survives restarts) |
| `sqlite_data` | SQLite database file |
| `upload_data` | Uploaded document files |
| `pg_data` | PostgreSQL data directory |

## Health Checks

```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
  interval: 10s
  timeout: 5s
  retries: 3
  start_period: 15s
```

The `depends_on` directive with `condition: service_healthy` ensures dependent services wait until the API is ready:

```yaml
streamlit:
  depends_on:
    api:
      condition: service_healthy
```

## Resource Limits

```yaml
services:
  api:
    deploy:
      resources:
        limits:
          cpus: "1.0"
          memory: 1G
```
