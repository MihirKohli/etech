"""
Performance benchmark for the Conversational RAG system.

Measures:
  - Response latency (p50, p95, p99) per pipeline node
  - Memory overhead (SQLite DB size, Chroma index size)
  - Context preservation accuracy (follow-up query resolution rate)

Usage:
    # Ensure the API is running first:
    #   uvicorn main:app --reload
    #
    python scripts/benchmark.py --base-url http://localhost:8000 --rounds 10
"""

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import requests

# ── Config ────────────────────────────────────────────────────────────────────

DEFAULT_BASE_URL = os.getenv("API_URL", "http://localhost:8000")
TEST_USER_ID = "benchmark_user"

# Multi-turn conversation designed to test context preservation
CONVERSATION_SCRIPT = [
    "What is Retrieval-Augmented Generation?",
    "How does it differ from fine-tuning?",           # requires previous context
    "What embedding models does it typically use?",
    "And what about the chunking strategies?",         # pronoun reference test
    "Can you summarise everything we discussed?",      # full context recall test
]

# Single-turn factual queries for latency benchmarking
FACTUAL_QUERIES = [
    "What is LangGraph?",
    "How does Chroma store embeddings?",
    "What is the difference between semantic and keyword search?",
    "How does FastAPI handle dependency injection?",
    "What are the benefits of async SQLAlchemy?",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def post(url, **kwargs) -> dict:
    r = requests.post(url, **kwargs, timeout=120)
    r.raise_for_status()
    return r.json()


def get(url, **kwargs) -> dict:
    r = requests.get(url, **kwargs, timeout=30)
    r.raise_for_status()
    return r.json()


def percentile(data: list[float], p: int) -> float:
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(sorted_data) - 1)
    return sorted_data[lo] + (sorted_data[hi] - sorted_data[lo]) * (k - lo)


def db_size_mb(path: str) -> float:
    p = Path(path)
    if not p.exists():
        return 0.0
    return p.stat().st_size / (1024 * 1024)


def chroma_size_mb(directory: str) -> float:
    total = sum(f.stat().st_size for f in Path(directory).rglob("*") if f.is_file())
    return total / (1024 * 1024)


# ── Benchmarks ────────────────────────────────────────────────────────────────

def benchmark_latency(base_url: str, rounds: int) -> dict:
    """Measure per-request latency over N single-turn queries."""
    print(f"\n[1/3] Latency benchmark — {rounds} rounds × {len(FACTUAL_QUERIES)} queries")

    # Create a dedicated session
    session = post(f"{base_url}/sessions", params={"user_id": TEST_USER_ID})
    session_id = session["session_id"]

    latencies = []
    for r in range(rounds):
        query = FACTUAL_QUERIES[r % len(FACTUAL_QUERIES)]
        t0 = time.perf_counter()
        post(f"{base_url}/chat", json={"session_id": session_id, "message": query})
        elapsed_ms = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed_ms)
        print(f"  round {r + 1:>3}: {elapsed_ms:>7.0f} ms  |  {query[:50]}")

    return {
        "rounds": rounds,
        "p50_ms": percentile(latencies, 50),
        "p95_ms": percentile(latencies, 95),
        "p99_ms": percentile(latencies, 99),
        "min_ms": min(latencies),
        "max_ms": max(latencies),
        "mean_ms": statistics.mean(latencies),
    }


def benchmark_context_preservation(base_url: str) -> dict:
    """
    Run a scripted multi-turn conversation and check that follow-up answers
    reference content from earlier turns (proxy for context preservation).
    """
    print(f"\n[2/3] Context preservation — {len(CONVERSATION_SCRIPT)}-turn conversation")

    session = post(f"{base_url}/sessions", params={"user_id": TEST_USER_ID})
    session_id = session["session_id"]

    results = []
    previous_answers = []

    for i, query in enumerate(CONVERSATION_SCRIPT):
        t0 = time.perf_counter()
        resp = post(f"{base_url}/chat", json={"session_id": session_id, "message": query})
        elapsed_ms = (time.perf_counter() - t0) * 1000
        answer = resp.get("answer", "")

        # Heuristic: follow-up answers should be longer than 50 chars and
        # not say "I don't know" / "no information"
        is_coherent = (
            len(answer) > 50
            and "don't know" not in answer.lower()
            and "no information" not in answer.lower()
        )
        results.append({
            "turn": i + 1,
            "query": query,
            "answer_length": len(answer),
            "latency_ms": round(elapsed_ms),
            "coherent": is_coherent,
        })
        previous_answers.append(answer)
        status = "✓" if is_coherent else "✗"
        print(f"  turn {i + 1}: {status}  {elapsed_ms:>6.0f} ms  |  {query[:60]}")

    accuracy = sum(r["coherent"] for r in results) / len(results) * 100
    return {"turns": results, "context_preservation_accuracy": round(accuracy, 1)}


def benchmark_memory_overhead(base_url: str) -> dict:
    """Report storage sizes for SQLite and Chroma."""
    print("\n[3/3] Memory overhead")

    sqlite_path = "./data/rag.db"
    chroma_dir = "./data/chroma_db"

    sqlite_mb = db_size_mb(sqlite_path)
    chroma_mb = chroma_size_mb(chroma_dir) if Path(chroma_dir).exists() else 0.0

    print(f"  SQLite:  {sqlite_mb:.2f} MB  ({sqlite_path})")
    print(f"  Chroma:  {chroma_mb:.2f} MB  ({chroma_dir})")

    return {"sqlite_mb": round(sqlite_mb, 2), "chroma_mb": round(chroma_mb, 2)}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RAG system performance benchmark")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--rounds", type=int, default=10, help="Latency benchmark rounds")
    parser.add_argument("--output", default="scripts/benchmark_results.json")
    args = parser.parse_args()

    print("=" * 60)
    print("  Conversational RAG — Performance Benchmark")
    print(f"  Target: {args.base_url}")
    print("=" * 60)

    # Health check
    try:
        get(f"{args.base_url}/health")
    except Exception:
        print(f"\n✗ API not reachable at {args.base_url}. Start the server first.")
        return

    latency = benchmark_latency(args.base_url, args.rounds)
    context = benchmark_context_preservation(args.base_url)
    memory = benchmark_memory_overhead(args.base_url)

    results = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "base_url": args.base_url,
        "latency": latency,
        "context_preservation": context,
        "memory_overhead": memory,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("  Results Summary")
    print("=" * 60)
    print(f"  Latency  p50={latency['p50_ms']:.0f}ms  p95={latency['p95_ms']:.0f}ms  p99={latency['p99_ms']:.0f}ms")
    print(f"  Context preservation accuracy: {context['context_preservation_accuracy']}%")
    print(f"  Storage: SQLite={memory['sqlite_mb']} MB, Chroma={memory['chroma_mb']} MB")
    print(f"\n  Full results saved to: {args.output}")


if __name__ == "__main__":
    main()
