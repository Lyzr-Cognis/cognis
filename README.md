# Cognis

Lightweight, local-first memory for LLM agents. Zero external services — everything runs in-process.

## Install

```bash
pip install lyzr-memory-lite
```

## Quick Start

```python
from cognis import Cognis

memory = Cognis(gemini_api_key="your-key", owner_id="user_123")

# Add memories from conversation
memory.add([
    {"role": "user", "content": "My name is Alice and I work at Google"},
    {"role": "assistant", "content": "Nice to meet you, Alice!"},
    {"role": "user", "content": "I love hiking and photography"},
])

# Search memories
results = memory.search("What does Alice do for work?")
for r in results:
    print(f"{r['content']} (score: {r['score']})")

# Get context for LLM
context = memory.get_context([{"role": "user", "content": "Tell me about myself"}])
print(context["context_string"])

# List all memories
for m in memory.get_all():
    print(f"  {m['memory_id']}: {m['content']}")

memory.close()
```

## Architecture

```
Query → Gemini Embedding (768D + 256D)
         ↓
    ┌────┴────────────┬──────────────┐
    │                 │              │
 256D HNSW        SQLite FTS5    256D Messages
 Shortlist (200)   BM25 Search   Immediate Recall
    │                 │              │
 768D Rerank         │              │
    │                 │              │
    └────┬────────────┴──────────────┘
         ↓
    RRF Fusion (70% Vector + 30% BM25, k=10)
         ↓
    Recency Boost + Temporal Boosting
         ↓
    Deduplicated Results
```

**Search latency: ~100-150ms locally** (embedding API call is the bottleneck).

## How It Works

- **Qdrant local mode**: In-process vector search, file-backed, no server
- **SQLite FTS5**: BM25 keyword search with Porter stemming
- **Gemini 2 Embeddings**: Matryoshka-style dual dimensions (768D + 256D)
- **Hybrid RRF fusion**: 70% vector + 30% BM25 — optimal from ablation studies
- **LLM-based extraction**: Gemini 2.0 Flash for fact extraction and memory operations

## Configuration

```python
from cognis import Cognis, CognisConfig

config = CognisConfig(
    embedding_model="gemini-embedding-exp-03-07",
    embedding_full_dim=768,
    embedding_small_dim=256,
    vector_weight=0.70,
    bm25_weight=0.30,
    similarity_threshold=0.3,
    llm_model="gemini-2.0-flash",
)

memory = Cognis(config=config, data_dir="./my_data")
```

## Dependencies

Only 3 core dependencies:
- `qdrant-client` — Vector store (local mode)
- `google-genai` — Gemini embeddings + LLM
- `pydantic` — Config validation

SQLite is Python stdlib.
