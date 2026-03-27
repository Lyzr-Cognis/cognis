![Cognis — Lightweight memory for AI agents](https://github.com/user-attachments/assets/78bb56fb-ebcb-4800-a4d4-2a56964630b4)

### Lightweight memory for AI agents

![PyPI](https://img.shields.io/pypi/v/lyzr-cognis?style=flat-square&color=BA998C) ![Python](https://img.shields.io/badge/python-%3E%3D3.10-5E403E?style=flat-square) ![License](https://img.shields.io/badge/license-MIT-27272A?style=flat-square) ![Deps](https://img.shields.io/badge/deps-3-D4A574?style=flat-square)

---

## Features

- **Hybrid search** — Two-stage Matryoshka vector search (256D shortlist, 768D rerank) + BM25 keyword matching via SQLite FTS5, fused with RRF (70/30 split, tuned from ablation studies)
- **Zero infrastructure** — Everything runs in-process. Qdrant local mode (file-backed) + SQLite. No Docker, no servers, just `pip install lyzr-cognis`
- **Smart extraction** — LLM-powered fact extraction with 13 auto-tagged categories, memory versioning (ADD/UPDATE/DELETE), and name-aware facts
- **Session management** — `owner_id` + `agent_id` + `session_id` scoping that matches the hosted Cognis platform. Memories are global, messages are session-scoped
- **Fast retrieval** — ~500ms search latency (embedding API bottleneck), ~4ms with cache hits

## Quick Start

**1. Install**

```bash
pip install lyzr-cognis
```

**2. Set your API keys**

```bash
export GEMINI_API_KEY="your-gemini-key"    # For embeddings
export OPENAI_API_KEY="your-openai-key"    # For extraction (gpt-4.1-mini)
```

**3. Use it**

```python
from cognis import Cognis

m = Cognis(owner_id="user_1")

# Add conversation messages — facts are extracted automatically
m.add([
    {"role": "user", "content": "My name is Alice and I work at Google as a data scientist."},
    {"role": "user", "content": "I love hiking and I'm a huge fan of Taylor Swift."},
])

# Search memories
resp = m.search("Where does Alice work?")
for r in resp["results"]:
    print(f"  {r['content']}  (score: {r['score']})")

# Get context for your LLM (short-term messages + long-term memories)
ctx = m.get_context([{"role": "user", "content": "Tell me about myself"}])
print(ctx["context_string"])

# List all extracted memories
for mem in m.get_all()["memories"]:
    cat = mem["metadata"]["category"]
    print(f"  [{cat}] {mem['content']}")

m.close()
```

## Architecture
![architecture](https://github.com/user-attachments/assets/671ccbd0-63a6-4277-8a4a-63c67e9dcd23)


## API Reference

### `Cognis(owner_id, agent_id, session_id, data_dir, config)`

Initialize a memory instance. At least one of `owner_id`, `agent_id`, or `session_id` is required.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `gemini_api_key` | `str` | `$GEMINI_API_KEY` | Gemini API key for embeddings |
| `owner_id` | `str` | — | Memory owner identifier |
| `agent_id` | `str` | `None` | Agent identifier |
| `session_id` | `str` | auto-generated | Session identifier |
| `data_dir` | `str` | `~/.cognis` | Local storage directory |
| `config` | `CognisConfig` | defaults | Configuration overrides |

### Methods

All methods accept optional `owner_id`, `agent_id`, `session_id` overrides per call.

| Method | Returns | Description |
|--------|---------|-------------|
| `add(messages)` | `{"success", "memories", "session_message_count"}` | Add messages and extract memories |
| `search(query, limit)` | `{"success", "results", "count", "query"}` | Hybrid RRF search |
| `get(memory_id)` | `{"success", "memory"}` | Get single memory by ID |
| `get_all(limit, offset)` | `{"success", "memories", "total", "limit", "offset"}` | List all memories |
| `delete(memory_id)` | `{"success", "message"}` | Delete a memory |
| `get_context(messages)` | `{"short_term", "long_term", "context_string"}` | Get LLM-ready context |
| `clear()` | `{"success", "message"}` | Clear all memories |
| `count()` | `int` | Count current memories |

### Session Management

```python
m.new_session()         # Generate new session ID
m.set_session("ses_x")  # Switch session
m.set_owner("user_2")   # Switch owner
m.set_agent("agent_2")  # Switch agent
```

**Scoping rules:**
- **Extracted memories** are global to `(owner_id, agent_id)` — persist across sessions
- **Raw messages** are scoped to `(owner_id, agent_id, session_id)` — session-local
- **Search** returns global memories + current session messages

### Per-call ID overrides

Pass IDs at call time instead of (or in addition to) init:

```python
m = Cognis(session_id="ses_1")

# Different owners per call
m.add(messages, owner_id="alice", agent_id="bot_1")
m.add(messages, owner_id="bob", agent_id="bot_1")

# Search scoped to specific owner
m.search("query", owner_id="alice")

# Context for specific session
m.get_context(messages, session_id="ses_morning")
```

## Configuration

```python
from cognis import Cognis, CognisConfig

config = CognisConfig(
    embedding_model="gemini/gemini-embedding-2-preview",
    embedding_full_dim=768,
    embedding_small_dim=256,
    vector_weight=0.70,       # RRF: 70% vector
    bm25_weight=0.30,         # RRF: 30% BM25
    rrf_k=10,                 # RRF constant
    similarity_threshold=0.3,
    llm_model="gpt-4.1-mini", # For fact extraction
)

m = Cognis(config=config, owner_id="user_1", data_dir="./my_data")
```

## Memory Categories

Extracted facts are auto-categorized into 13 categories:

`identity` `relationships` `work_career` `learning` `wellness` `lifestyle` `interests` `preferences` `plans_goals` `experiences` `opinions` `context` `misc`

## Dependencies

Only 3 core dependencies:

| Package | Size | Purpose |
|---------|------|---------|
| `qdrant-client` | 3 MB | Vector store (local mode, no server) |
| `litellm` | 55 MB | LLM + embedding provider abstraction |
| `pydantic` | 7 MB | Config validation |

SQLite is Python stdlib. Total install: ~156 MB. Wheel size: 33 KB.

## Development

```bash
uv venv --python 3.12 .venv
uv pip install -e ".[dev]" python-dotenv openai
uv run pytest tests/ -v
```

## License

[MIT](LICENSE)

---

Built by [Lyzr](https://lyzr.ai)
