"""
Configuration for Cognis memory system.

Tuned defaults from ablation studies on LoCoMo benchmark.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional


# 13 unified categories (from production cognis)
DEFAULT_CATEGORIES = {
    "identity": "Personal identity, name, demographics, age, location",
    "relationships": "Family, friends, social connections, pets",
    "work_career": "Job, profession, workplace, colleagues, business",
    "learning": "Education, skills, knowledge, certifications, languages",
    "wellness": "Health, fitness, medical conditions, diet, exercise",
    "lifestyle": "Daily habits, routines, sleep, transportation",
    "interests": "Hobbies, passions, entertainment, sports, games",
    "preferences": "Likes, dislikes, choices, favorites, style",
    "plans_goals": "Future plans, aspirations, goals, dreams, intentions",
    "experiences": "Past events, travel, memories, experiences",
    "opinions": "Views, beliefs, attitudes, political, philosophical",
    "context": "Session-specific context, current tasks, immediate needs",
    "misc": "Anything that doesn't fit other categories",
}

# Sector-specific decay rates
SECTOR_DECAY_RATES = {
    "episodic": 0.15,
    "semantic": 0.05,
    "procedural": 0.02,
    "emotional": 0.10,
    "reflective": 0.08,
}


@dataclass
class CognisConfig:
    """Configuration for Cognis lightweight memory."""

    # Embedding (Gemini 2)
    embedding_model: str = "gemini/gemini-embedding-2-preview"
    embedding_full_dim: int = 768
    embedding_small_dim: int = 256

    # RRF fusion weights (optimal per ablation study: 70/30 vector/BM25)
    vector_weight: float = 0.70
    bm25_weight: float = 0.30
    rrf_k: int = 10
    similarity_threshold: float = 0.3
    shortlist_size: int = 200

    # Recency boost (from provider.py:2030-2041)
    recency_boost_weight: float = 0.25
    recency_half_life_seconds: float = 120.0

    # Temporal reasoning
    enable_temporal_decay: bool = True
    enable_temporal_query_detection: bool = True

    # LLM for extraction
    llm_model: str = "gpt-4.1-mini"

    # Memory scoping
    global_memory: bool = False

    # Qdrant local collections
    qdrant_collection_full: str = "memories_full"
    qdrant_collection_small: str = "memories_small"

    # Extraction settings
    unified_operation_top_k: int = 10

    # Update/dedup thresholds (from production config)
    update_similarity_threshold: float = 0.85
    add_similarity_threshold: float = 0.70

    # Categories
    categories: Dict[str, str] = field(default_factory=lambda: dict(DEFAULT_CATEGORIES))

    # Immediate recall
    enable_immediate_recall: bool = True
    immediate_recall_ttl_hours: int = 48

    # Optional remote cross-encoder reranking. Disabled unless an endpoint is configured.
    remote_rerank_url: Optional[str] = None
    rerank_provider: str = ""
    remote_rerank_timeout: float = 5.0
    enable_relevance_gate: bool = False
    relevance_gate_threshold: float = 0.05  # legacy absolute cutoff (unused when noise-floor gate active)
    relevance_gate_noise_floor: float = 1e-4  # top rerank score below this => no plausibly relevant memory
    relevance_gate_relative_frac: float = 0.05  # keep items scoring >= frac * top score
    relevance_gate_min_keep: int = 10  # floor of candidates kept once noise floor is cleared
    enable_chain_attachment: bool = True
    chain_max_ancestors: int = 3

    # v3 dual-tier serving: raw verbatim evidence alongside extracted facts.
    enable_raw_evidence: bool = False
    raw_evidence_limit: int = 20  # candidate raw messages fused into ranking
    raw_evidence_window: int = 2  # neighbor turns attached around a served raw hit
    enable_conflict_attachment: bool = True  # surface conflicts_with pairs on serve
    global_query_limit: int = 30  # widened limit for summarization/global queries

    @classmethod
    def from_env(cls, **overrides) -> "CognisConfig":
        """Build a config honoring COGNIS_* environment variables.

        Recognized: COGNIS_RERANK_PROVIDER, COGNIS_REMOTE_RERANK_URL,
        COGNIS_REMOTE_RERANK_TIMEOUT, COGNIS_ENABLE_RELEVANCE_GATE,
        COGNIS_RELEVANCE_GATE_THRESHOLD, COGNIS_ENABLE_CHAIN_ATTACHMENT.
        Explicit ``overrides`` win over environment values.
        """
        import os

        def _b(v: str) -> bool:
            return v.strip().lower() in ("1", "true", "yes", "on")

        env_map = {}
        if os.getenv("COGNIS_RERANK_PROVIDER") is not None:
            env_map["rerank_provider"] = os.environ["COGNIS_RERANK_PROVIDER"]
        if os.getenv("COGNIS_REMOTE_RERANK_URL"):
            env_map["remote_rerank_url"] = os.environ["COGNIS_REMOTE_RERANK_URL"]
        if os.getenv("COGNIS_REMOTE_RERANK_TIMEOUT"):
            env_map["remote_rerank_timeout"] = float(os.environ["COGNIS_REMOTE_RERANK_TIMEOUT"])
        if os.getenv("COGNIS_ENABLE_RELEVANCE_GATE") is not None:
            env_map["enable_relevance_gate"] = _b(os.environ["COGNIS_ENABLE_RELEVANCE_GATE"])
        if os.getenv("COGNIS_RELEVANCE_GATE_THRESHOLD"):
            env_map["relevance_gate_threshold"] = float(os.environ["COGNIS_RELEVANCE_GATE_THRESHOLD"])
        if os.getenv("COGNIS_RELEVANCE_GATE_NOISE_FLOOR"):
            env_map["relevance_gate_noise_floor"] = float(os.environ["COGNIS_RELEVANCE_GATE_NOISE_FLOOR"])
        if os.getenv("COGNIS_RELEVANCE_GATE_RELATIVE_FRAC"):
            env_map["relevance_gate_relative_frac"] = float(os.environ["COGNIS_RELEVANCE_GATE_RELATIVE_FRAC"])
        if os.getenv("COGNIS_RELEVANCE_GATE_MIN_KEEP"):
            env_map["relevance_gate_min_keep"] = int(os.environ["COGNIS_RELEVANCE_GATE_MIN_KEEP"])
        if os.getenv("COGNIS_ENABLE_CHAIN_ATTACHMENT") is not None:
            env_map["enable_chain_attachment"] = _b(os.environ["COGNIS_ENABLE_CHAIN_ATTACHMENT"])
        if os.getenv("COGNIS_ENABLE_RAW_EVIDENCE") is not None:
            env_map["enable_raw_evidence"] = _b(os.environ["COGNIS_ENABLE_RAW_EVIDENCE"])
        if os.getenv("COGNIS_RAW_EVIDENCE_LIMIT"):
            env_map["raw_evidence_limit"] = int(os.environ["COGNIS_RAW_EVIDENCE_LIMIT"])
        if os.getenv("COGNIS_GLOBAL_QUERY_LIMIT"):
            env_map["global_query_limit"] = int(os.environ["COGNIS_GLOBAL_QUERY_LIMIT"])
        env_map.update(overrides)
        return cls(**env_map)
