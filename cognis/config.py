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
