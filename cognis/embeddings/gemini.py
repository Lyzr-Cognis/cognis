"""
Gemini embedding via LiteLLM.

Uses litellm.embedding() for provider-agnostic embedding calls.
Single API call at full_dim, then Matryoshka truncation for small_dim.
"""

import math
import os
import logging
from typing import List

from cognis.embeddings.base import BaseEmbedder, EmbeddingResult
from cognis.utils import LRUCache

logger = logging.getLogger(__name__)


def _truncate_and_normalize(vec: List[float], dim: int) -> List[float]:
    """Truncate vector to dim and L2-normalize (Matryoshka property)."""
    truncated = vec[:dim]
    norm = math.sqrt(sum(x * x for x in truncated))
    if norm > 0:
        return [x / norm for x in truncated]
    return truncated


class GeminiEmbedder(BaseEmbedder):
    """
    Gemini embedding via LiteLLM.

    Uses litellm.embedding() which supports gemini/ prefix for Gemini models.
    Single API call at full_dim (768), then truncate+normalize to small_dim (256).
    """

    def __init__(
        self,
        api_key: str = None,
        model: str = "gemini/gemini-embedding-001",
        full_dim: int = 768,
        small_dim: int = 256,
    ):
        resolved_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not resolved_key:
            raise ValueError("Gemini API key required. Set GEMINI_API_KEY env var or pass api_key.")

        os.environ["GEMINI_API_KEY"] = resolved_key
        self._model = model
        self._full_dim = full_dim
        self._small_dim = small_dim
        self._cache = LRUCache(max_size=500, ttl_seconds=3600)

    def _embed(self, text: str) -> EmbeddingResult:
        """Embed text with single LiteLLM call, derive small_dim via truncation."""
        import litellm

        cache_key = f"emb:{text[:200]}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        response = litellm.embedding(
            model=self._model,
            input=[text],
            dimensions=self._full_dim,
        )

        full_vec = response.data[0]["embedding"]
        small_vec = _truncate_and_normalize(full_vec, self._small_dim)

        result = EmbeddingResult(embeddings={
            self._full_dim: full_vec,
            self._small_dim: small_vec,
        })

        self._cache.set(cache_key, result)
        return result

    def embed_query(self, text: str) -> EmbeddingResult:
        return self._embed(text)

    def embed_document(self, text: str) -> EmbeddingResult:
        return self._embed(text)

    def embed_documents_batch(self, texts: List[str]) -> List[EmbeddingResult]:
        return [self.embed_document(t) for t in texts]
