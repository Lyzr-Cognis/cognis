"""
Base embedder interface for Cognis.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class EmbeddingResult:
    """Result of embedding a text, containing vectors at multiple dimensions."""
    embeddings: Dict[int, List[float]] = field(default_factory=dict)

    def get(self, dim: int) -> Optional[List[float]]:
        return self.embeddings.get(dim)


class BaseEmbedder(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    def embed_query(self, text: str) -> EmbeddingResult:
        """Embed a search query. Returns vectors at configured dimensions."""
        ...

    @abstractmethod
    def embed_document(self, text: str) -> EmbeddingResult:
        """Embed a document for storage. Returns vectors at configured dimensions."""
        ...

    @abstractmethod
    def embed_documents_batch(self, texts: List[str]) -> List[EmbeddingResult]:
        """Batch embed multiple documents."""
        ...
