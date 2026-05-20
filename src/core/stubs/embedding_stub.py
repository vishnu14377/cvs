"""Deterministic embeddings stub for CI integration tests."""

from __future__ import annotations

from langchain_core.embeddings import Embeddings
from src.core.logger import get_logger

logger = get_logger(__name__)


class StubEmbeddings(Embeddings):
    """Return constant-unit vectors of a fixed dimension.

    All vectors are identical, so under cosine distance every document
    is equidistant from the query — pgvector returns results in stable
    insertion order rather than NaN (which zero vectors would produce).
    The tiny non-zero value keeps the vector norm non-zero so the cosine
    denominator is well-defined.
    """

    # Small non-zero constant: avoids NaN from ||v|| = 0 in cosine distance
    # while remaining deterministic across queries and documents.
    _FILL = 1e-3

    def __init__(self, dimension: int = 768) -> None:
        self._dimension = dimension
        logger.info("StubEmbeddings initialized (dim=%d)", dimension)

    def embed_query(self, text: str) -> list[float]:
        return [self._FILL] * self._dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[self._FILL] * self._dimension for _ in texts]

    async def aembed_query(self, text: str) -> list[float]:
        return [self._FILL] * self._dimension

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[self._FILL] * self._dimension for _ in texts]
