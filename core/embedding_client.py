"""
Embedding client singleton using GoogleGenerativeAIEmbeddings with Vertex AI backend.

Provides a thread-safe singleton instance of GoogleGenerativeAIEmbeddings for consistent
embedding access across the application. Uses Application Default Credentials (ADC) for
authentication with Vertex AI backend.

Usage:
    from src.core.embedding_client import EmbeddingClient, get_embedding_client

    # Get singleton instance
    client = EmbeddingClient()

    # Embed a single query
    query_embedding = client.embed_query("What is machine learning?")

    # Embed multiple documents
    doc_embeddings = client.embed_documents(["doc1 content", "doc2 content"])

    # Get embeddings instance for use with PGVector
    embeddings = client.embeddings
    vector_store = create_vector_store(embeddings)

Authentication:
    Uses Application Default Credentials (ADC) via Vertex AI backend.
    If authentication fails, run 'gcloud auth application-default login'.
"""

from __future__ import annotations

import threading

from langchain_core.embeddings import Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from src.core.config import vectorstore_config
from src.core.logger import get_logger
from src.core.stubs import StubEmbeddings

logger = get_logger(__name__)


class EmbeddingClient:
    """
    Thread-safe singleton embedding client using GoogleGenerativeAIEmbeddings.

    Uses Vertex AI backend with Application Default Credentials (ADC).
    The singleton pattern ensures only one client instance exists across
    the entire application, avoiding redundant initialization.

    Example:
        >>> client = EmbeddingClient()
        >>> embedding = client.embed_query("What is AI?")
        >>> print(len(embedding))  # embedding dimension
        768

        >>> # Use with PGVector
        >>> from core.pgvector_store import create_vector_store
        >>> store = create_vector_store(client.embeddings)
    """

    _instance: EmbeddingClient | None = None
    _lock: threading.Lock = threading.Lock()
    _initialized: bool = False

    def __new__(cls) -> EmbeddingClient:
        """Create or return the singleton instance."""
        if cls._instance is None:
            with cls._lock:
                # Double-checked locking for thread safety
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize the embedding client (only runs once)."""
        # Skip if already initialized (singleton pattern)
        if EmbeddingClient._initialized:
            return

        with EmbeddingClient._lock:
            if EmbeddingClient._initialized:
                return

            self._model_id = vectorstore_config.EMBEDDING_MODEL_ID
            self._project_id = vectorstore_config.GCP_PROJECT
            self._location = vectorstore_config.GCP_REGION

            logger.info(
                "Initializing EmbeddingClient: model=%s, project=%s, location=%s",
                self._model_id,
                self._project_id,
                self._location,
            )

            # Create embeddings backend based on VERTEX_AI_MODE
            self._embeddings: Embeddings
            if vectorstore_config.VERTEX_AI_MODE == "stub":
                logger.warning(
                    "VERTEX_AI_MODE=stub — EmbeddingClient using StubEmbeddings "
                    "(no Vertex AI calls will be made)"
                )
                self._embeddings = StubEmbeddings(
                    dimension=vectorstore_config.EMBEDDING_DIMENSION,
                )
            else:
                self._embeddings = GoogleGenerativeAIEmbeddings(
                    model=self._model_id,
                    project=self._project_id,
                    location=self._location,
                    vertexai=True,  # Use Vertex AI backend with ADC
                )

            EmbeddingClient._initialized = True
            mode_desc = (
                "stub (StubEmbeddings)"
                if vectorstore_config.VERTEX_AI_MODE == "stub"
                else "Vertex AI backend with ADC"
            )
            logger.info("EmbeddingClient initialized successfully (%s)", mode_desc)

    @property
    def embeddings(self) -> Embeddings:
        """
        Get the underlying GoogleGenerativeAIEmbeddings instance.

        Use this property when you need to pass embeddings to other components
        like PGVector or LangChain chains.

        Returns:
            The GoogleGenerativeAIEmbeddings instance.
        """
        return self._embeddings

    @property
    def model_id(self) -> str:
        """Get the embedding model ID."""
        return self._model_id

    @property
    def project_id(self) -> str | None:
        """Get the GCP project ID."""
        return self._project_id

    @property
    def location(self) -> str:
        """Get the GCP location/region."""
        return self._location

    def embed_query(self, text: str) -> list[float]:
        """
        Embed a single query text.

        Uses RETRIEVAL_QUERY task type by default, optimized for search queries.

        Args:
            text: The query text to embed.

        Returns:
            List of floats representing the embedding vector.
        """
        return self._embeddings.embed_query(text)

    async def aembed_query(self, text: str) -> list[float]:
        """
        Embed a single query text asynchronously.

        Args:
            text: The query text to embed.

        Returns:
            List of floats representing the embedding vector.
        """
        return await self._embeddings.aembed_query(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Embed multiple documents.

        Uses RETRIEVAL_DOCUMENT task type by default, optimized for document storage.

        Args:
            texts: List of document texts to embed.

        Returns:
            List of embedding vectors, one for each document.
        """
        return self._embeddings.embed_documents(texts)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Embed multiple documents asynchronously.

        Args:
            texts: List of document texts to embed.

        Returns:
            List of embedding vectors, one for each document.
        """
        return await self._embeddings.aembed_documents(texts)

    @classmethod
    def reset(cls) -> None:
        """
        Reset the singleton instance.

        Useful for testing or when you need to reinitialize with different parameters.
        After reset, the next instantiation will create a fresh client.
        """
        with cls._lock:
            cls._instance = None
            cls._initialized = False
            logger.debug("EmbeddingClient singleton reset")


# Convenience function
def get_embedding_client() -> EmbeddingClient:
    """
    Get the singleton EmbeddingClient instance.

    Returns:
        The singleton EmbeddingClient instance.
    """
    return EmbeddingClient()


def get_embeddings() -> Embeddings:
    """
    Get the embeddings instance for use with vector stores.

    Convenience function that returns the embeddings directly.

    Returns:
        The GoogleGenerativeAIEmbeddings instance.

    Example:
        from core.embedding_client import get_embeddings
        from core.pgvector_store import create_vector_store

        embeddings = get_embeddings()
        store = create_vector_store(embeddings)
    """
    return get_embedding_client().embeddings


if __name__ == "__main__":
    # Example usage
    client = get_embedding_client()
    query_embedding = client.embed_query("What is AI?")
    print(f"Query embedding (length {len(query_embedding)}): {query_embedding[:5]}...")
