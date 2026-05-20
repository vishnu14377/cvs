"""
PGVector Store factory for RAG vector operations.

This module provides a simple factory function to create LangChain's PGVector
instance using the singleton CloudSQL engine. Use LangChain's PGVector class
directly for all vector operations.

Usage:
    from src.core.pgvector_store import create_vector_store
    from src.core.embedding_client import get_embeddings

    # Create vector store (uses singleton engine)
    vector_store = create_vector_store(get_embeddings(), collection_name="my_docs")

    # Use LangChain's PGVector API directly
    vector_store.add_texts(["doc1", "doc2"], metadatas=[{"key": "val1"}, {"key": "val2"}])
    results = vector_store.similarity_search("query", k=5)
    vector_store.delete(ids=["id1", "id2"])
"""

from __future__ import annotations

from langchain_core.embeddings import Embeddings
from langchain_postgres import PGVector
from langchain_postgres.vectorstores import DistanceStrategy
from src.core.config import vectorstore_config
from src.core.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# Factory Function
# =============================================================================


def create_vector_store(
    embeddings: Embeddings,
    collection_name: str | None = None,
    pre_delete_collection: bool = False,
) -> PGVector:
    """
    Create a LangChain PGVector instance using the singleton CloudSQL engine.

    Uses cosine distance strategy, which is the standard choice for text
    embeddings and works well with normalized vectors from embedding models.

    Args:
        embeddings: LangChain Embeddings instance.
        collection_name: Name of the collection (default from config).
        pre_delete_collection: Delete existing collection on init.

    Returns:
        LangChain PGVector instance.

    Example:
        from src.core.pgvector_store import create_vector_store
        from src.core.embedding_client import get_embeddings

        # Create vector store
        store = create_vector_store(get_embeddings(), collection_name="my_docs")

        # Add documents
        ids = store.add_texts(
            texts=["AI is transforming healthcare"],
            metadatas=[{"source": "article.pdf", "session_id": "user-123"}]
        )

        # Search with metadata filter
        results = store.similarity_search(
            "healthcare AI",
            k=5,
            filter={"session_id": "user-123"}
        )

        # Delete by IDs
        store.delete(ids=ids)

        # Get as retriever for LangChain chains
        retriever = store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 5, "filter": {"session_id": "user-123"}}
        )
    """
    from src.core.cloudsql_pg_client import get_cloudsql_client

    # Get singleton engine
    client = get_cloudsql_client()
    engine = client.engine

    # Resolve collection name
    coll_name = collection_name or vectorstore_config.COLLECTION_NAME

    # Create and return PGVector instance
    vector_store = PGVector(
        embeddings=embeddings,
        collection_name=coll_name,
        connection=engine,
        distance_strategy=DistanceStrategy.COSINE,
        pre_delete_collection=pre_delete_collection,
        use_jsonb=True,
    )

    logger.info(f"PGVector created: collection='{coll_name}'")
    return vector_store


if __name__ == "__main__":
    # Example usage
    from src.core.embedding_client import get_embedding_client

    embeddings = get_embedding_client().embeddings
    store = create_vector_store(
        embeddings, collection_name="test_collection", pre_delete_collection=True
    )
    print("PGVector store created successfully.")
