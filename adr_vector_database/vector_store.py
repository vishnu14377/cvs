"""
Vector Store module for ADR Vector Database.

Provides functions for managing documents in the PGVector database:
- Batch insertion of documents
- Deletion by IDs
- Collection cleanup

All search operations should be done through the SessionRetriever.

Usage:
    from adr_vector_database.vector_store import VectorStoreManager

    manager = VectorStoreManager(collection_name="adr_documents")
    ids = manager.batch_insert(documents)
    manager.delete_by_ids(ids)
    manager.cleanup_collection()
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from src.core.config import vectorstore_config
from src.core.embedding_client import get_embedding_client
from src.core.logger import get_logger
from src.core.pgvector_store import create_vector_store

logger = get_logger(__name__)


class VectorStoreManager:
    """
    Manages vector store operations for ADR documents.

    Handles document insertion, deletion, and collection management.
    For search operations, use SessionRetriever instead.

    Example:
        >>> manager = VectorStoreManager(collection_name="session-123")
        >>> ids = manager.batch_insert(documents)
        >>> manager.cleanup_collection()
    """

    def __init__(
        self,
        collection_name: str | None = None,
        batch_size: int | None = None,
    ):
        """
        Initialize the vector store manager.

        Args:
            collection_name: PGVector collection name. Defaults to config value.
            batch_size: Default batch size for insertions. Defaults to config value.
        """
        self.collection_name = collection_name or vectorstore_config.COLLECTION_NAME
        self.batch_size = batch_size or vectorstore_config.DEFAULT_BATCH_SIZE
        self._vector_store: VectorStore | None = None

        logger.debug(
            "VectorStoreManager initialized: collection='%s', batch_size=%d",
            self.collection_name,
            self.batch_size,
        )

    def _get_vector_store(self) -> VectorStore:
        """Get or create the vector store."""
        if self._vector_store is None:
            embeddings = get_embedding_client().embeddings
            self._vector_store = create_vector_store(
                embeddings=embeddings,
                collection_name=self.collection_name,
            )
            logger.debug("Vector store initialized for collection '%s'", self.collection_name)
        return self._vector_store

    @property
    def vector_store(self) -> VectorStore:
        """Get the underlying vector store instance."""
        return self._get_vector_store()

    def batch_insert(
        self,
        documents: list[Document],
        batch_size: int | None = None,
    ) -> list[str]:
        """
        Insert documents into the vector store in batches.

        Args:
            documents: List of LangChain Documents to insert.
            batch_size: Number of documents per batch.

        Returns:
            List of vector IDs for all inserted documents.
        """
        if not documents:
            logger.warning("No documents provided for batch insert")
            return []

        batch_size = batch_size or self.batch_size
        vector_store = self._get_vector_store()
        all_ids: list[str] = []

        total_docs = len(documents)
        num_batches = (total_docs + batch_size - 1) // batch_size

        logger.info(
            "Batch insert: %d documents in %d batches",
            total_docs,
            num_batches,
        )

        for i in range(0, total_docs, batch_size):
            batch = documents[i : i + batch_size]
            batch_num = (i // batch_size) + 1

            try:
                ids = vector_store.add_documents(batch)
                all_ids.extend(ids)
                logger.debug("Batch %d/%d: inserted %d documents", batch_num, num_batches, len(ids))
            except Exception as e:
                logger.error("Batch %d/%d failed: %s", batch_num, num_batches, e)
                raise

        logger.info("Batch insert complete: %d documents inserted", len(all_ids))
        return all_ids

    def insert(self, documents: list[Document]) -> list[str]:
        """
        Insert documents into the vector store (single batch).

        Args:
            documents: List of LangChain Documents to insert.

        Returns:
            List of vector IDs for the inserted documents.
        """
        if not documents:
            logger.warning("No documents provided for insert")
            return []

        vector_store = self._get_vector_store()
        ids = vector_store.add_documents(documents)
        logger.info("Inserted %d documents into vector store", len(ids))
        return ids

    def delete_by_ids(self, ids: list[str]) -> None:
        """
        Delete documents by their vector IDs.

        Args:
            ids: List of vector IDs to delete.
        """
        if not ids:
            return

        vector_store = self._get_vector_store()
        vector_store.delete(ids=ids)
        logger.info("Deleted %d documents from vector store", len(ids))

    def delete_session(self, session_id: str) -> int:
        """
        Delete all documents belonging to a specific session.

        Uses a direct SQL query to find and remove only the rows whose
        ``session_id`` metadata matches the given value, leaving all other
        documents in the collection untouched.

        Args:
            session_id: The session whose documents should be removed.

        Returns:
            Number of documents deleted.
        """
        from sqlalchemy import text
        from src.core.cloudsql_pg_client import get_cloudsql_client

        engine = get_cloudsql_client().engine
        assert engine is not None

        query = text("""
            DELETE FROM langchain_pg_embedding
            WHERE collection_id = (
                SELECT uuid FROM langchain_pg_collection WHERE name = :collection_name
            )
            AND cmetadata->>'session_id' = :session_id
        """)

        try:
            with engine.connect() as conn:
                result = conn.execute(
                    query,
                    {
                        "collection_name": self.collection_name,
                        "session_id": session_id,
                    },
                )
                conn.commit()
                deleted = result.rowcount
            logger.info(
                "Deleted %d documents for session '%s' from collection '%s'",
                deleted,
                session_id,
                self.collection_name,
            )
            return deleted
        except Exception as e:
            logger.error(
                "Failed to delete session '%s' from collection '%s': %s",
                session_id,
                self.collection_name,
                e,
            )
            raise

    def cleanup_collection(self) -> bool:
        """
        Delete all documents in the collection.

        Returns:
            True if cleanup was successful, False otherwise.
        """
        try:
            embeddings = get_embedding_client().embeddings
            self._vector_store = create_vector_store(
                embeddings=embeddings,
                collection_name=self.collection_name,
                pre_delete_collection=True,
            )
            logger.info("Cleaned up collection '%s'", self.collection_name)
            return True
        except Exception as e:
            logger.error("Failed to cleanup collection '%s': %s", self.collection_name, e)
            return False


# =============================================================================
# Convenience Functions
# =============================================================================


def batch_insert_documents(
    documents: list[Document],
    collection_name: str | None = None,
    batch_size: int | None = None,
) -> list[str]:
    """Batch insert documents into the vector store."""
    manager = VectorStoreManager(collection_name=collection_name, batch_size=batch_size)
    return manager.batch_insert(documents)


def insert_documents(
    documents: list[Document],
    collection_name: str | None = None,
) -> list[str]:
    """Insert documents into the vector store."""
    manager = VectorStoreManager(collection_name=collection_name)
    return manager.insert(documents)


def cleanup_collection(collection_name: str) -> bool:
    """Delete all documents in a collection."""
    manager = VectorStoreManager(collection_name=collection_name)
    return manager.cleanup_collection()


def delete_session_documents(session_id: str, collection_name: str | None = None) -> int:
    """Delete all documents belonging to a session.

    Args:
        session_id: The session whose documents should be removed.
        collection_name: PGVector collection name. Defaults to config value.

    Returns:
        Number of documents deleted.
    """
    manager = VectorStoreManager(collection_name=collection_name)
    return manager.delete_session(session_id)


def get_vector_store_manager(
    collection_name: str | None = None,
    batch_size: int | None = None,
) -> VectorStoreManager:
    """Factory function to create a VectorStoreManager."""
    return VectorStoreManager(collection_name=collection_name, batch_size=batch_size)
