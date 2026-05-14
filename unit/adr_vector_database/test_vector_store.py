"""Tests for ADR Vector Database vector store manager.

The VectorStoreManager only handles:
- batch_insert: Insert documents in batches
- insert: Insert documents (single batch)
- delete_by_ids: Delete documents by IDs
- cleanup_collection: Delete all documents in a collection

All search operations should go through SessionRetriever.
"""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document
from src.adr_vector_database.vector_store import (
    VectorStoreManager,
    batch_insert_documents,
    cleanup_collection,
    get_vector_store_manager,
    insert_documents,
)
from src.core.config import vectorstore_config

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_documents():
    """Sample documents for testing."""
    return [
        Document(page_content=f"Content {i}", metadata={"session_id": "sess-123", "page": i})
        for i in range(5)
    ]


@pytest.fixture
def mock_vector_store():
    """Create a mock vector store."""
    mock = MagicMock()
    mock.add_documents.return_value = ["id-1", "id-2", "id-3", "id-4", "id-5"]
    return mock


# =============================================================================
# VectorStoreManager Initialization Tests
# =============================================================================


class TestVectorStoreManagerInit:
    """Tests for VectorStoreManager initialization."""

    @patch("src.adr_vector_database.vector_store.create_vector_store")
    @patch("src.adr_vector_database.vector_store.get_embedding_client")
    def test_default_initialization(self, mock_get_embeddings, mock_create_store):
        """Test default initialization."""
        manager = VectorStoreManager()
        assert manager.batch_size == vectorstore_config.DEFAULT_BATCH_SIZE

    @patch("src.adr_vector_database.vector_store.create_vector_store")
    @patch("src.adr_vector_database.vector_store.get_embedding_client")
    def test_custom_parameters(self, mock_get_embeddings, mock_create_store):
        """Test custom parameter initialization."""
        manager = VectorStoreManager(collection_name="custom", batch_size=50)
        assert manager.collection_name == "custom"
        assert manager.batch_size == 50

    @patch("src.adr_vector_database.vector_store.create_vector_store")
    @patch("src.adr_vector_database.vector_store.get_embedding_client")
    def test_lazy_vector_store_initialization(self, mock_get_embeddings, mock_create_store):
        """Test that vector store is lazily initialized."""
        manager = VectorStoreManager()
        # Vector store should not be created yet
        mock_create_store.assert_not_called()

        # Access vector store to trigger initialization
        _ = manager.vector_store
        mock_create_store.assert_called_once()


# =============================================================================
# Insert Operation Tests
# =============================================================================


class TestInsertOperations:
    """Tests for insert operations."""

    @patch("src.adr_vector_database.vector_store.create_vector_store")
    @patch("src.adr_vector_database.vector_store.get_embedding_client")
    def test_insert_documents(
        self, mock_get_embeddings, mock_create_store, mock_vector_store, sample_documents
    ):
        """Test inserting documents."""
        mock_get_embeddings.return_value.embeddings = MagicMock()
        mock_create_store.return_value = mock_vector_store

        manager = VectorStoreManager()
        ids = manager.insert(sample_documents)

        mock_vector_store.add_documents.assert_called_once_with(sample_documents)
        assert len(ids) == 5

    @patch("src.adr_vector_database.vector_store.create_vector_store")
    @patch("src.adr_vector_database.vector_store.get_embedding_client")
    def test_insert_empty_list(self, mock_get_embeddings, mock_create_store):
        """Test inserting empty list returns empty list."""
        manager = VectorStoreManager()
        ids = manager.insert([])

        assert ids == []
        mock_create_store.assert_not_called()

    @patch("src.adr_vector_database.vector_store.create_vector_store")
    @patch("src.adr_vector_database.vector_store.get_embedding_client")
    def test_batch_insert_single_batch(
        self, mock_get_embeddings, mock_create_store, mock_vector_store, sample_documents
    ):
        """Test batch insert with documents fitting in a single batch."""
        mock_get_embeddings.return_value.embeddings = MagicMock()
        mock_create_store.return_value = mock_vector_store

        manager = VectorStoreManager(batch_size=10)
        ids = manager.batch_insert(sample_documents)

        # Should be single batch (5 docs < 10 batch size)
        assert mock_vector_store.add_documents.call_count == 1
        assert len(ids) == 5

    @patch("src.adr_vector_database.vector_store.create_vector_store")
    @patch("src.adr_vector_database.vector_store.get_embedding_client")
    def test_batch_insert_multiple_batches(
        self, mock_get_embeddings, mock_create_store, mock_vector_store
    ):
        """Test batch insert splits into multiple batches."""
        mock_get_embeddings.return_value.embeddings = MagicMock()
        mock_create_store.return_value = mock_vector_store
        mock_vector_store.add_documents.side_effect = [
            ["id-1", "id-2"],
            ["id-3", "id-4"],
            ["id-5"],
        ]

        docs = [Document(page_content=f"Content {i}") for i in range(5)]
        manager = VectorStoreManager(batch_size=2)
        ids = manager.batch_insert(docs)

        # Should be 3 batches (5 docs / 2 batch size = ceil(2.5) = 3)
        assert mock_vector_store.add_documents.call_count == 3
        assert len(ids) == 5

    @patch("src.adr_vector_database.vector_store.create_vector_store")
    @patch("src.adr_vector_database.vector_store.get_embedding_client")
    def test_batch_insert_empty_list(self, mock_get_embeddings, mock_create_store):
        """Test batch insert with empty list."""
        manager = VectorStoreManager()
        ids = manager.batch_insert([])

        assert ids == []
        mock_create_store.assert_not_called()

    @patch("src.adr_vector_database.vector_store.create_vector_store")
    @patch("src.adr_vector_database.vector_store.get_embedding_client")
    def test_batch_insert_override_batch_size(
        self, mock_get_embeddings, mock_create_store, mock_vector_store
    ):
        """Test batch insert with overridden batch size."""
        mock_get_embeddings.return_value.embeddings = MagicMock()
        mock_create_store.return_value = mock_vector_store
        mock_vector_store.add_documents.return_value = ["id-1", "id-2", "id-3"]

        docs = [Document(page_content=f"Content {i}") for i in range(3)]
        manager = VectorStoreManager(batch_size=1)  # Default batch size 1
        manager.batch_insert(docs, batch_size=10)  # Override to 10

        # Should be single batch (using overridden batch size)
        assert mock_vector_store.add_documents.call_count == 1


# =============================================================================
# Delete Operation Tests
# =============================================================================


class TestDeleteOperations:
    """Tests for delete operations."""

    @patch("src.adr_vector_database.vector_store.create_vector_store")
    @patch("src.adr_vector_database.vector_store.get_embedding_client")
    def test_delete_by_ids(self, mock_get_embeddings, mock_create_store, mock_vector_store):
        """Test deleting documents by IDs."""
        mock_get_embeddings.return_value.embeddings = MagicMock()
        mock_create_store.return_value = mock_vector_store

        manager = VectorStoreManager()
        manager.delete_by_ids(["id-1", "id-2", "id-3"])

        mock_vector_store.delete.assert_called_once_with(ids=["id-1", "id-2", "id-3"])

    @patch("src.adr_vector_database.vector_store.create_vector_store")
    @patch("src.adr_vector_database.vector_store.get_embedding_client")
    def test_delete_empty_list(self, mock_get_embeddings, mock_create_store):
        """Test deleting with empty list does nothing."""
        manager = VectorStoreManager()
        manager.delete_by_ids([])

        mock_create_store.assert_not_called()


# =============================================================================
# Cleanup Operation Tests
# =============================================================================


class TestCleanupOperations:
    """Tests for cleanup operations."""

    @patch("src.adr_vector_database.vector_store.create_vector_store")
    @patch("src.adr_vector_database.vector_store.get_embedding_client")
    def test_cleanup_collection(self, mock_get_embeddings, mock_create_store):
        """Test cleaning up a collection."""
        mock_get_embeddings.return_value.embeddings = MagicMock()
        mock_create_store.return_value = MagicMock()

        manager = VectorStoreManager(collection_name="test-collection")
        result = manager.cleanup_collection()

        assert result is True
        mock_create_store.assert_called_once()
        # Should use pre_delete_collection=True
        call_kwargs = mock_create_store.call_args.kwargs
        assert call_kwargs.get("pre_delete_collection") is True

    @patch("src.adr_vector_database.vector_store.create_vector_store")
    @patch("src.adr_vector_database.vector_store.get_embedding_client")
    def test_cleanup_collection_failure(self, mock_get_embeddings, mock_create_store):
        """Test cleanup failure returns False."""
        mock_get_embeddings.return_value.embeddings = MagicMock()
        mock_create_store.side_effect = Exception("Database error")

        manager = VectorStoreManager()
        result = manager.cleanup_collection()

        assert result is False


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    @patch("src.adr_vector_database.vector_store.create_vector_store")
    @patch("src.adr_vector_database.vector_store.get_embedding_client")
    def test_batch_insert_documents_function(
        self, mock_get_embeddings, mock_create_store, mock_vector_store, sample_documents
    ):
        """Test batch_insert_documents convenience function."""
        mock_get_embeddings.return_value.embeddings = MagicMock()
        mock_create_store.return_value = mock_vector_store

        ids = batch_insert_documents(sample_documents, collection_name="test")

        assert len(ids) == 5

    @patch("src.adr_vector_database.vector_store.create_vector_store")
    @patch("src.adr_vector_database.vector_store.get_embedding_client")
    def test_insert_documents_function(
        self, mock_get_embeddings, mock_create_store, mock_vector_store, sample_documents
    ):
        """Test insert_documents convenience function."""
        mock_get_embeddings.return_value.embeddings = MagicMock()
        mock_create_store.return_value = mock_vector_store

        ids = insert_documents(sample_documents, collection_name="test")

        assert len(ids) == 5

    @patch("src.adr_vector_database.vector_store.create_vector_store")
    @patch("src.adr_vector_database.vector_store.get_embedding_client")
    def test_cleanup_collection_function(self, mock_get_embeddings, mock_create_store):
        """Test cleanup_collection convenience function."""
        mock_get_embeddings.return_value.embeddings = MagicMock()
        mock_create_store.return_value = MagicMock()

        result = cleanup_collection("test-collection")

        assert result is True

    def test_get_vector_store_manager_function(self):
        """Test get_vector_store_manager factory function."""
        manager = get_vector_store_manager(collection_name="test", batch_size=50)

        assert isinstance(manager, VectorStoreManager)
        assert manager.collection_name == "test"
        assert manager.batch_size == 50
