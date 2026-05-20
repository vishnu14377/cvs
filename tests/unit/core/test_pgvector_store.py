"""Tests for PGVector Store factory."""

from unittest.mock import MagicMock, patch

import pytest
from langchain_postgres.vectorstores import DistanceStrategy

from src.core.pgvector_store import create_vector_store


@pytest.fixture
def mock_embeddings():
    """Create mock embeddings instance."""
    mock = MagicMock()
    mock.embed_query.return_value = [0.1] * 768
    mock.embed_documents.return_value = [[0.1] * 768]
    return mock


@pytest.fixture
def mock_engine():
    """Create mock SQLAlchemy engine."""
    return MagicMock()


@pytest.fixture
def mock_cloudsql_client(mock_engine):
    """Create mock CloudSQL client."""
    client = MagicMock()
    client.engine = mock_engine
    return client


# =============================================================================
# create_vector_store Tests
# =============================================================================


class TestCreateVectorStore:
    """Tests for create_vector_store function."""

    @patch("core.cloudsql_pg_client.get_cloudsql_client")
    @patch("core.pgvector_store.PGVector")
    def test_creates_pgvector_instance(
        self, mock_pgvector_class, mock_get_client, mock_embeddings, mock_cloudsql_client
    ):
        """Test that create_vector_store creates a PGVector instance."""
        mock_get_client.return_value = mock_cloudsql_client
        mock_pgvector_instance = MagicMock()
        mock_pgvector_class.return_value = mock_pgvector_instance

        result = create_vector_store(mock_embeddings)

        mock_pgvector_class.assert_called_once()
        assert result == mock_pgvector_instance

    @patch("core.cloudsql_pg_client.get_cloudsql_client")
    @patch("core.pgvector_store.PGVector")
    def test_uses_singleton_engine(
        self, mock_pgvector_class, mock_get_client, mock_embeddings, mock_cloudsql_client
    ):
        """Test that create_vector_store uses the singleton engine."""
        mock_get_client.return_value = mock_cloudsql_client
        mock_pgvector_class.return_value = MagicMock()

        create_vector_store(mock_embeddings)

        mock_get_client.assert_called_once()
        call_kwargs = mock_pgvector_class.call_args[1]
        assert call_kwargs["connection"] == mock_cloudsql_client.engine

    @patch("core.cloudsql_pg_client.get_cloudsql_client")
    @patch("core.pgvector_store.PGVector")
    def test_passes_embeddings(
        self, mock_pgvector_class, mock_get_client, mock_embeddings, mock_cloudsql_client
    ):
        """Test that create_vector_store passes embeddings."""
        mock_get_client.return_value = mock_cloudsql_client
        mock_pgvector_class.return_value = MagicMock()

        create_vector_store(mock_embeddings)

        call_kwargs = mock_pgvector_class.call_args[1]
        assert call_kwargs["embeddings"] == mock_embeddings

    @patch("core.cloudsql_pg_client.get_cloudsql_client")
    @patch("core.pgvector_store.PGVector")
    def test_custom_collection_name(
        self, mock_pgvector_class, mock_get_client, mock_embeddings, mock_cloudsql_client
    ):
        """Test that create_vector_store uses custom collection name."""
        mock_get_client.return_value = mock_cloudsql_client
        mock_pgvector_class.return_value = MagicMock()

        create_vector_store(mock_embeddings, collection_name="custom_collection")

        call_kwargs = mock_pgvector_class.call_args[1]
        assert call_kwargs["collection_name"] == "custom_collection"

    @patch("core.cloudsql_pg_client.get_cloudsql_client")
    @patch("core.pgvector_store.PGVector")
    @patch("core.pgvector_store.vectorstore_config")
    def test_default_collection_name_from_config(
        self,
        mock_config,
        mock_pgvector_class,
        mock_get_client,
        mock_embeddings,
        mock_cloudsql_client,
    ):
        """Test that create_vector_store uses default collection name from config."""
        mock_get_client.return_value = mock_cloudsql_client
        mock_pgvector_class.return_value = MagicMock()
        mock_config.COLLECTION_NAME = "config_collection"

        create_vector_store(mock_embeddings)

        call_kwargs = mock_pgvector_class.call_args[1]
        assert call_kwargs["collection_name"] == "config_collection"

    @patch("core.cloudsql_pg_client.get_cloudsql_client")
    @patch("core.pgvector_store.PGVector")
    def test_uses_cosine_distance_strategy(
        self, mock_pgvector_class, mock_get_client, mock_embeddings, mock_cloudsql_client
    ):
        """Test that create_vector_store always uses cosine distance strategy."""
        mock_get_client.return_value = mock_cloudsql_client
        mock_pgvector_class.return_value = MagicMock()

        create_vector_store(mock_embeddings)

        call_kwargs = mock_pgvector_class.call_args[1]
        assert call_kwargs["distance_strategy"] == DistanceStrategy.COSINE

    @patch("core.cloudsql_pg_client.get_cloudsql_client")
    @patch("core.pgvector_store.PGVector")
    def test_pre_delete_collection_false_by_default(
        self, mock_pgvector_class, mock_get_client, mock_embeddings, mock_cloudsql_client
    ):
        """Test that pre_delete_collection is False by default."""
        mock_get_client.return_value = mock_cloudsql_client
        mock_pgvector_class.return_value = MagicMock()

        create_vector_store(mock_embeddings)

        call_kwargs = mock_pgvector_class.call_args[1]
        assert call_kwargs["pre_delete_collection"] is False

    @patch("core.cloudsql_pg_client.get_cloudsql_client")
    @patch("core.pgvector_store.PGVector")
    def test_pre_delete_collection_true(
        self, mock_pgvector_class, mock_get_client, mock_embeddings, mock_cloudsql_client
    ):
        """Test create_vector_store with pre_delete_collection=True."""
        mock_get_client.return_value = mock_cloudsql_client
        mock_pgvector_class.return_value = MagicMock()

        create_vector_store(mock_embeddings, pre_delete_collection=True)

        call_kwargs = mock_pgvector_class.call_args[1]
        assert call_kwargs["pre_delete_collection"] is True

    @patch("core.cloudsql_pg_client.get_cloudsql_client")
    @patch("core.pgvector_store.PGVector")
    def test_use_jsonb_is_true(
        self, mock_pgvector_class, mock_get_client, mock_embeddings, mock_cloudsql_client
    ):
        """Test that use_jsonb is always True."""
        mock_get_client.return_value = mock_cloudsql_client
        mock_pgvector_class.return_value = MagicMock()

        create_vector_store(mock_embeddings)

        call_kwargs = mock_pgvector_class.call_args[1]
        assert call_kwargs["use_jsonb"] is True


# =============================================================================
# Integration-style Tests (with mocked dependencies)
# =============================================================================


class TestVectorStoreUsage:
    """Tests for typical vector store usage patterns."""

    @patch("core.cloudsql_pg_client.get_cloudsql_client")
    @patch("core.pgvector_store.PGVector")
    def test_add_texts_flow(
        self, mock_pgvector_class, mock_get_client, mock_embeddings, mock_cloudsql_client
    ):
        """Test adding texts to the vector store."""
        mock_get_client.return_value = mock_cloudsql_client
        mock_pgvector_instance = MagicMock()
        mock_pgvector_instance.add_texts.return_value = ["id1", "id2"]
        mock_pgvector_class.return_value = mock_pgvector_instance

        store = create_vector_store(mock_embeddings)
        ids = store.add_texts(["text1", "text2"])

        mock_pgvector_instance.add_texts.assert_called_once_with(["text1", "text2"])
        assert ids == ["id1", "id2"]

    @patch("core.cloudsql_pg_client.get_cloudsql_client")
    @patch("core.pgvector_store.PGVector")
    def test_similarity_search_flow(
        self, mock_pgvector_class, mock_get_client, mock_embeddings, mock_cloudsql_client
    ):
        """Test similarity search on the vector store."""
        mock_get_client.return_value = mock_cloudsql_client
        mock_pgvector_instance = MagicMock()
        mock_doc = MagicMock()
        mock_doc.page_content = "result"
        mock_pgvector_instance.similarity_search.return_value = [mock_doc]
        mock_pgvector_class.return_value = mock_pgvector_instance

        store = create_vector_store(mock_embeddings)
        results = store.similarity_search("query", k=5)

        mock_pgvector_instance.similarity_search.assert_called_once_with("query", k=5)
        assert len(results) == 1

    @patch("core.cloudsql_pg_client.get_cloudsql_client")
    @patch("core.pgvector_store.PGVector")
    def test_similarity_search_with_filter(
        self, mock_pgvector_class, mock_get_client, mock_embeddings, mock_cloudsql_client
    ):
        """Test similarity search with metadata filter."""
        mock_get_client.return_value = mock_cloudsql_client
        mock_pgvector_instance = MagicMock()
        mock_pgvector_instance.similarity_search.return_value = []
        mock_pgvector_class.return_value = mock_pgvector_instance

        store = create_vector_store(mock_embeddings)
        store.similarity_search("query", k=5, filter={"session_id": "user-123"})

        mock_pgvector_instance.similarity_search.assert_called_once_with(
            "query", k=5, filter={"session_id": "user-123"}
        )

    @patch("core.cloudsql_pg_client.get_cloudsql_client")
    @patch("core.pgvector_store.PGVector")
    def test_delete_flow(
        self, mock_pgvector_class, mock_get_client, mock_embeddings, mock_cloudsql_client
    ):
        """Test deleting documents from the vector store."""
        mock_get_client.return_value = mock_cloudsql_client
        mock_pgvector_instance = MagicMock()
        mock_pgvector_class.return_value = mock_pgvector_instance

        store = create_vector_store(mock_embeddings)
        store.delete(ids=["id1", "id2"])

        mock_pgvector_instance.delete.assert_called_once_with(ids=["id1", "id2"])

    @patch("core.cloudsql_pg_client.get_cloudsql_client")
    @patch("core.pgvector_store.PGVector")
    def test_as_retriever_flow(
        self, mock_pgvector_class, mock_get_client, mock_embeddings, mock_cloudsql_client
    ):
        """Test getting a retriever from the vector store."""
        mock_get_client.return_value = mock_cloudsql_client
        mock_pgvector_instance = MagicMock()
        mock_retriever = MagicMock()
        mock_pgvector_instance.as_retriever.return_value = mock_retriever
        mock_pgvector_class.return_value = mock_pgvector_instance

        store = create_vector_store(mock_embeddings)
        retriever = store.as_retriever(search_type="mmr", search_kwargs={"k": 5})

        mock_pgvector_instance.as_retriever.assert_called_once_with(
            search_type="mmr", search_kwargs={"k": 5}
        )
        assert retriever == mock_retriever


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling."""

    @patch("core.cloudsql_pg_client.get_cloudsql_client")
    def test_client_connection_error(self, mock_get_client, mock_embeddings):
        """Test handling of client connection errors."""
        mock_get_client.side_effect = Exception("Connection failed")

        with pytest.raises(Exception, match="Connection failed"):
            create_vector_store(mock_embeddings)

    @patch("core.cloudsql_pg_client.get_cloudsql_client")
    @patch("core.pgvector_store.PGVector")
    def test_pgvector_creation_error(
        self, mock_pgvector_class, mock_get_client, mock_embeddings, mock_cloudsql_client
    ):
        """Test handling of PGVector creation errors."""
        mock_get_client.return_value = mock_cloudsql_client
        mock_pgvector_class.side_effect = Exception("PGVector creation failed")

        with pytest.raises(Exception, match="PGVector creation failed"):
            create_vector_store(mock_embeddings)
