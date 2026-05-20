"""Tests for Embedding Client."""

from unittest.mock import MagicMock, patch

import pytest
from src.core.config import vectorstore_config
from src.core.embedding_client import (
    EmbeddingClient,
    get_embedding_client,
    get_embeddings,
)


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the singleton client before and after each test."""
    EmbeddingClient.reset()
    yield
    EmbeddingClient.reset()


@pytest.fixture
def mock_embeddings():
    """Create mock embeddings instance."""
    mock = MagicMock()
    mock.embed_query.return_value = [0.1] * 768
    mock.embed_documents.return_value = [[0.1] * 768, [0.2] * 768]
    return mock


# =============================================================================
# EmbeddingClient Initialization Tests
# =============================================================================


class TestEmbeddingClientInit:
    """Tests for EmbeddingClient initialization."""

    @patch("src.core.embedding_client.GoogleGenerativeAIEmbeddings")
    def test_initialization_creates_embeddings(self, mock_embeddings_class):
        """Test that initialization creates GoogleGenerativeAIEmbeddings."""
        mock_instance = MagicMock()
        mock_embeddings_class.return_value = mock_instance

        client = EmbeddingClient()

        mock_embeddings_class.assert_called_once()
        assert client._embeddings == mock_instance

    @patch("src.core.embedding_client.GoogleGenerativeAIEmbeddings")
    def test_initialization_uses_vertex_ai(self, mock_embeddings_class):
        """Test that initialization uses Vertex AI backend."""
        mock_instance = MagicMock()
        mock_embeddings_class.return_value = mock_instance

        EmbeddingClient()

        call_kwargs = mock_embeddings_class.call_args[1]
        assert call_kwargs["vertexai"] is True

    @patch("src.core.embedding_client.GoogleGenerativeAIEmbeddings")
    def test_initialization_uses_config_values(self, mock_embeddings_class):
        """Test that initialization uses values from config."""
        mock_instance = MagicMock()
        mock_embeddings_class.return_value = mock_instance

        with patch("src.core.embedding_client.vectorstore_config") as mock_config:
            mock_config.EMBEDDING_MODEL_ID = "test-model"
            mock_config.GCP_PROJECT = "test-project"
            mock_config.GCP_REGION = "us-east1"

            EmbeddingClient.reset()
            EmbeddingClient()

            call_kwargs = mock_embeddings_class.call_args[1]
            assert call_kwargs["model"] == "test-model"
            assert call_kwargs["project"] == "test-project"
            assert call_kwargs["location"] == "us-east1"


# =============================================================================
# Singleton Pattern Tests
# =============================================================================


class TestSingletonPattern:
    """Tests for singleton pattern."""

    @patch("src.core.embedding_client.GoogleGenerativeAIEmbeddings")
    def test_returns_same_instance(self, mock_embeddings_class):
        """Test that multiple calls return the same instance."""
        mock_embeddings_class.return_value = MagicMock()

        client1 = EmbeddingClient()
        client2 = EmbeddingClient()

        assert client1 is client2

    @patch("src.core.embedding_client.GoogleGenerativeAIEmbeddings")
    def test_only_initializes_once(self, mock_embeddings_class):
        """Test that embeddings are only created once."""
        mock_embeddings_class.return_value = MagicMock()

        EmbeddingClient()
        EmbeddingClient()
        EmbeddingClient()

        mock_embeddings_class.assert_called_once()

    @patch("src.core.embedding_client.GoogleGenerativeAIEmbeddings")
    def test_reset_clears_singleton(self, mock_embeddings_class):
        """Test that reset clears the singleton."""
        mock_embeddings_class.return_value = MagicMock()

        client1 = EmbeddingClient()
        EmbeddingClient.reset()
        client2 = EmbeddingClient()

        assert client1 is not client2

    @patch("src.core.embedding_client.GoogleGenerativeAIEmbeddings")
    def test_reset_allows_reinitialization(self, mock_embeddings_class):
        """Test that reset allows new initialization."""
        mock_embeddings_class.return_value = MagicMock()

        EmbeddingClient()
        EmbeddingClient.reset()
        EmbeddingClient()

        assert mock_embeddings_class.call_count == 2


# =============================================================================
# Property Tests
# =============================================================================


class TestProperties:
    """Tests for EmbeddingClient properties."""

    @patch("src.core.embedding_client.GoogleGenerativeAIEmbeddings")
    def test_embeddings_property(self, mock_embeddings_class):
        """Test embeddings property returns the embeddings instance."""
        mock_instance = MagicMock()
        mock_embeddings_class.return_value = mock_instance

        client = EmbeddingClient()

        assert client.embeddings == mock_instance

    @patch("src.core.embedding_client.GoogleGenerativeAIEmbeddings")
    def test_model_id_property(self, mock_embeddings_class):
        """Test model_id property."""
        mock_embeddings_class.return_value = MagicMock()

        with patch("src.core.embedding_client.vectorstore_config") as mock_config:
            mock_config.EMBEDDING_MODEL_ID = "test-model"
            mock_config.GCP_PROJECT = "test-project"
            mock_config.GCP_REGION = "us-central1"

            EmbeddingClient.reset()
            client = EmbeddingClient()

            assert client.model_id == "test-model"

    @patch("src.core.embedding_client.GoogleGenerativeAIEmbeddings")
    def test_project_id_property(self, mock_embeddings_class):
        """Test project_id property."""
        mock_embeddings_class.return_value = MagicMock()

        with patch("src.core.embedding_client.vectorstore_config") as mock_config:
            mock_config.EMBEDDING_MODEL_ID = "test-model"
            mock_config.GCP_PROJECT = "test-project"
            mock_config.GCP_REGION = "us-central1"

            EmbeddingClient.reset()
            client = EmbeddingClient()

            assert client.project_id == "test-project"

    @patch("src.core.embedding_client.GoogleGenerativeAIEmbeddings")
    def test_location_property(self, mock_embeddings_class):
        """Test location property."""
        mock_embeddings_class.return_value = MagicMock()

        with patch("src.core.embedding_client.vectorstore_config") as mock_config:
            mock_config.EMBEDDING_MODEL_ID = "test-model"
            mock_config.GCP_PROJECT = "test-project"
            mock_config.GCP_REGION = "us-east1"

            EmbeddingClient.reset()
            client = EmbeddingClient()

            assert client.location == "us-east1"


# =============================================================================
# Embedding Methods Tests
# =============================================================================


class TestEmbeddingMethods:
    """Tests for embedding methods."""

    @patch("src.core.embedding_client.GoogleGenerativeAIEmbeddings")
    def test_embed_query(self, mock_embeddings_class):
        """Test embed_query method."""
        mock_instance = MagicMock()
        mock_instance.embed_query.return_value = [0.1, 0.2, 0.3]
        mock_embeddings_class.return_value = mock_instance

        client = EmbeddingClient()
        result = client.embed_query("test query")

        mock_instance.embed_query.assert_called_once_with("test query")
        assert result == [0.1, 0.2, 0.3]

    @patch("src.core.embedding_client.GoogleGenerativeAIEmbeddings")
    def test_embed_documents(self, mock_embeddings_class):
        """Test embed_documents method."""
        mock_instance = MagicMock()
        mock_instance.embed_documents.return_value = [[0.1], [0.2]]
        mock_embeddings_class.return_value = mock_instance

        client = EmbeddingClient()
        result = client.embed_documents(["doc1", "doc2"])

        mock_instance.embed_documents.assert_called_once_with(["doc1", "doc2"])
        assert result == [[0.1], [0.2]]

    @patch("src.core.embedding_client.GoogleGenerativeAIEmbeddings")
    @pytest.mark.asyncio
    async def test_aembed_query(self, mock_embeddings_class):
        """Test async embed_query method."""
        mock_instance = MagicMock()
        mock_instance.aembed_query = MagicMock(return_value=[0.1, 0.2, 0.3])
        mock_embeddings_class.return_value = mock_instance

        # Make it async
        async def async_return():
            return [0.1, 0.2, 0.3]

        mock_instance.aembed_query = MagicMock(return_value=async_return())

        client = EmbeddingClient()
        result = await client.aembed_query("test query")

        assert result == [0.1, 0.2, 0.3]

    @patch("src.core.embedding_client.GoogleGenerativeAIEmbeddings")
    @pytest.mark.asyncio
    async def test_aembed_documents(self, mock_embeddings_class):
        """Test async embed_documents method."""
        mock_instance = MagicMock()

        async def async_return():
            return [[0.1], [0.2]]

        mock_instance.aembed_documents = MagicMock(return_value=async_return())
        mock_embeddings_class.return_value = mock_instance

        client = EmbeddingClient()
        result = await client.aembed_documents(["doc1", "doc2"])

        assert result == [[0.1], [0.2]]


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    @patch("src.core.embedding_client.GoogleGenerativeAIEmbeddings")
    def test_get_embedding_client(self, mock_embeddings_class):
        """Test get_embedding_client returns singleton."""
        mock_embeddings_class.return_value = MagicMock()

        client1 = get_embedding_client()
        client2 = get_embedding_client()

        assert client1 is client2
        assert isinstance(client1, EmbeddingClient)

    @patch("src.core.embedding_client.GoogleGenerativeAIEmbeddings")
    def test_get_embeddings(self, mock_embeddings_class):
        """Test get_embeddings returns embeddings instance."""
        mock_instance = MagicMock()
        mock_embeddings_class.return_value = mock_instance

        embeddings = get_embeddings()

        assert embeddings == mock_instance


# =============================================================================
# Thread Safety Tests
# =============================================================================


class TestThreadSafety:
    """Tests for thread safety."""

    @patch("src.core.embedding_client.GoogleGenerativeAIEmbeddings")
    def test_concurrent_access(self, mock_embeddings_class):
        """Test that concurrent access returns same instance."""
        import threading

        mock_embeddings_class.return_value = MagicMock()

        clients = []

        def get_client():
            client = EmbeddingClient()
            clients.append(client)

        threads = [threading.Thread(target=get_client) for _ in range(10)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All clients should be the same instance
        assert all(c is clients[0] for c in clients)

    @patch("src.core.embedding_client.GoogleGenerativeAIEmbeddings")
    def test_concurrent_initialization(self, mock_embeddings_class):
        """Test that concurrent access only creates one embeddings instance."""
        import threading

        mock_embeddings_class.return_value = MagicMock()

        def get_client():
            EmbeddingClient()

        threads = [threading.Thread(target=get_client) for _ in range(10)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should only create embeddings once
        mock_embeddings_class.assert_called_once()


# =============================================================================
# Stub Mode Gate Tests
# =============================================================================


class TestEmbeddingClientStubMode:
    def setup_method(self) -> None:
        from src.core.embedding_client import EmbeddingClient

        EmbeddingClient.reset()

    def teardown_method(self) -> None:
        from src.core.embedding_client import EmbeddingClient

        EmbeddingClient.reset()

    def test_stub_mode_uses_stub_embeddings(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from src.core.stubs.embedding_stub import StubEmbeddings

        monkeypatch.setattr("src.core.embedding_client.vectorstore_config.VERTEX_AI_MODE", "stub")
        from src.core.embedding_client import EmbeddingClient

        EmbeddingClient.reset()
        client = EmbeddingClient()
        assert isinstance(client.embeddings, StubEmbeddings)
        assert len(client.embed_query("x")) == vectorstore_config.EMBEDDING_DIMENSION

    def test_real_mode_uses_google_generative_ai_embeddings(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("src.core.embedding_client.vectorstore_config.VERTEX_AI_MODE", "real")
        with patch("src.core.embedding_client.GoogleGenerativeAIEmbeddings") as mock_emb:
            mock_emb.return_value = MagicMock()
            from src.core.embedding_client import EmbeddingClient

            EmbeddingClient.reset()
            _ = EmbeddingClient()
            mock_emb.assert_called_once()
            call_kwargs = mock_emb.call_args.kwargs
            assert "model" in call_kwargs
            assert "project" in call_kwargs
            assert "location" in call_kwargs
            assert call_kwargs.get("vertexai") is True
