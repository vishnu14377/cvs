"""Tests for ADR Vector Database session retriever."""

from unittest.mock import MagicMock, create_autospec, patch

import pytest
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from src.adr_vector_database.retriever import (
    HybridRetrieverManager,
    SessionRetriever,
    VectorStoreSingleton,
    get_hybrid_retriever,
    get_hybrid_retriever_manager,
    get_session_documents,
    get_session_retriever,
    get_vector_store_singleton,
)


@pytest.fixture
def sample_documents():
    """Sample documents for testing."""
    return [
        Document(page_content="Diagnosis: Type 2 Diabetes", metadata={"session_id": "sess-123"}),
        Document(page_content="Treatment: Metformin 500mg", metadata={"session_id": "sess-123"}),
        Document(page_content="Follow-up in 3 months", metadata={"session_id": "sess-123"}),
    ]


@pytest.fixture
def mock_vector_store():
    """Create a mock vector store."""
    mock = MagicMock()
    mock_retriever = MagicMock(spec=BaseRetriever)
    mock.as_retriever.return_value = mock_retriever
    return mock


@pytest.fixture
def mock_base_retriever():
    """Create a mock retriever that passes Pydantic validation."""
    mock = create_autospec(BaseRetriever, instance=True)
    mock.invoke.return_value = []
    return mock


@pytest.fixture
def reset_singletons():
    """Reset singletons before and after each test."""
    VectorStoreSingleton._instance = None
    HybridRetrieverManager._instance = None
    yield
    VectorStoreSingleton._instance = None
    HybridRetrieverManager._instance = None


class TestVectorStoreSingleton:
    """Tests for VectorStoreSingleton."""

    def test_singleton_pattern(self, reset_singletons):
        """Test that only one instance is created."""
        singleton1 = get_vector_store_singleton()
        singleton2 = get_vector_store_singleton()
        assert singleton1 is singleton2

    def test_direct_instantiation_singleton(self, reset_singletons):
        """Test singleton via direct instantiation."""
        singleton1 = VectorStoreSingleton()
        singleton2 = VectorStoreSingleton()
        assert singleton1 is singleton2

    def test_reset(self, reset_singletons):
        """Test reset clears all cached vector stores."""
        singleton = get_vector_store_singleton()
        singleton._vector_stores = {"adr_documents": MagicMock(), "policy_documents": MagicMock()}

        singleton.reset()

        assert singleton._vector_stores == {}

    def test_caches_separate_stores_per_collection(self, reset_singletons):
        """Different collections must not share a cached VectorStore."""
        singleton = get_vector_store_singleton()
        with (
            patch("src.adr_vector_database.retriever.create_vector_store") as mock_create,
            patch("src.adr_vector_database.retriever.get_embedding_client"),
        ):
            mock_create.side_effect = lambda embeddings, collection_name: MagicMock(
                name=collection_name
            )
            store_adr = singleton.get_vector_store(collection_name="adr_documents")
            store_policy = singleton.get_vector_store(collection_name="policy_documents")
            # Second call for adr_documents should reuse the cached one.
            store_adr_again = singleton.get_vector_store(collection_name="adr_documents")

        assert store_adr is not store_policy
        assert store_adr is store_adr_again
        assert mock_create.call_count == 2


class TestGetSessionRetriever:
    """Tests for get_session_retriever factory function."""

    @patch("src.adr_vector_database.retriever.get_vector_store_singleton")
    def test_creates_retriever(self, mock_singleton, reset_singletons, mock_vector_store):
        """Test that factory creates a valid retriever."""
        mock_singleton.return_value.get_vector_store.return_value = mock_vector_store

        retriever = get_session_retriever(session_id="sess-123")

        assert retriever is not None
        mock_vector_store.as_retriever.assert_called_once()

    @patch("src.adr_vector_database.retriever.get_vector_store_singleton")
    def test_applies_session_filter(self, mock_singleton, reset_singletons, mock_vector_store):
        """Test that session filter is applied."""
        mock_singleton.return_value.get_vector_store.return_value = mock_vector_store

        get_session_retriever(session_id="my-session")

        call_kwargs = mock_vector_store.as_retriever.call_args.kwargs
        assert call_kwargs["search_kwargs"]["filter"] == {"session_id": "my-session"}

    @patch("src.adr_vector_database.retriever.get_vector_store_singleton")
    def test_similarity_search_type(self, mock_singleton, reset_singletons, mock_vector_store):
        """Test similarity search type configuration."""
        mock_singleton.return_value.get_vector_store.return_value = mock_vector_store

        get_session_retriever(session_id="sess-123", search_type="similarity", k=5)

        call_kwargs = mock_vector_store.as_retriever.call_args.kwargs
        assert call_kwargs["search_type"] == "similarity"
        assert call_kwargs["search_kwargs"]["k"] == 5

    @patch("src.adr_vector_database.retriever.get_vector_store_singleton")
    def test_mmr_search_type(self, mock_singleton, reset_singletons, mock_vector_store):
        """Test MMR search type configuration."""
        mock_singleton.return_value.get_vector_store.return_value = mock_vector_store

        get_session_retriever(
            session_id="sess-123",
            search_type="mmr",
            k=5,
            fetch_k=20,
            lambda_mult=0.5,
        )

        call_kwargs = mock_vector_store.as_retriever.call_args.kwargs
        assert call_kwargs["search_type"] == "mmr"
        assert call_kwargs["search_kwargs"]["fetch_k"] == 20
        assert call_kwargs["search_kwargs"]["lambda_mult"] == 0.5

    @patch("src.adr_vector_database.retriever.get_vector_store_singleton")
    def test_threshold_search_type(self, mock_singleton, reset_singletons, mock_vector_store):
        """Test threshold search type configuration."""
        mock_singleton.return_value.get_vector_store.return_value = mock_vector_store

        get_session_retriever(
            session_id="sess-123",
            search_type="similarity_score_threshold",
            score_threshold=0.8,
        )

        call_kwargs = mock_vector_store.as_retriever.call_args.kwargs
        assert call_kwargs["search_type"] == "similarity_score_threshold"
        assert call_kwargs["search_kwargs"]["score_threshold"] == 0.8


class TestHybridRetrieverManager:
    """Tests for HybridRetrieverManager singleton."""

    def test_singleton_pattern(self, reset_singletons):
        """Test that only one instance is created."""
        manager1 = get_hybrid_retriever_manager()
        manager2 = get_hybrid_retriever_manager()
        assert manager1 is manager2

    def test_direct_instantiation_singleton(self, reset_singletons):
        """Test singleton via direct instantiation."""
        manager1 = HybridRetrieverManager()
        manager2 = HybridRetrieverManager()
        assert manager1 is manager2

    @patch("src.adr_vector_database.retriever.EnsembleRetriever")
    @patch("src.adr_vector_database.retriever.BM25Retriever")
    @patch("src.adr_vector_database.retriever.get_session_documents")
    @patch("src.adr_vector_database.retriever.get_session_retriever")
    def test_get_retriever_creates_new(
        self,
        mock_get_retriever,
        mock_get_docs,
        mock_bm25,
        mock_ensemble,
        reset_singletons,
        sample_documents,
        mock_base_retriever,
    ):
        """Test that get_retriever creates a new retriever for new session."""
        mock_get_docs.return_value = sample_documents
        mock_get_retriever.return_value = mock_base_retriever
        mock_bm25.from_documents.return_value = mock_base_retriever
        mock_ensemble.return_value = MagicMock()

        manager = get_hybrid_retriever_manager()
        retriever = manager.get_retriever("session-123")

        assert retriever is not None
        assert "session-123" in manager.active_sessions

    @patch("src.adr_vector_database.retriever.EnsembleRetriever")
    @patch("src.adr_vector_database.retriever.BM25Retriever")
    @patch("src.adr_vector_database.retriever.get_session_documents")
    @patch("src.adr_vector_database.retriever.get_session_retriever")
    def test_get_retriever_caches(
        self,
        mock_get_retriever,
        mock_get_docs,
        mock_bm25,
        mock_ensemble,
        reset_singletons,
        sample_documents,
        mock_base_retriever,
    ):
        """Test that get_retriever caches and reuses retrievers."""
        mock_get_docs.return_value = sample_documents
        mock_get_retriever.return_value = mock_base_retriever
        mock_bm25.from_documents.return_value = mock_base_retriever
        mock_ensemble.return_value = MagicMock()

        manager = get_hybrid_retriever_manager()
        retriever1 = manager.get_retriever("session-123")
        retriever2 = manager.get_retriever("session-123")

        assert retriever1 is retriever2
        assert mock_get_docs.call_count == 1  # Only called once

    @patch("src.adr_vector_database.retriever.EnsembleRetriever")
    @patch("src.adr_vector_database.retriever.BM25Retriever")
    @patch("src.adr_vector_database.retriever.get_session_documents")
    @patch("src.adr_vector_database.retriever.get_session_retriever")
    def test_force_refresh(
        self,
        mock_get_retriever,
        mock_get_docs,
        mock_bm25,
        mock_ensemble,
        reset_singletons,
        sample_documents,
        mock_base_retriever,
    ):
        """Test that force_refresh rebuilds the retriever."""
        mock_get_docs.return_value = sample_documents
        mock_get_retriever.return_value = mock_base_retriever
        mock_bm25.from_documents.return_value = mock_base_retriever
        mock_ensemble.return_value = MagicMock()

        manager = get_hybrid_retriever_manager()
        manager.get_retriever("session-123")
        manager.get_retriever("session-123", force_refresh=True)

        assert mock_get_docs.call_count == 2

    @patch("src.adr_vector_database.retriever.get_session_documents")
    def test_raises_on_no_documents(self, mock_get_docs, reset_singletons):
        """Test that error is raised when no documents found."""
        mock_get_docs.return_value = []

        manager = get_hybrid_retriever_manager()

        with pytest.raises(ValueError, match="No documents found"):
            manager.get_retriever("empty-session")

    @patch("src.adr_vector_database.retriever.EnsembleRetriever")
    @patch("src.adr_vector_database.retriever.BM25Retriever")
    @patch("src.adr_vector_database.retriever.get_session_documents")
    @patch("src.adr_vector_database.retriever.get_session_retriever")
    def test_clear_session(
        self,
        mock_get_retriever,
        mock_get_docs,
        mock_bm25,
        mock_ensemble,
        reset_singletons,
        sample_documents,
        mock_base_retriever,
    ):
        """Test clearing a specific session."""
        mock_get_docs.return_value = sample_documents
        mock_get_retriever.return_value = mock_base_retriever
        mock_bm25.from_documents.return_value = mock_base_retriever
        mock_ensemble.return_value = MagicMock()

        manager = get_hybrid_retriever_manager()
        manager.get_retriever("session-123")

        result = manager.clear_session("session-123")

        assert result is True
        assert "session-123" not in manager.active_sessions

    def test_clear_session_nonexistent(self, reset_singletons):
        """Test clearing a nonexistent session."""
        manager = get_hybrid_retriever_manager()
        result = manager.clear_session("nonexistent")
        assert result is False

    @patch("src.adr_vector_database.retriever.EnsembleRetriever")
    @patch("src.adr_vector_database.retriever.BM25Retriever")
    @patch("src.adr_vector_database.retriever.get_session_documents")
    @patch("src.adr_vector_database.retriever.get_session_retriever")
    def test_clear_all(
        self,
        mock_get_retriever,
        mock_get_docs,
        mock_bm25,
        mock_ensemble,
        reset_singletons,
        sample_documents,
        mock_base_retriever,
    ):
        """Test clearing all sessions."""
        mock_get_docs.return_value = sample_documents
        mock_get_retriever.return_value = mock_base_retriever
        mock_bm25.from_documents.return_value = mock_base_retriever
        mock_ensemble.return_value = MagicMock()

        manager = get_hybrid_retriever_manager()
        manager.get_retriever("session-1")
        manager.get_retriever("session-2")

        count = manager.clear_all()

        assert count == 2
        assert len(manager.active_sessions) == 0

    @patch("src.adr_vector_database.retriever.EnsembleRetriever")
    @patch("src.adr_vector_database.retriever.BM25Retriever")
    @patch("src.adr_vector_database.retriever.get_session_documents")
    @patch("src.adr_vector_database.retriever.get_session_retriever")
    def test_is_cached(
        self,
        mock_get_retriever,
        mock_get_docs,
        mock_bm25,
        mock_ensemble,
        reset_singletons,
        sample_documents,
        mock_base_retriever,
    ):
        """Test is_cached method."""
        mock_get_docs.return_value = sample_documents
        mock_get_retriever.return_value = mock_base_retriever
        mock_bm25.from_documents.return_value = mock_base_retriever
        mock_ensemble.return_value = MagicMock()

        manager = get_hybrid_retriever_manager()

        assert manager.is_cached("session-123") is False
        manager.get_retriever("session-123")
        assert manager.is_cached("session-123") is True


class TestGetHybridRetriever:
    """Tests for get_hybrid_retriever function."""

    @patch("src.adr_vector_database.retriever.get_hybrid_retriever_manager")
    def test_uses_cache_by_default(self, mock_manager, reset_singletons):
        """Test that cache is used by default."""
        manager_instance = MagicMock()
        mock_manager.return_value = manager_instance

        get_hybrid_retriever(session_id="sess-123")

        manager_instance.get_retriever.assert_called_once()

    @patch("src.adr_vector_database.retriever.EnsembleRetriever")
    @patch("src.adr_vector_database.retriever.BM25Retriever")
    @patch("src.adr_vector_database.retriever.get_session_documents")
    @patch("src.adr_vector_database.retriever.get_session_retriever")
    def test_non_cached_mode(
        self,
        mock_get_retriever,
        mock_get_docs,
        mock_bm25,
        mock_ensemble,
        reset_singletons,
        sample_documents,
        mock_base_retriever,
    ):
        """Test non-cached mode builds new retriever."""
        mock_get_docs.return_value = sample_documents
        mock_get_retriever.return_value = mock_base_retriever
        mock_bm25.from_documents.return_value = mock_base_retriever
        mock_ensemble.return_value = MagicMock()

        retriever = get_hybrid_retriever(session_id="sess-123", use_cache=False)

        assert retriever is not None
        mock_get_docs.assert_called_once()

    @patch("src.adr_vector_database.retriever.EnsembleRetriever")
    @patch("src.adr_vector_database.retriever.BM25Retriever")
    @patch("src.adr_vector_database.retriever.get_session_retriever")
    def test_with_provided_documents(
        self,
        mock_get_retriever,
        mock_bm25,
        mock_ensemble,
        reset_singletons,
        sample_documents,
        mock_base_retriever,
    ):
        """Test with pre-provided documents."""
        mock_get_retriever.return_value = mock_base_retriever
        mock_bm25.from_documents.return_value = mock_base_retriever
        mock_ensemble.return_value = MagicMock()

        retriever = get_hybrid_retriever(
            session_id="sess-123",
            documents=sample_documents,
            use_cache=False,
        )

        assert retriever is not None

    @patch("src.adr_vector_database.retriever.get_session_documents")
    def test_raises_on_no_documents(self, mock_get_docs, reset_singletons):
        """Test error when no documents found."""
        mock_get_docs.return_value = []

        with pytest.raises(ValueError, match="No documents found"):
            get_hybrid_retriever(session_id="empty-session", use_cache=False)


class TestGetSessionDocuments:
    """Tests for get_session_documents function."""

    @patch("src.core.cloudsql_pg_client.get_cloudsql_client")
    def test_fetches_documents(self, mock_client, reset_singletons, sample_documents):
        """Test fetching documents from database."""
        mock_engine = MagicMock()
        mock_conn = MagicMock()
        mock_result = [(doc.page_content, doc.metadata) for doc in sample_documents]
        mock_conn.execute.return_value = mock_result
        mock_engine.connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_engine.connect.return_value.__exit__ = MagicMock(return_value=None)
        mock_client.return_value.engine = mock_engine

        documents = get_session_documents(session_id="sess-123")

        assert len(documents) == 3
        mock_conn.execute.assert_called_once()

    @patch("src.core.cloudsql_pg_client.get_cloudsql_client")
    def test_returns_empty_list_when_no_docs(self, mock_client, reset_singletons):
        """Test returns empty list when no documents found."""
        mock_engine = MagicMock()
        mock_conn = MagicMock()
        mock_conn.execute.return_value = []
        mock_engine.connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_engine.connect.return_value.__exit__ = MagicMock(return_value=None)
        mock_client.return_value.engine = mock_engine

        documents = get_session_documents(session_id="nonexistent")

        assert documents == []


class TestSessionIsolation:
    """Tests to verify session isolation."""

    @patch("src.adr_vector_database.retriever.get_vector_store_singleton")
    def test_different_sessions_different_filters(
        self, mock_singleton, reset_singletons, mock_vector_store
    ):
        """Test that different sessions use different filters."""
        mock_singleton.return_value.get_vector_store.return_value = mock_vector_store

        get_session_retriever(session_id="session-A")
        get_session_retriever(session_id="session-B")

        calls = mock_vector_store.as_retriever.call_args_list
        assert calls[0].kwargs["search_kwargs"]["filter"] == {"session_id": "session-A"}
        assert calls[1].kwargs["search_kwargs"]["filter"] == {"session_id": "session-B"}


class TestModuleExports:
    """Tests for module-level exports."""

    def test_session_retriever_alias(self):
        """Test SessionRetriever is alias for BaseRetriever."""
        assert SessionRetriever is BaseRetriever

    def test_imports_from_package(self):
        """Test imports work from the package."""
        from src.adr_vector_database import (
            HybridRetrieverManager,
            SessionRetriever,
            VectorStoreSingleton,
            get_hybrid_retriever,
            get_hybrid_retriever_manager,
            get_session_documents,
            get_session_retriever,
            get_vector_store_singleton,
        )

        assert get_session_retriever is not None
        assert get_hybrid_retriever is not None
        assert get_hybrid_retriever_manager is not None
        assert get_session_documents is not None
        assert get_vector_store_singleton is not None
        assert VectorStoreSingleton is not None
        assert HybridRetrieverManager is not None
        assert SessionRetriever is not None
