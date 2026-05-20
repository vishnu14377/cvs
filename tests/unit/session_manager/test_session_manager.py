"""
Unit tests for the SessionManager class.

Tests cover:
- Construction and validation
- Read-only properties
- initialize() — session ID generation + document processing
- retriever property — lazy creation, caching, hybrid vs semantic
- get_retriever() — force refresh behaviour
- agent property — delegates to singleton get_agent()
- _ensure_initialised guard

All external dependencies (AdrDocumentProcessor, retriever factories,
agent_factory) are mocked so these tests run in isolation.

Run with: pytest tests/unit/session_manager/test_session_manager.py -v
"""

from unittest.mock import MagicMock, patch

import pytest
from src.session_manager.core.session_manager import SessionManager

# =============================================================================
# Patch targets (module where the names are looked up)
# =============================================================================

_SM = "src.session_manager.core.session_manager"


# =============================================================================
# Test: Construction
# =============================================================================


class TestConstruction:
    """Tests for SessionManager.__init__."""

    def test_accepts_valid_gcs_uri(self):
        """Should accept a non-empty GCS URI."""
        mgr = SessionManager(gcs_uri="gs://bucket/doc.pdf")
        assert mgr.gcs_uri == "gs://bucket/doc.pdf"

    def test_strips_whitespace_from_gcs_uri(self):
        """Should strip leading/trailing whitespace from the URI."""
        mgr = SessionManager(gcs_uri="  gs://bucket/doc.pdf  ")
        assert mgr.gcs_uri == "gs://bucket/doc.pdf"

    def test_raises_on_empty_gcs_uri(self):
        """Should raise ValueError for an empty string."""
        with pytest.raises(ValueError, match="gcs_uri must not be empty"):
            SessionManager(gcs_uri="")

    def test_raises_on_whitespace_only_gcs_uri(self):
        """Should raise ValueError for a whitespace-only string."""
        with pytest.raises(ValueError, match="gcs_uri must not be empty"):
            SessionManager(gcs_uri="   ")

    def test_default_values(self):
        """Should have sensible defaults for optional params."""
        mgr = SessionManager(gcs_uri="gs://b/d.pdf")
        assert mgr.session_id is None
        assert mgr.result is None

    def test_stores_retriever_options(self):
        """Should store hybrid/retriever config for later use."""
        mgr = SessionManager(
            gcs_uri="gs://b/d.pdf",
            use_hybrid=True,
            bm25_weight=0.7,
            semantic_weight=0.3,
            retriever_k=10,
        )
        assert mgr._use_hybrid is True
        assert mgr._bm25_weight == 0.7
        assert mgr._semantic_weight == 0.3
        assert mgr._retriever_k == 10


# =============================================================================
# Test: Read-only properties before initialization
# =============================================================================


class TestPropertiesBeforeInit:
    """Properties should return None before initialize() is called."""

    def test_session_id_is_none(self):
        mgr = SessionManager(gcs_uri="gs://b/d.pdf")
        assert mgr.session_id is None

    def test_result_is_none(self):
        mgr = SessionManager(gcs_uri="gs://b/d.pdf")
        assert mgr.result is None

    def test_gcs_uri_is_always_available(self):
        mgr = SessionManager(gcs_uri="gs://b/d.pdf")
        assert mgr.gcs_uri == "gs://b/d.pdf"


# =============================================================================
# Test: initialize()
# =============================================================================


class TestInitialize:
    """Tests for SessionManager.initialize()."""

    @patch(f"{_SM}.AdrDocumentProcessor")
    @patch(f"{_SM}.generate_session_id", return_value="adr-20260414-aabbccdd")
    def test_returns_session_id_and_result(
        self, mock_gen_id, mock_processor_cls, mock_processing_result
    ):
        """Should return (session_id, AdrProcessingResult)."""
        mock_processor_cls.return_value.process.return_value = mock_processing_result

        mgr = SessionManager(gcs_uri="gs://b/d.pdf")
        session_id, result = mgr.initialize()

        assert session_id == "adr-20260414-aabbccdd"
        assert result is mock_processing_result

    @patch(f"{_SM}.AdrDocumentProcessor")
    @patch(f"{_SM}.generate_session_id", return_value="adr-20260414-aabbccdd")
    def test_sets_session_id_property(
        self, mock_gen_id, mock_processor_cls, mock_processing_result
    ):
        """After initialize(), session_id property should be set."""
        mock_processor_cls.return_value.process.return_value = mock_processing_result

        mgr = SessionManager(gcs_uri="gs://b/d.pdf")
        mgr.initialize()

        assert mgr.session_id == "adr-20260414-aabbccdd"

    @patch(f"{_SM}.AdrDocumentProcessor")
    @patch(f"{_SM}.generate_session_id", return_value="adr-20260414-aabbccdd")
    def test_sets_result_property(self, mock_gen_id, mock_processor_cls, mock_processing_result):
        """After initialize(), result property should be set."""
        mock_processor_cls.return_value.process.return_value = mock_processing_result

        mgr = SessionManager(gcs_uri="gs://b/d.pdf")
        mgr.initialize()

        assert mgr.result is mock_processing_result

    @patch(f"{_SM}.AdrDocumentProcessor")
    @patch(f"{_SM}.generate_session_id", return_value="adr-20260414-aabbccdd")
    def test_passes_config_to_processor(
        self, mock_gen_id, mock_processor_cls, mock_processing_result
    ):
        """Should forward construction params to AdrDocumentProcessor."""
        mock_processor_cls.return_value.process.return_value = mock_processing_result

        mgr = SessionManager(
            gcs_uri="gs://b/d.pdf",
            model_type="llm",
            size_limit_mb=10.0,
            max_workers=3,
            collection_name="test_col",
            chunk_size=500,
            chunk_overlap=50,
            batch_size=32,
        )
        mgr.initialize()

        mock_processor_cls.assert_called_once_with(
            session_id="adr-20260414-aabbccdd",
            model_type="llm",
            size_limit_mb=10.0,
            pages_per_chunk=None,
            max_workers=3,
            collection_name="test_col",
            chunk_size=500,
            chunk_overlap=50,
            batch_size=32,
        )

    @patch(f"{_SM}.AdrDocumentProcessor")
    @patch(f"{_SM}.generate_session_id", return_value="adr-20260414-aabbccdd")
    def test_passes_process_kwargs(self, mock_gen_id, mock_processor_cls, mock_processing_result):
        """Should forward runtime params to processor.process()."""
        mock_processor_cls.return_value.process.return_value = mock_processing_result

        mgr = SessionManager(
            gcs_uri="gs://b/d.pdf",
            timeout=30.0,
            additional_metadata={"key": "val"},
            skip_ocr=True,
            skip_ingestion=True,
        )
        mgr.initialize()

        mock_processor_cls.return_value.process.assert_called_once_with(
            gcs_uri="gs://b/d.pdf",
            timeout=30.0,
            additional_metadata={"key": "val"},
            skip_ocr=True,
            skip_ingestion=True,
        )


# =============================================================================
# Test: _ensure_initialised guard
# =============================================================================


class TestEnsureInitialised:
    """Tests for the _ensure_initialised guard."""

    def test_retriever_before_init_raises(self):
        """Accessing retriever before initialize() should raise."""
        mgr = SessionManager(gcs_uri="gs://b/d.pdf")
        with pytest.raises(RuntimeError, match="Session not initialised"):
            _ = mgr.retriever

    def test_agent_before_init_raises(self):
        """Accessing agent before initialize() should raise."""
        mgr = SessionManager(gcs_uri="gs://b/d.pdf")
        with pytest.raises(RuntimeError, match="Session not initialised"):
            _ = mgr.agent


# =============================================================================
# Test: retriever property (semantic)
# =============================================================================


class TestSemanticRetriever:
    """Tests for the default (semantic) retriever path."""

    @patch(f"{_SM}.get_agent")
    @patch(f"{_SM}.get_session_retriever")
    @patch(f"{_SM}.AdrDocumentProcessor")
    @patch(f"{_SM}.generate_session_id", return_value="adr-20260414-aabbccdd")
    def test_creates_semantic_retriever(
        self,
        mock_gen_id,
        mock_processor_cls,
        mock_get_retriever,
        mock_get_agent,
        mock_processing_result,
        mock_retriever,
    ):
        """Should call get_session_retriever when use_hybrid=False."""
        mock_processor_cls.return_value.process.return_value = mock_processing_result
        mock_get_retriever.return_value = mock_retriever

        mgr = SessionManager(gcs_uri="gs://b/d.pdf", retriever_k=5)
        mgr.initialize()
        ret = mgr.retriever

        mock_get_retriever.assert_called_once_with(
            session_id="adr-20260414-aabbccdd",
            search_type=None,
            k=5,
            collection_name=None,
        )
        assert ret is mock_retriever

    @patch(f"{_SM}.get_agent")
    @patch(f"{_SM}.get_session_retriever")
    @patch(f"{_SM}.AdrDocumentProcessor")
    @patch(f"{_SM}.generate_session_id", return_value="adr-20260414-aabbccdd")
    def test_caches_retriever(
        self,
        mock_gen_id,
        mock_processor_cls,
        mock_get_retriever,
        mock_get_agent,
        mock_processing_result,
        mock_retriever,
    ):
        """Accessing retriever twice should only create it once."""
        mock_processor_cls.return_value.process.return_value = mock_processing_result
        mock_get_retriever.return_value = mock_retriever

        mgr = SessionManager(gcs_uri="gs://b/d.pdf")
        mgr.initialize()

        ret1 = mgr.retriever
        ret2 = mgr.retriever

        assert ret1 is ret2
        assert mock_get_retriever.call_count == 1


# =============================================================================
# Test: retriever property (hybrid)
# =============================================================================


class TestHybridRetriever:
    """Tests for the hybrid retriever path."""

    @patch(f"{_SM}.get_agent")
    @patch(f"{_SM}.get_hybrid_retriever")
    @patch(f"{_SM}.AdrDocumentProcessor")
    @patch(f"{_SM}.generate_session_id", return_value="adr-20260414-aabbccdd")
    def test_creates_hybrid_retriever(
        self,
        mock_gen_id,
        mock_processor_cls,
        mock_get_hybrid,
        mock_get_agent,
        mock_processing_result,
        mock_retriever,
    ):
        """Should call get_hybrid_retriever when use_hybrid=True."""
        mock_processor_cls.return_value.process.return_value = mock_processing_result
        mock_get_hybrid.return_value = mock_retriever

        mgr = SessionManager(
            gcs_uri="gs://b/d.pdf",
            use_hybrid=True,
            bm25_weight=0.6,
            semantic_weight=0.4,
            retriever_k=8,
            collection_name="my_col",
        )
        mgr.initialize()
        ret = mgr.retriever

        mock_get_hybrid.assert_called_once_with(
            session_id="adr-20260414-aabbccdd",
            k=8,
            bm25_weight=0.6,
            semantic_weight=0.4,
            semantic_search_type=None,
            collection_name="my_col",
        )
        assert ret is mock_retriever


# =============================================================================
# Test: get_retriever (force_refresh)
# =============================================================================


class TestGetRetriever:
    """Tests for get_retriever with force_refresh."""

    @patch(f"{_SM}.get_agent")
    @patch(f"{_SM}.get_session_retriever")
    @patch(f"{_SM}.AdrDocumentProcessor")
    @patch(f"{_SM}.generate_session_id", return_value="adr-20260414-aabbccdd")
    def test_force_refresh_recreates_retriever(
        self,
        mock_gen_id,
        mock_processor_cls,
        mock_get_retriever,
        mock_get_agent,
        mock_processing_result,
    ):
        """force_refresh=True should discard the cached retriever."""
        mock_processor_cls.return_value.process.return_value = mock_processing_result
        ret_a = MagicMock(name="retriever_a")
        ret_b = MagicMock(name="retriever_b")
        mock_get_retriever.side_effect = [ret_a, ret_b]

        mgr = SessionManager(gcs_uri="gs://b/d.pdf")
        mgr.initialize()

        first = mgr.get_retriever()
        second = mgr.get_retriever(force_refresh=True)

        assert first is ret_a
        assert second is ret_b
        assert mock_get_retriever.call_count == 2

    @patch(f"{_SM}.get_agent")
    @patch(f"{_SM}.get_session_retriever")
    @patch(f"{_SM}.AdrDocumentProcessor")
    @patch(f"{_SM}.generate_session_id", return_value="adr-20260414-aabbccdd")
    def test_no_refresh_returns_cached(
        self,
        mock_gen_id,
        mock_processor_cls,
        mock_get_retriever,
        mock_get_agent,
        mock_processing_result,
    ):
        """force_refresh=False (default) should return the cached retriever."""
        mock_processor_cls.return_value.process.return_value = mock_processing_result
        mock_get_retriever.return_value = MagicMock()

        mgr = SessionManager(gcs_uri="gs://b/d.pdf")
        mgr.initialize()

        first = mgr.get_retriever()
        second = mgr.get_retriever(force_refresh=False)

        assert first is second
        assert mock_get_retriever.call_count == 1


# =============================================================================
# Test: agent property
# =============================================================================


class TestAgentProperty:
    """Tests for the agent property."""

    @patch(f"{_SM}.get_agent")
    @patch(f"{_SM}.AdrDocumentProcessor")
    @patch(f"{_SM}.generate_session_id", return_value="adr-20260414-aabbccdd")
    def test_returns_singleton_agent(
        self,
        mock_gen_id,
        mock_processor_cls,
        mock_get_agent,
        mock_processing_result,
        mock_agent_graph,
    ):
        """Should delegate to get_agent() and return the singleton graph."""
        mock_processor_cls.return_value.process.return_value = mock_processing_result
        mock_get_agent.return_value = mock_agent_graph

        mgr = SessionManager(gcs_uri="gs://b/d.pdf")
        mgr.initialize()

        agent = mgr.agent

        mock_get_agent.assert_called_once()
        assert agent is mock_agent_graph

    @patch(f"{_SM}.get_agent")
    @patch(f"{_SM}.AdrDocumentProcessor")
    @patch(f"{_SM}.generate_session_id", return_value="adr-20260414-aabbccdd")
    def test_multiple_accesses_call_get_agent_each_time(
        self,
        mock_gen_id,
        mock_processor_cls,
        mock_get_agent,
        mock_processing_result,
        mock_agent_graph,
    ):
        """Each access to .agent calls get_agent() (caching is its job)."""
        mock_processor_cls.return_value.process.return_value = mock_processing_result
        mock_get_agent.return_value = mock_agent_graph

        mgr = SessionManager(gcs_uri="gs://b/d.pdf")
        mgr.initialize()

        _ = mgr.agent
        _ = mgr.agent

        assert mock_get_agent.call_count == 2
