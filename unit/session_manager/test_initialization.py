"""
Unit tests for the initialize_session() entry-point.

Tests cover:
- Creates a SessionManager and calls initialize()
- Returns (session_id, AdrProcessingResult, SessionManager)
- Forwards all parameters correctly

All external dependencies are mocked.

Run with: pytest tests/unit/session_manager/test_initialization.py -v
"""

from unittest.mock import MagicMock, patch

from src.session_manager.initialization import initialize_session

_INIT = "src.session_manager.initialization"


# =============================================================================
# Test: initialize_session
# =============================================================================


class TestInitializeSession:
    """Tests for the initialize_session convenience function."""

    @patch(f"{_INIT}.SessionManager")
    def test_returns_tuple_of_three(self, mock_sm_cls):
        """Should return (session_id, result, manager)."""
        mock_result = MagicMock(success=True)
        mock_manager = mock_sm_cls.return_value
        mock_manager.initialize.return_value = ("adr-20260414-aabb", mock_result)

        session_id, result, manager = initialize_session(gcs_uri="gs://b/d.pdf")

        assert session_id == "adr-20260414-aabb"
        assert result is mock_result
        assert manager is mock_manager

    @patch(f"{_INIT}.SessionManager")
    def test_calls_manager_initialize(self, mock_sm_cls):
        """Should call manager.initialize() exactly once."""
        mock_sm_cls.return_value.initialize.return_value = ("id", MagicMock())

        initialize_session(gcs_uri="gs://b/d.pdf")

        mock_sm_cls.return_value.initialize.assert_called_once()

    @patch(f"{_INIT}.SessionManager")
    def test_forwards_all_params(self, mock_sm_cls):
        """Should forward every keyword argument to SessionManager.__init__."""
        mock_sm_cls.return_value.initialize.return_value = ("id", MagicMock())

        initialize_session(
            gcs_uri="gs://b/d.pdf",
            model_type="llm",
            size_limit_mb=10.0,
            pages_per_chunk=5,
            max_workers=3,
            collection_name="col",
            chunk_size=500,
            chunk_overlap=50,
            batch_size=32,
            timeout=60.0,
            additional_metadata={"k": "v"},
            skip_ocr=True,
            skip_ingestion=True,
            use_hybrid=True,
            bm25_weight=0.7,
            semantic_weight=0.3,
            search_type="mmr",
            retriever_k=10,
        )

        mock_sm_cls.assert_called_once_with(
            gcs_uri="gs://b/d.pdf",
            model_type="llm",
            size_limit_mb=10.0,
            pages_per_chunk=5,
            max_workers=3,
            collection_name="col",
            chunk_size=500,
            chunk_overlap=50,
            batch_size=32,
            timeout=60.0,
            additional_metadata={"k": "v"},
            skip_ocr=True,
            skip_ingestion=True,
            use_hybrid=True,
            bm25_weight=0.7,
            semantic_weight=0.3,
            search_type="mmr",
            retriever_k=10,
        )

    @patch(f"{_INIT}.SessionManager")
    def test_default_params(self, mock_sm_cls):
        """With only gcs_uri, should use defaults for everything else."""
        mock_sm_cls.return_value.initialize.return_value = ("id", MagicMock())

        initialize_session(gcs_uri="gs://b/d.pdf")

        mock_sm_cls.assert_called_once_with(
            gcs_uri="gs://b/d.pdf",
            model_type="mistral",
            size_limit_mb=5.0,
            pages_per_chunk=None,
            max_workers=5,
            collection_name=None,
            chunk_size=None,
            chunk_overlap=None,
            batch_size=None,
            timeout=None,
            additional_metadata=None,
            skip_ocr=False,
            skip_ingestion=False,
            use_hybrid=False,
            bm25_weight=0.5,
            semantic_weight=0.5,
            search_type=None,
            retriever_k=None,
        )
