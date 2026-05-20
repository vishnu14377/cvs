"""
Unit tests for the session deletion module.

Tests cover:
- SessionDeletionResult dataclass behaviour
- Input validation (empty / whitespace session IDs)
- Happy-path deletion with all five steps succeeding
- Individual step failures are caught and recorded as errors
- Boolean flags to skip individual cleanup steps
- Best-effort semantics (one failure doesn't prevent the rest)
- Collection name forwarding
- Session ID whitespace stripping
- Conversation-history clearing (with / without graph)

All external dependencies (delete_session_documents, get_hybrid_retriever_manager,
clear_session_history, delete_session_folder, cleanup_local_data) are mocked so
these tests run in isolation.

Run with: pytest tests/unit/session_manager/test_deletion.py -v
"""

from unittest.mock import patch, sentinel

import pytest
from src.session_manager.deletion import SessionDeletionResult, delete_session

# =============================================================================
# Patch targets (module where the names are looked up at runtime)
# =============================================================================

_DEL = "src.session_manager.deletion"


# =============================================================================
# Test: SessionDeletionResult dataclass
# =============================================================================


class TestSessionDeletionResult:
    """Tests for the SessionDeletionResult dataclass."""

    def test_defaults(self):
        """All numeric/boolean fields default to zero/False, errors is empty."""
        r = SessionDeletionResult(session_id="s1")
        assert r.session_id == "s1"
        assert r.vectors_deleted == 0
        assert r.retriever_cache_cleared is False
        assert r.history_cleared is False
        assert r.gcs_blobs_deleted == 0
        assert r.local_files_cleaned is False
        assert r.errors == []

    def test_success_when_no_errors(self):
        """success property is True when errors list is empty."""
        r = SessionDeletionResult(session_id="s1")
        assert r.success is True

    def test_not_success_when_errors(self):
        """success property is False when errors list is non-empty."""
        r = SessionDeletionResult(session_id="s1", errors=["boom"])
        assert r.success is False

    def test_errors_are_independent_per_instance(self):
        """Default errors list is not shared across instances."""
        r1 = SessionDeletionResult(session_id="s1")
        r2 = SessionDeletionResult(session_id="s2")
        r1.errors.append("oops")
        assert r2.errors == []


# =============================================================================
# Test: Input validation
# =============================================================================


class TestInputValidation:
    """Tests for delete_session argument validation."""

    def test_empty_session_id_raises(self):
        """Should raise ValueError for an empty string."""
        with pytest.raises(ValueError, match="session_id must not be empty"):
            delete_session("")

    def test_whitespace_only_session_id_raises(self):
        """Should raise ValueError for whitespace-only string."""
        with pytest.raises(ValueError, match="session_id must not be empty"):
            delete_session("   ")

    @patch(f"{_DEL}.cleanup_local_data", return_value=True)
    @patch(f"{_DEL}.delete_session_folder", return_value=0)
    @patch(f"{_DEL}.get_hybrid_retriever_manager")
    @patch(f"{_DEL}.delete_session_documents", return_value=0)
    def test_strips_whitespace(self, mock_vec, mock_ret, mock_gcs, mock_local):
        """Session ID should be stripped before use."""
        mock_ret.return_value.clear_session.return_value = False
        result = delete_session("  adr-123  ")
        assert result.session_id == "adr-123"


# =============================================================================
# Test: Happy-path (all steps succeed)
# =============================================================================


class TestHappyPath:
    """Tests for delete_session when every step succeeds."""

    @patch(f"{_DEL}.cleanup_local_data", return_value=True)
    @patch(f"{_DEL}.delete_session_folder", return_value=5)
    @patch(f"{_DEL}.clear_session_history", return_value=True)
    @patch(f"{_DEL}.get_hybrid_retriever_manager")
    @patch(f"{_DEL}.delete_session_documents", return_value=42)
    def test_all_steps_succeed(self, mock_vec, mock_ret, mock_hist, mock_gcs, mock_local):
        """All five fields should be populated and success should be True."""
        mock_ret.return_value.clear_session.return_value = True
        mock_graph = sentinel.graph

        result = delete_session(
            "adr-20260414-abc",
            graph=mock_graph,
            delete_vectors=True,
            clear_retriever_cache=True,
            clear_history=True,
            delete_gcs=True,
            delete_local=True,
        )

        assert result.session_id == "adr-20260414-abc"
        assert result.vectors_deleted == 42
        assert result.retriever_cache_cleared is True
        assert result.history_cleared is True
        assert result.gcs_blobs_deleted == 5
        assert result.local_files_cleaned is True
        assert result.success is True
        assert result.errors == []
        mock_hist.assert_called_once_with(mock_graph, "adr-20260414-abc")

    @patch(f"{_DEL}.cleanup_local_data", return_value=True)
    @patch(f"{_DEL}.delete_session_folder", return_value=5)
    @patch(f"{_DEL}.clear_session_history", return_value=True)
    @patch(f"{_DEL}.get_hybrid_retriever_manager")
    @patch(f"{_DEL}.delete_session_documents", return_value=42)
    def test_calls_delete_session_documents_with_correct_args(
        self, mock_vec, mock_ret, mock_hist, mock_gcs, mock_local
    ):
        """Should forward session_id and collection_name to delete_session_documents."""
        mock_ret.return_value.clear_session.return_value = True

        delete_session(
            "adr-123",
            collection_name="custom_col",
            delete_gcs=True,
        )

        mock_vec.assert_called_once_with(
            session_id="adr-123",
            collection_name="custom_col",
        )

    @patch(f"{_DEL}.cleanup_local_data", return_value=True)
    @patch(f"{_DEL}.delete_session_folder", return_value=0)
    @patch(f"{_DEL}.clear_session_history", return_value=True)
    @patch(f"{_DEL}.get_hybrid_retriever_manager")
    @patch(f"{_DEL}.delete_session_documents", return_value=0)
    def test_collection_name_defaults_to_none(
        self, mock_vec, mock_ret, mock_hist, mock_gcs, mock_local
    ):
        """When collection_name is not provided it should be passed as None."""
        mock_ret.return_value.clear_session.return_value = False

        delete_session("adr-123")

        mock_vec.assert_called_once_with(
            session_id="adr-123",
            collection_name=None,
        )

    @patch(f"{_DEL}.cleanup_local_data", return_value=True)
    @patch(f"{_DEL}.delete_session_folder", return_value=0)
    @patch(f"{_DEL}.clear_session_history", return_value=True)
    @patch(f"{_DEL}.get_hybrid_retriever_manager")
    @patch(f"{_DEL}.delete_session_documents", return_value=0)
    def test_retriever_cache_cleared_true_when_nothing_cached(
        self, mock_vec, mock_ret, mock_hist, mock_gcs, mock_local
    ):
        """retriever_cache_cleared should be True even when clear_session finds nothing — 'nothing to clear' is still a successful clear."""
        mock_ret.return_value.clear_session.return_value = False

        result = delete_session("adr-123")

        assert result.retriever_cache_cleared is True
        assert result.success is True

    @patch(f"{_DEL}.cleanup_local_data", return_value=True)
    @patch(f"{_DEL}.delete_session_folder", return_value=0)
    @patch(f"{_DEL}.clear_session_history", return_value=True)
    @patch(f"{_DEL}.get_hybrid_retriever_manager")
    @patch(f"{_DEL}.delete_session_documents", return_value=0)
    def test_history_cleared_when_graph_provided(
        self, mock_vec, mock_ret, mock_hist, mock_gcs, mock_local
    ):
        """history_cleared should be True when graph is provided and clear succeeds."""
        mock_ret.return_value.clear_session.return_value = True
        mock_graph = sentinel.graph

        result = delete_session("adr-123", graph=mock_graph, clear_history=True)

        assert result.history_cleared is True
        mock_hist.assert_called_once_with(mock_graph, "adr-123")

    @patch(f"{_DEL}.cleanup_local_data", return_value=True)
    @patch(f"{_DEL}.delete_session_folder", return_value=0)
    @patch(f"{_DEL}.clear_session_history", return_value=True)
    @patch(f"{_DEL}.get_hybrid_retriever_manager")
    @patch(f"{_DEL}.delete_session_documents", return_value=0)
    def test_history_not_cleared_when_no_graph(
        self, mock_vec, mock_ret, mock_hist, mock_gcs, mock_local
    ):
        """history_cleared should stay False when graph is not provided."""
        mock_ret.return_value.clear_session.return_value = True

        result = delete_session("adr-123", clear_history=True)

        assert result.history_cleared is False
        mock_hist.assert_not_called()


# =============================================================================
# Test: Boolean flags — skipping individual steps
# =============================================================================


class TestFlags:
    """Tests that boolean flags correctly skip cleanup steps."""

    @patch(f"{_DEL}.cleanup_local_data", return_value=True)
    @patch(f"{_DEL}.delete_session_folder", return_value=0)
    @patch(f"{_DEL}.clear_session_history", return_value=True)
    @patch(f"{_DEL}.get_hybrid_retriever_manager")
    @patch(f"{_DEL}.delete_session_documents", return_value=10)
    def test_skip_vectors(self, mock_vec, mock_ret, mock_hist, mock_gcs, mock_local):
        """delete_vectors=False should skip vector-store deletion."""
        mock_ret.return_value.clear_session.return_value = False

        result = delete_session("adr-123", delete_vectors=False)

        mock_vec.assert_not_called()
        assert result.vectors_deleted == 0

    @patch(f"{_DEL}.cleanup_local_data", return_value=True)
    @patch(f"{_DEL}.delete_session_folder", return_value=0)
    @patch(f"{_DEL}.clear_session_history", return_value=True)
    @patch(f"{_DEL}.get_hybrid_retriever_manager")
    @patch(f"{_DEL}.delete_session_documents", return_value=0)
    def test_skip_retriever_cache(self, mock_vec, mock_ret, mock_hist, mock_gcs, mock_local):
        """clear_retriever_cache=False should skip cache clearing."""
        result = delete_session("adr-123", clear_retriever_cache=False)

        mock_ret.assert_not_called()
        assert result.retriever_cache_cleared is False

    @patch(f"{_DEL}.cleanup_local_data", return_value=True)
    @patch(f"{_DEL}.delete_session_folder", return_value=0)
    @patch(f"{_DEL}.clear_session_history", return_value=True)
    @patch(f"{_DEL}.get_hybrid_retriever_manager")
    @patch(f"{_DEL}.delete_session_documents", return_value=0)
    def test_skip_history(self, mock_vec, mock_ret, mock_hist, mock_gcs, mock_local):
        """clear_history=False should skip conversation-history clearing."""
        mock_ret.return_value.clear_session.return_value = True
        mock_graph = sentinel.graph

        result = delete_session("adr-123", graph=mock_graph, clear_history=False)

        mock_hist.assert_not_called()
        assert result.history_cleared is False

    @patch(f"{_DEL}.cleanup_local_data", return_value=True)
    @patch(f"{_DEL}.delete_session_folder", return_value=10)
    @patch(f"{_DEL}.clear_session_history", return_value=True)
    @patch(f"{_DEL}.get_hybrid_retriever_manager")
    @patch(f"{_DEL}.delete_session_documents", return_value=0)
    def test_skip_gcs(self, mock_vec, mock_ret, mock_hist, mock_gcs, mock_local):
        """delete_gcs=False should skip GCS deletion."""
        mock_ret.return_value.clear_session.return_value = False

        result = delete_session("adr-123", delete_gcs=False)

        mock_gcs.assert_not_called()
        assert result.gcs_blobs_deleted == 0

    @patch(f"{_DEL}.cleanup_local_data", return_value=True)
    @patch(f"{_DEL}.delete_session_folder", return_value=0)
    @patch(f"{_DEL}.clear_session_history", return_value=True)
    @patch(f"{_DEL}.get_hybrid_retriever_manager")
    @patch(f"{_DEL}.delete_session_documents", return_value=0)
    def test_skip_local(self, mock_vec, mock_ret, mock_hist, mock_gcs, mock_local):
        """delete_local=False should skip local file deletion."""
        mock_ret.return_value.clear_session.return_value = False

        result = delete_session("adr-123", delete_local=False)

        mock_local.assert_not_called()
        assert result.local_files_cleaned is False

    @patch(f"{_DEL}.cleanup_local_data")
    @patch(f"{_DEL}.delete_session_folder")
    @patch(f"{_DEL}.clear_session_history")
    @patch(f"{_DEL}.get_hybrid_retriever_manager")
    @patch(f"{_DEL}.delete_session_documents")
    def test_skip_all(self, mock_vec, mock_ret, mock_hist, mock_gcs, mock_local):
        """When all flags are False, nothing should be called."""
        result = delete_session(
            "adr-123",
            delete_vectors=False,
            clear_retriever_cache=False,
            clear_history=False,
            delete_gcs=False,
            delete_local=False,
        )

        mock_vec.assert_not_called()
        mock_ret.assert_not_called()
        mock_hist.assert_not_called()
        mock_gcs.assert_not_called()
        mock_local.assert_not_called()

        assert result.vectors_deleted == 0
        assert result.retriever_cache_cleared is False
        assert result.history_cleared is False
        assert result.gcs_blobs_deleted == 0
        assert result.local_files_cleaned is False
        assert result.success is True


# =============================================================================
# Test: Error handling (best-effort semantics)
# =============================================================================


class TestErrorHandling:
    """Tests that failures in one step don't block the others."""

    @patch(f"{_DEL}.cleanup_local_data", return_value=True)
    @patch(f"{_DEL}.delete_session_folder", return_value=3)
    @patch(f"{_DEL}.clear_session_history", return_value=True)
    @patch(f"{_DEL}.get_hybrid_retriever_manager")
    @patch(f"{_DEL}.delete_session_documents", side_effect=RuntimeError("db down"))
    def test_vector_failure_does_not_block_others(
        self, mock_vec, mock_ret, mock_hist, mock_gcs, mock_local
    ):
        """If vector deletion fails, the other steps still run."""
        mock_ret.return_value.clear_session.return_value = True

        result = delete_session("adr-123", graph=sentinel.graph, delete_gcs=True)

        assert result.vectors_deleted == 0
        assert result.retriever_cache_cleared is True
        assert result.history_cleared is True
        assert result.gcs_blobs_deleted == 3
        assert result.local_files_cleaned is True
        assert result.success is False
        assert len(result.errors) == 1
        assert "db down" in result.errors[0]

    @patch(f"{_DEL}.cleanup_local_data", return_value=True)
    @patch(f"{_DEL}.delete_session_folder", return_value=0)
    @patch(f"{_DEL}.clear_session_history", return_value=True)
    @patch(f"{_DEL}.get_hybrid_retriever_manager", side_effect=RuntimeError("cache err"))
    @patch(f"{_DEL}.delete_session_documents", return_value=5)
    def test_retriever_failure_does_not_block_others(
        self, mock_vec, mock_ret, mock_hist, mock_gcs, mock_local
    ):
        """If retriever cache clearing fails, the other steps still run."""
        result = delete_session("adr-123")

        assert result.vectors_deleted == 5
        assert result.retriever_cache_cleared is False
        assert result.local_files_cleaned is True
        assert result.success is False
        assert "cache err" in result.errors[0]

    @patch(f"{_DEL}.cleanup_local_data", return_value=True)
    @patch(f"{_DEL}.delete_session_folder", return_value=0)
    @patch(f"{_DEL}.clear_session_history", side_effect=RuntimeError("checkpointer err"))
    @patch(f"{_DEL}.get_hybrid_retriever_manager")
    @patch(f"{_DEL}.delete_session_documents", return_value=5)
    def test_history_failure_does_not_block_others(
        self, mock_vec, mock_ret, mock_hist, mock_gcs, mock_local
    ):
        """If conversation history clearing fails, the other steps still run."""
        mock_ret.return_value.clear_session.return_value = True

        result = delete_session("adr-123", graph=sentinel.graph)

        assert result.vectors_deleted == 5
        assert result.retriever_cache_cleared is True
        assert result.history_cleared is False
        assert result.local_files_cleaned is True
        assert result.success is False
        assert "checkpointer err" in result.errors[0]

    @patch(f"{_DEL}.cleanup_local_data", return_value=True)
    @patch(f"{_DEL}.delete_session_folder", side_effect=PermissionError("no access"))
    @patch(f"{_DEL}.clear_session_history", return_value=True)
    @patch(f"{_DEL}.get_hybrid_retriever_manager")
    @patch(f"{_DEL}.delete_session_documents", return_value=5)
    def test_gcs_failure_does_not_block_others(
        self, mock_vec, mock_ret, mock_hist, mock_gcs, mock_local
    ):
        """If GCS deletion fails, the other steps still run."""
        mock_ret.return_value.clear_session.return_value = True

        result = delete_session("adr-123", delete_gcs=True)

        assert result.vectors_deleted == 5
        assert result.retriever_cache_cleared is True
        assert result.gcs_blobs_deleted == 0
        assert result.local_files_cleaned is True
        assert result.success is False
        assert "no access" in result.errors[0]

    @patch(f"{_DEL}.cleanup_local_data", side_effect=OSError("disk error"))
    @patch(f"{_DEL}.delete_session_folder", return_value=3)
    @patch(f"{_DEL}.clear_session_history", return_value=True)
    @patch(f"{_DEL}.get_hybrid_retriever_manager")
    @patch(f"{_DEL}.delete_session_documents", return_value=10)
    def test_local_failure_does_not_block_others(
        self, mock_vec, mock_ret, mock_hist, mock_gcs, mock_local
    ):
        """If local cleanup fails, the other steps still run."""
        mock_ret.return_value.clear_session.return_value = True

        result = delete_session("adr-123", delete_gcs=True)

        assert result.vectors_deleted == 10
        assert result.retriever_cache_cleared is True
        assert result.gcs_blobs_deleted == 3
        assert result.local_files_cleaned is False
        assert result.success is False
        assert "disk error" in result.errors[0]

    @patch(f"{_DEL}.cleanup_local_data", side_effect=OSError("disk error"))
    @patch(f"{_DEL}.delete_session_folder", side_effect=PermissionError("no access"))
    @patch(f"{_DEL}.clear_session_history", side_effect=RuntimeError("checkpointer err"))
    @patch(f"{_DEL}.get_hybrid_retriever_manager", side_effect=RuntimeError("cache err"))
    @patch(f"{_DEL}.delete_session_documents", side_effect=RuntimeError("db down"))
    def test_all_steps_fail(self, mock_vec, mock_ret, mock_hist, mock_gcs, mock_local):
        """When all steps fail, all errors are collected."""
        result = delete_session("adr-123", graph=sentinel.graph, delete_gcs=True)

        assert result.vectors_deleted == 0
        assert result.retriever_cache_cleared is False
        assert result.history_cleared is False
        assert result.gcs_blobs_deleted == 0
        assert result.local_files_cleaned is False
        assert result.success is False
        assert len(result.errors) == 5
        assert "db down" in result.errors[0]
        assert "cache err" in result.errors[1]
        assert "checkpointer err" in result.errors[2]
        assert "no access" in result.errors[3]
        assert "disk error" in result.errors[4]

    @patch(f"{_DEL}.cleanup_local_data", side_effect=OSError("disk error"))
    @patch(f"{_DEL}.delete_session_folder", side_effect=PermissionError("no access"))
    @patch(f"{_DEL}.clear_session_history", side_effect=RuntimeError("checkpointer err"))
    @patch(f"{_DEL}.get_hybrid_retriever_manager")
    @patch(f"{_DEL}.delete_session_documents", side_effect=RuntimeError("db down"))
    def test_skipped_steps_dont_add_errors(
        self, mock_vec, mock_ret, mock_hist, mock_gcs, mock_local
    ):
        """Skipped steps should not contribute errors even if their mocks would fail."""
        result = delete_session(
            "adr-123",
            delete_vectors=True,
            clear_retriever_cache=False,
            clear_history=False,
            delete_gcs=False,
            delete_local=True,
        )

        # Only vectors (step 1) and local (step 5) ran and failed
        assert len(result.errors) == 2
        assert "db down" in result.errors[0]
        assert "disk error" in result.errors[1]


# =============================================================================
# Test: Default flag values
# =============================================================================


class TestDefaultFlags:
    """Tests that the default flag values match the module's current defaults."""

    @patch(f"{_DEL}.cleanup_local_data", return_value=True)
    @patch(f"{_DEL}.delete_session_folder", return_value=0)
    @patch(f"{_DEL}.clear_session_history", return_value=True)
    @patch(f"{_DEL}.get_hybrid_retriever_manager")
    @patch(f"{_DEL}.delete_session_documents", return_value=0)
    def test_default_flags_run_vectors_retriever_local_skip_gcs(
        self, mock_vec, mock_ret, mock_hist, mock_gcs, mock_local
    ):
        """By default: vectors ✓, retriever ✓, history ✓ (needs graph), gcs ✗, local ✓."""
        mock_ret.return_value.clear_session.return_value = False

        delete_session("adr-123")

        # Vectors and local should be called
        mock_vec.assert_called_once()
        mock_ret.assert_called_once()
        mock_local.assert_called_once()

        # GCS should NOT be called (delete_gcs defaults to False)
        mock_gcs.assert_not_called()

        # History clear is default=True but graph=None, so not called
        mock_hist.assert_not_called()

    @patch(f"{_DEL}.cleanup_local_data", return_value=True)
    @patch(f"{_DEL}.delete_session_folder", return_value=0)
    @patch(f"{_DEL}.clear_session_history", return_value=True)
    @patch(f"{_DEL}.get_hybrid_retriever_manager")
    @patch(f"{_DEL}.delete_session_documents", return_value=0)
    def test_default_flags_with_graph_clears_history(
        self, mock_vec, mock_ret, mock_hist, mock_gcs, mock_local
    ):
        """When graph is provided, default flags should also clear history."""
        mock_ret.return_value.clear_session.return_value = True
        mock_graph = sentinel.graph

        result = delete_session("adr-123", graph=mock_graph)

        mock_hist.assert_called_once_with(mock_graph, "adr-123")
        assert result.history_cleared is True
