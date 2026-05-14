"""Tests for policy search tool."""

from unittest.mock import MagicMock, patch

from src.tools.policy_search import PolicySearchTool


class TestPolicySearchTool:
    """Tests for PolicySearchTool."""

    def test_tool_name(self):
        tool = PolicySearchTool()
        assert tool.name == "policy_search"

    @patch("src.tools.policy_search.get_vector_store_singleton")
    def test_search_returns_results(self, mock_singleton_fn):
        mock_retriever = MagicMock()
        mock_doc = MagicMock()
        mock_doc.page_content = "Policy text about cardiology"
        mock_doc.metadata = {"source": "policy.pdf", "page": 3, "policy_name": "CPB-123"}
        mock_retriever.invoke.return_value = [mock_doc]
        mock_store = MagicMock()
        mock_store.as_retriever.return_value = mock_retriever
        mock_singleton_fn.return_value.get_vector_store.return_value = mock_store

        tool = PolicySearchTool()
        result = tool._run(query="cardiology guidelines", session_id="")
        assert "cardiology" in result.lower()
        assert "1 relevant" in result.lower()
        # Verify retriever does NOT filter by session_id — policies span all sessions
        call_kwargs = mock_store.as_retriever.call_args.kwargs
        assert "filter" not in call_kwargs.get("search_kwargs", {})

    @patch("src.tools.policy_search.get_vector_store_singleton")
    def test_search_empty_results(self, mock_singleton_fn):
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = []
        mock_store = MagicMock()
        mock_store.as_retriever.return_value = mock_retriever
        mock_singleton_fn.return_value.get_vector_store.return_value = mock_store

        tool = PolicySearchTool()
        result = tool._run(query="nonexistent topic", session_id="")
        assert "no relevant" in result.lower()
