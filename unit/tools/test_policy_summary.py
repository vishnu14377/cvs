"""Tests for policy summary tool."""

from unittest.mock import MagicMock, patch

from src.tools.policy_summary import PolicySummaryTool


class TestPolicySummaryTool:
    def test_tool_name(self):
        tool = PolicySummaryTool()
        assert tool.name == "policy_summary"

    def _mock_repo(self, policy_ids):
        repo = MagicMock()
        repo.list_all.return_value = [MagicMock(policy_id=pid) for pid in policy_ids]
        return repo

    @patch("src.tools.policy_summary.get_policy_repository")
    @patch("src.tools.policy_summary.LangChainClient")
    @patch("src.tools.policy_summary.get_session_documents")
    def test_returns_per_page_summaries(self, mock_get_docs, mock_llm_cls, mock_repo_fn):
        doc1 = MagicMock()
        doc1.page_content = "Coverage criteria for cardiac catheterization."
        doc1.metadata = {"source": "CPB-0123.pdf", "page": 1, "session_id": "pol_abc"}

        doc2 = MagicMock()
        doc2.page_content = "Medical necessity requirements."
        doc2.metadata = {"source": "CPB-0123.pdf", "page": 2, "session_id": "pol_abc"}

        mock_repo_fn.return_value = self._mock_repo(["pol_abc"])
        mock_get_docs.return_value = [doc1, doc2]

        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Summary."
        mock_llm.invoke.return_value = mock_response
        mock_llm_cls.return_value = MagicMock(client=mock_llm)

        tool = PolicySummaryTool()
        result = tool._run(session_id="chat_sess_xyz")

        # Tool fetches by policy_id, NOT the chat session_id
        mock_get_docs.assert_called_once_with(
            session_id="pol_abc", collection_name="policy_documents"
        )
        assert "CPB-0123.pdf" in result
        assert mock_llm.invoke.call_count == 2

    @patch("src.tools.policy_summary.get_policy_repository")
    @patch("src.tools.policy_summary.LangChainClient")
    @patch("src.tools.policy_summary.get_session_documents")
    def test_no_documents_returns_message(self, mock_get_docs, mock_llm_cls, mock_repo_fn):
        mock_repo_fn.return_value = self._mock_repo([])
        mock_get_docs.return_value = []

        tool = PolicySummaryTool()
        result = tool._run(session_id="chat_sess_xyz")

        assert "no policy documents found" in result.lower()

    @patch("src.tools.policy_summary.get_policy_repository")
    @patch("src.tools.policy_summary.LangChainClient")
    @patch("src.tools.policy_summary.get_session_documents")
    def test_iterates_all_policies(self, mock_get_docs, mock_llm_cls, mock_repo_fn):
        mock_repo_fn.return_value = self._mock_repo(["pol_1", "pol_2", "pol_3"])
        mock_get_docs.return_value = []

        tool = PolicySummaryTool()
        tool._run(session_id="chat_sess_xyz")

        assert mock_get_docs.call_count == 3
        called_session_ids = [c.kwargs["session_id"] for c in mock_get_docs.call_args_list]
        assert called_session_ids == ["pol_1", "pol_2", "pol_3"]

    @patch("src.tools.policy_summary.get_policy_repository")
    @patch("src.tools.policy_summary.LangChainClient")
    @patch("src.tools.policy_summary.get_session_documents")
    def test_output_format(self, mock_get_docs, mock_llm_cls, mock_repo_fn):
        doc = MagicMock()
        doc.page_content = "Policy text."
        doc.metadata = {"source": "CPB-0456.pdf", "page": 5, "session_id": "pol_abc"}
        mock_repo_fn.return_value = self._mock_repo(["pol_abc"])
        mock_get_docs.return_value = [doc]

        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Policy summary."
        mock_llm.invoke.return_value = mock_response
        mock_llm_cls.return_value = MagicMock(client=mock_llm)

        tool = PolicySummaryTool()
        result = tool._run(session_id="chat_sess_xyz")

        assert "--- CPB-0456.pdf | Page 5 ---" in result

    @patch("src.tools.policy_summary.get_policy_repository")
    @patch("src.tools.policy_summary.LangChainClient")
    @patch("src.tools.policy_summary.get_session_documents")
    def test_handles_llm_error_gracefully(self, mock_get_docs, mock_llm_cls, mock_repo_fn):
        doc = MagicMock()
        doc.page_content = "Text."
        doc.metadata = {"source": "p.pdf", "page": 1, "session_id": "pol_abc"}
        mock_repo_fn.return_value = self._mock_repo(["pol_abc"])
        mock_get_docs.return_value = [doc]

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = RuntimeError("LLM down")
        mock_llm_cls.return_value = MagicMock(client=mock_llm)

        tool = PolicySummaryTool()
        result = tool._run(session_id="chat_sess_xyz")

        assert "error" in result.lower()
