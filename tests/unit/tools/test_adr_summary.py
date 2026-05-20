"""Tests for ADR summary tool."""

from unittest.mock import MagicMock, patch

from src.tools.adr_summary import AdrSummaryTool


class TestAdrSummaryTool:
    def test_tool_name(self):
        tool = AdrSummaryTool()
        assert tool.name == "adr_summary"

    @patch("src.tools.adr_summary.LangChainClient")
    @patch("src.tools.adr_summary.get_session_documents")
    def test_returns_per_page_summaries(self, mock_get_docs, mock_llm_cls):
        doc_page1 = MagicMock()
        doc_page1.page_content = "Patient admitted with chest pain."
        doc_page1.metadata = {"source": "discharge.pdf", "page": 1, "session_id": "s1"}

        doc_page2 = MagicMock()
        doc_page2.page_content = "Echo showed normal EF."
        doc_page2.metadata = {"source": "discharge.pdf", "page": 2, "session_id": "s1"}

        mock_get_docs.return_value = [doc_page1, doc_page2]

        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Patient presented with chest pain."
        mock_llm.invoke.return_value = mock_response
        mock_llm_cls.return_value = MagicMock(client=mock_llm)

        tool = AdrSummaryTool()
        result = tool._run(session_id="s1")

        assert "discharge.pdf" in result
        assert "Page 1" in result
        assert "Page 2" in result
        assert mock_llm.invoke.call_count == 2

    @patch("src.tools.adr_summary.LangChainClient")
    @patch("src.tools.adr_summary.get_session_documents")
    def test_groups_multiple_chunks_same_page(self, mock_get_docs, mock_llm_cls):
        chunk1 = MagicMock()
        chunk1.page_content = "First chunk."
        chunk1.metadata = {"source": "notes.pdf", "page": 1, "session_id": "s1"}

        chunk2 = MagicMock()
        chunk2.page_content = "Second chunk."
        chunk2.metadata = {"source": "notes.pdf", "page": 1, "session_id": "s1"}

        mock_get_docs.return_value = [chunk1, chunk2]

        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Combined summary."
        mock_llm.invoke.return_value = mock_response
        mock_llm_cls.return_value = MagicMock(client=mock_llm)

        tool = AdrSummaryTool()
        tool._run(session_id="s1")

        assert mock_llm.invoke.call_count == 1

    @patch("src.tools.adr_summary.LangChainClient")
    @patch("src.tools.adr_summary.get_session_documents")
    def test_no_documents_returns_message(self, mock_get_docs, mock_llm_cls):
        mock_get_docs.return_value = []

        tool = AdrSummaryTool()
        result = tool._run(session_id="empty")

        assert "no adr documents found" in result.lower()

    @patch("src.tools.adr_summary.LangChainClient")
    @patch("src.tools.adr_summary.get_session_documents")
    def test_output_format(self, mock_get_docs, mock_llm_cls):
        doc = MagicMock()
        doc.page_content = "Clinical text."
        doc.metadata = {"source": "report.pdf", "page": 3, "session_id": "s1"}
        mock_get_docs.return_value = [doc]

        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Summarized findings."
        mock_llm.invoke.return_value = mock_response
        mock_llm_cls.return_value = MagicMock(client=mock_llm)

        tool = AdrSummaryTool()
        result = tool._run(session_id="s1")

        assert "--- report.pdf | Page 3 ---" in result
        assert "Summarized findings." in result

    @patch("src.tools.adr_summary.LangChainClient")
    @patch("src.tools.adr_summary.get_session_documents")
    def test_handles_llm_error_gracefully(self, mock_get_docs, mock_llm_cls):
        doc = MagicMock()
        doc.page_content = "Text."
        doc.metadata = {"source": "f.pdf", "page": 1, "session_id": "s1"}
        mock_get_docs.return_value = [doc]

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = RuntimeError("LLM down")
        mock_llm_cls.return_value = MagicMock(client=mock_llm)

        tool = AdrSummaryTool()
        result = tool._run(session_id="s1")

        assert "error" in result.lower()
