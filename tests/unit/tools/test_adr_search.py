"""
Unit tests for the ADR Search Tool.

Tests cover:
- ADRSearchTool functionality
- Input validation
- Error handling
- Factory functions
"""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document
from pydantic import ValidationError
from src.tools.adr_search import (
    ADRSearchInput,
    ADRSearchTool,
    adr_hybrid_search_tool,
    adr_search_tool,
    get_adr_search_tool,
)


@pytest.fixture
def mock_retriever():
    """Create a mock retriever."""
    retriever = MagicMock()
    retriever.invoke.return_value = [
        Document(
            page_content="Patient diagnosed with hypertension.",
            metadata={"source": "medical_record.pdf", "page": 1, "session_id": "test-session"},
        ),
        Document(
            page_content="Treatment includes blood pressure medication.",
            metadata={"source": "treatment_plan.pdf", "page": 2, "session_id": "test-session"},
        ),
    ]
    return retriever


@pytest.fixture
def mock_empty_retriever():
    """Create a mock retriever that returns no results."""
    retriever = MagicMock()
    retriever.invoke.return_value = []
    return retriever


@pytest.fixture
def mock_async_retriever():
    """Create a mock retriever for async tests.

    ``_arun`` now calls ``retriever.invoke`` in a thread-pool executor
    (because the PGVector store has no async engine), so we mock
    ``.invoke`` rather than ``.ainvoke``.
    """
    retriever = MagicMock()
    retriever.invoke.return_value = [
        Document(
            page_content="Patient diagnosed with hypertension.",
            metadata={"source": "medical_record.pdf", "page": 1, "session_id": "test-session"},
        ),
        Document(
            page_content="Treatment includes blood pressure medication.",
            metadata={"source": "treatment_plan.pdf", "page": 2, "session_id": "test-session"},
        ),
    ]
    return retriever


@pytest.fixture
def mock_async_empty_retriever():
    """Create a mock retriever for async tests that returns no results."""
    retriever = MagicMock()
    retriever.invoke.return_value = []
    return retriever


class TestADRSearchInput:
    """Tests for the ADRSearchInput schema."""

    def test_valid_input(self):
        """Test valid input creation with query and session_id."""
        input_data = ADRSearchInput(
            query="What is the diagnosis?",
            session_id="session-123",
        )
        assert input_data.query == "What is the diagnosis?"
        assert input_data.session_id == "session-123"

    def test_missing_query_raises(self):
        """Test that missing query raises validation error."""
        with pytest.raises(ValidationError):
            ADRSearchInput(session_id="session-123")

    def test_session_id_not_in_tool_schema(self):
        """Test that session_id is excluded from the LLM-facing tool schema.

        session_id is hidden via model_json_schema override so the LLM never
        sees it, but the tool node can still inject it at runtime.
        """
        tool = ADRSearchTool()
        # args_schema.model_json_schema() is what bind_tools sends to the LLM
        schema = tool.args_schema.model_json_schema()
        assert "session_id" not in schema.get("properties", {})
        assert "session_id" not in schema.get("required", [])


class TestADRSearchTool:
    """Tests for the ADRSearchTool."""

    def test_tool_properties(self):
        """Test tool has correct properties."""
        tool = ADRSearchTool()

        assert tool.name == "adr_search"
        assert "ADR" in tool.description
        assert tool.args_schema == ADRSearchInput

    def test_default_configuration(self):
        """Test default tool configuration."""
        tool = ADRSearchTool()

        assert tool.use_hybrid is False
        assert tool.bm25_weight == 0.5
        assert tool.semantic_weight == 0.5

    def test_hybrid_configuration(self):
        """Test hybrid tool configuration."""
        tool = ADRSearchTool(use_hybrid=True, bm25_weight=0.7, semantic_weight=0.3)

        assert tool.use_hybrid is True
        assert tool.bm25_weight == 0.7
        assert tool.semantic_weight == 0.3

    @patch("src.tools.adr_search.get_session_retriever")
    def test_run_semantic_search(self, mock_get_retriever, mock_retriever):
        """Test semantic search tool execution."""
        mock_get_retriever.return_value = mock_retriever

        tool = ADRSearchTool(use_hybrid=False)
        result = tool._run(query="diagnosis", session_id="test-session")

        assert "Found 2 relevant document(s)" in result
        assert "hypertension" in result
        assert "blood pressure medication" in result
        mock_get_retriever.assert_called_once()
        mock_retriever.invoke.assert_called_once_with("diagnosis")

    @patch("src.tools.adr_search.get_hybrid_retriever")
    def test_run_hybrid_search(self, mock_get_hybrid, mock_retriever):
        """Test hybrid search tool execution."""
        mock_get_hybrid.return_value = mock_retriever

        tool = ADRSearchTool(use_hybrid=True)
        result = tool._run(query="MRN 12345", session_id="test-session")

        assert "Found 2 relevant document(s)" in result
        mock_get_hybrid.assert_called_once()
        mock_retriever.invoke.assert_called_once_with("MRN 12345")

    @patch("src.tools.adr_search.get_session_retriever")
    def test_run_no_results(self, mock_get_retriever, mock_empty_retriever):
        """Test tool execution with no results."""
        mock_get_retriever.return_value = mock_empty_retriever

        tool = ADRSearchTool()
        result = tool._run(query="unknown query", session_id="test-session")

        assert "No relevant documents found" in result

    @patch("src.tools.adr_search.get_session_retriever")
    def test_run_error_handling(self, mock_get_retriever):
        """Test tool handles errors gracefully."""
        mock_get_retriever.side_effect = Exception("Database connection failed")

        tool = ADRSearchTool()
        result = tool._run(query="diagnosis", session_id="test-session")

        assert "Error searching ADR documents" in result
        assert "Database connection failed" in result

    @patch("src.tools.adr_search.get_session_retriever")
    def test_invoke_method(self, mock_get_retriever, mock_retriever):
        """Test tool invocation via invoke method."""
        mock_get_retriever.return_value = mock_retriever

        tool = ADRSearchTool()
        result = tool.invoke({"query": "diagnosis", "session_id": "test-session"})

        assert "Found 2 relevant document(s)" in result

    @patch("src.tools.adr_search.get_session_retriever")
    def test_format_results_with_metadata(self, mock_get_retriever):
        """Test result formatting includes metadata."""
        retriever = MagicMock()
        retriever.invoke.return_value = [
            Document(
                page_content="Test content",
                metadata={"source": "test.pdf", "page": 5},
            ),
        ]
        mock_get_retriever.return_value = retriever

        tool = ADRSearchTool()
        result = tool._run(query="test", session_id="test-session")

        assert "Source: test.pdf" in result
        assert "Page: 5" in result
        assert "Test content" in result


class TestADRSearchToolAsync:
    """Tests for the async _arun method."""

    @patch("src.tools.adr_search.get_session_retriever")
    @pytest.mark.asyncio
    async def test_arun_semantic_search(self, mock_get_retriever, mock_async_retriever):
        """Test async semantic search delegates to sync invoke in executor."""
        mock_get_retriever.return_value = mock_async_retriever

        tool = ADRSearchTool(use_hybrid=False)
        result = await tool._arun(query="diagnosis", session_id="test-session")

        assert "Found 2 relevant document(s)" in result
        assert "hypertension" in result
        mock_async_retriever.invoke.assert_called_once_with("diagnosis")

    @patch("src.tools.adr_search.get_hybrid_retriever")
    @pytest.mark.asyncio
    async def test_arun_hybrid_search(self, mock_get_hybrid, mock_async_retriever):
        """Test async hybrid search delegates to sync invoke in executor."""
        mock_get_hybrid.return_value = mock_async_retriever

        tool = ADRSearchTool(use_hybrid=True)
        result = await tool._arun(query="MRN 12345", session_id="test-session")

        assert "Found 2 relevant document(s)" in result
        mock_get_hybrid.assert_called_once()
        mock_async_retriever.invoke.assert_called_once_with("MRN 12345")

    @patch("src.tools.adr_search.get_session_retriever")
    @pytest.mark.asyncio
    async def test_arun_no_results(self, mock_get_retriever, mock_async_empty_retriever):
        """Test async search with no results."""
        mock_get_retriever.return_value = mock_async_empty_retriever

        tool = ADRSearchTool()
        result = await tool._arun(query="unknown", session_id="test-session")

        assert "No relevant documents found" in result

    @patch("src.tools.adr_search.get_session_retriever")
    @pytest.mark.asyncio
    async def test_arun_error_handling(self, mock_get_retriever):
        """Test async search handles errors gracefully."""
        mock_get_retriever.side_effect = Exception("Connection timeout")

        tool = ADRSearchTool()
        result = await tool._arun(query="diagnosis", session_id="test-session")

        assert "Error searching ADR documents" in result
        assert "Connection timeout" in result

    @patch("src.tools.adr_search.get_session_retriever")
    @pytest.mark.asyncio
    async def test_arun_invoke_exception(self, mock_get_retriever):
        """Test async search handles invoke exceptions gracefully."""
        retriever = MagicMock()
        retriever.invoke.side_effect = Exception("Vector store unavailable")
        mock_get_retriever.return_value = retriever

        tool = ADRSearchTool()
        result = await tool._arun(query="diagnosis", session_id="test-session")

        assert "Error searching ADR documents" in result
        assert "Vector store unavailable" in result


class TestGetADRSearchTool:
    """Tests for the get_adr_search_tool factory function."""

    def test_default_configuration(self):
        """Test factory with default configuration."""
        tool = get_adr_search_tool()

        assert isinstance(tool, ADRSearchTool)
        assert tool.name == "adr_search"
        assert tool.use_hybrid is False

    def test_hybrid_configuration(self):
        """Test factory with hybrid configuration."""
        tool = get_adr_search_tool(use_hybrid=True)

        assert tool.use_hybrid is True

    def test_custom_weights(self):
        """Test factory with custom BM25/semantic weights."""
        tool = get_adr_search_tool(
            use_hybrid=True,
            bm25_weight=0.7,
            semantic_weight=0.3,
        )

        assert tool.bm25_weight == 0.7
        assert tool.semantic_weight == 0.3

    def test_custom_search_parameters(self):
        """Test factory with custom search parameters."""
        tool = get_adr_search_tool(
            search_type="mmr",
            k=10,
            score_threshold=0.8,
        )

        assert tool.search_type == "mmr"
        assert tool.k == 10
        assert tool.score_threshold == 0.8


class TestPreConfiguredTools:
    """Tests for pre-configured tool instances."""

    def test_semantic_tool_exists(self):
        """Test that pre-configured semantic tool exists."""
        assert adr_search_tool is not None
        assert isinstance(adr_search_tool, ADRSearchTool)
        assert adr_search_tool.use_hybrid is False

    def test_hybrid_tool_exists(self):
        """Test that pre-configured hybrid tool exists."""
        assert adr_hybrid_search_tool is not None
        assert isinstance(adr_hybrid_search_tool, ADRSearchTool)
        assert adr_hybrid_search_tool.use_hybrid is True


class TestModuleExports:
    """Tests for module-level exports."""

    def test_imports_from_package(self):
        """Test imports work from the tools package."""
        from src.tools import (
            ADRSearchInput,
            ADRSearchTool,
            adr_hybrid_search_tool,
            adr_search_tool,
            get_adr_search_tool,
        )

        assert ADRSearchTool is not None
        assert ADRSearchInput is not None
        assert get_adr_search_tool is not None
        assert adr_search_tool is not None
        assert adr_hybrid_search_tool is not None
