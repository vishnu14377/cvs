"""
Unit tests for the LLM OCR client using LangChain.

Tests cover:
- Client initialization
- Document processing workflow
- Message building for GCS files
- Structured output handling
- Error handling (invalid URI, invalid file type, API errors)
- Validation error handling
- Singleton access functions

Run with: pytest tests/unit/test_llm_ocr_client.py -v
"""

from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError
from src.ocr.data_models.llm_response import DocumentExtraction, PageExtraction
from src.ocr.llm_ocr_client import (
    SYSTEM_PROMPT,
    LLMOcrClient,
    get_llm_ocr_client,
    is_pdf_file,
    reset_llm_ocr_client,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset all singletons before and after each test."""
    reset_llm_ocr_client()

    # Reset LangChainClient
    from src.core.langchain_client import LangChainClient

    LangChainClient.reset()

    yield

    reset_llm_ocr_client()
    LangChainClient.reset()


@pytest.fixture
def mock_langchain_client():
    """Create a mock LangChainClient."""
    mock_client = MagicMock()
    mock_client.model_id = "gemini-1.5-flash"
    mock_structured_llm = MagicMock()
    mock_client.with_structured_output.return_value = mock_structured_llm
    return mock_client, mock_structured_llm


@pytest.fixture
def mock_ocr_client(mock_langchain_client):
    """Create an LLMOcrClient with mocked LangChain client."""
    mock_client, mock_structured_llm = mock_langchain_client

    with patch("src.ocr.llm_ocr_client.LangChainClient") as mock_lc_class:
        mock_lc_class.return_value = mock_client
        client = LLMOcrClient()
        yield client, mock_structured_llm


# =============================================================================
# Test: Helper Functions
# =============================================================================


class TestIsPdfFile:
    """Tests for is_pdf_file helper function."""

    def test_pdf_lowercase(self):
        assert is_pdf_file("document.pdf") is True

    def test_pdf_uppercase(self):
        assert is_pdf_file("DOCUMENT.PDF") is True

    def test_pdf_mixed_case(self):
        assert is_pdf_file("Document.Pdf") is True

    def test_not_pdf(self):
        assert is_pdf_file("document.txt") is False
        assert is_pdf_file("image.png") is False

    def test_pdf_in_path(self):
        assert is_pdf_file("/path/to/file.pdf") is True
        assert is_pdf_file("gs://bucket/folder/file.pdf") is True


# =============================================================================
# Test: Client Initialization
# =============================================================================


class TestLLMOcrClientInit:
    """Tests for LLMOcrClient initialization."""

    def test_init_default_parameters(self):
        """Client should initialize with default parameters."""
        with patch("src.ocr.llm_ocr_client.LangChainClient") as mock_lc:
            mock_instance = MagicMock()
            mock_instance.model_id = "gemini-1.5-flash"
            mock_lc.return_value = mock_instance

            client = LLMOcrClient()

            assert client._temperature == 0.1
            assert client._max_output_tokens is None
            mock_instance.with_structured_output.assert_called_once_with(DocumentExtraction)

    def test_init_custom_temperature(self):
        """Client should accept custom temperature."""
        with patch("src.ocr.llm_ocr_client.LangChainClient") as mock_lc:
            mock_instance = MagicMock()
            mock_instance.model_id = "gemini-1.5-flash"
            mock_lc.return_value = mock_instance

            client = LLMOcrClient(temperature=0.5)

            assert client._temperature == 0.5

    def test_init_custom_max_tokens(self):
        """Client should accept custom max_output_tokens."""
        with patch("src.ocr.llm_ocr_client.LangChainClient") as mock_lc:
            mock_instance = MagicMock()
            mock_instance.model_id = "gemini-1.5-flash"
            mock_lc.return_value = mock_instance

            client = LLMOcrClient(max_output_tokens=4096)

            assert client._max_output_tokens == 4096


# =============================================================================
# Test: process_document - Success Cases
# =============================================================================


class TestProcessDocumentSuccess:
    """Tests for successful document processing."""

    def test_process_valid_gcs_pdf(self, mock_ocr_client):
        """Should successfully process a valid GCS PDF."""
        client, mock_structured_llm = mock_ocr_client

        # Mock successful extraction
        mock_extraction = DocumentExtraction(
            pages=[
                PageExtraction(page_number=1, page_text="Page 1 content", page_summary="Summary 1", page_insight="Insight 1"),
                PageExtraction(page_number=2, page_text="Page 2 content", page_summary="Summary 2", page_insight="Insight 2"),
            ]
        )
        mock_structured_llm.invoke.return_value = mock_extraction

        result = client.process_document("gs://bucket/path/document.pdf")

        assert result["success"] is True
        assert len(result["pages"]) == 2
        assert result["pages"][0]["index"] == 1
        assert result["pages"][1]["index"] == 2
        assert result["error"] is None

    def test_process_with_custom_prompt(self, mock_ocr_client):
        """Should use custom prompt when provided."""
        client, mock_structured_llm = mock_ocr_client

        mock_extraction = DocumentExtraction(pages=[])
        mock_structured_llm.invoke.return_value = mock_extraction

        result = client.process_document("gs://bucket/doc.pdf", prompt="Extract only tables")

        # Check that invoke was called (we can't easily verify the prompt contents)
        mock_structured_llm.invoke.assert_called_once()
        assert result["success"] is True


# =============================================================================
# Test: process_document - Validation Errors
# =============================================================================


class TestProcessDocumentValidation:
    """Tests for input validation in process_document."""

    def test_invalid_gcs_uri_rejects_local_path(self, mock_ocr_client):
        """Should reject local file paths."""
        client, _ = mock_ocr_client

        result = client.process_document("/local/path/document.pdf")

        assert result["success"] is False
        assert "Invalid GCS URI" in result["error"]
        assert result["pages"] == []

    def test_invalid_gcs_uri_rejects_http(self, mock_ocr_client):
        """Should reject HTTP URLs."""
        client, _ = mock_ocr_client

        result = client.process_document("https://example.com/document.pdf")

        assert result["success"] is False
        assert "Invalid GCS URI" in result["error"]

    def test_invalid_file_type_rejects_non_pdf(self, mock_ocr_client):
        """Should reject non-PDF files."""
        client, _ = mock_ocr_client

        result = client.process_document("gs://bucket/document.txt")

        assert result["success"] is False
        assert "Invalid file type" in result["error"]
        assert "Only PDF files are supported" in result["error"]


# =============================================================================
# Test: process_document - Error Handling
# =============================================================================


class TestProcessDocumentErrors:
    """Tests for error handling in process_document."""

    def test_handles_validation_error(self, mock_ocr_client):
        """Should handle Pydantic validation errors."""
        client, mock_structured_llm = mock_ocr_client

        mock_structured_llm.invoke.side_effect = ValidationError.from_exception_data(
            title="DocumentExtraction",
            line_errors=[],
        )

        result = client.process_document("gs://bucket/doc.pdf")

        assert result["success"] is False
        assert (
            "validation failed" in result["error"].lower()
            or "0 validation errors" in result["error"].lower()
        )

    def test_handles_api_exception(self, mock_ocr_client):
        """Should handle general API exceptions."""
        client, mock_structured_llm = mock_ocr_client

        mock_structured_llm.invoke.side_effect = Exception("API timeout")

        result = client.process_document("gs://bucket/doc.pdf")

        assert result["success"] is False
        assert "Gemini prediction failed" in result["error"]
        assert "API timeout" in result["error"]


# =============================================================================
# Test: Message Building
# =============================================================================


class TestBuildMessages:
    """Tests for _build_messages method."""

    def test_builds_messages_with_gcs_uri(self, mock_ocr_client):
        """Should build correct message structure for GCS PDF."""
        client, _ = mock_ocr_client

        messages = client._build_messages("gs://bucket/document.pdf")

        # Should have system message and human message
        assert len(messages) == 2

        # Check human message structure
        human_msg = messages[1]
        content = human_msg.content
        assert len(content) == 2

        # Check media part
        media_part = content[0]
        assert media_part["type"] == "media"
        assert media_part["file_uri"] == "gs://bucket/document.pdf"
        assert media_part["mime_type"] == "application/pdf"

        # Check text part
        text_part = content[1]
        assert text_part["type"] == "text"

    def test_builds_messages_with_custom_prompt(self, mock_ocr_client):
        """Should use custom prompt in message."""
        client, _ = mock_ocr_client

        messages = client._build_messages("gs://bucket/doc.pdf", prompt="Extract only headers")

        human_msg = messages[1]
        text_part = human_msg.content[1]
        assert text_part["text"] == "Extract only headers"

    def test_system_prompt_is_used(self, mock_ocr_client):
        """Should include system prompt in messages."""
        client, _ = mock_ocr_client

        messages = client._build_messages("gs://bucket/doc.pdf")

        system_msg = messages[0]
        assert system_msg.content == SYSTEM_PROMPT


# =============================================================================
# Test: Structured LLM Configuration
# =============================================================================


class TestGetStructuredLLM:
    """Tests for _get_structured_llm method."""

    def test_returns_default_when_no_overrides(self, mock_ocr_client):
        """Should return pre-configured LLM when no overrides."""
        client, mock_structured_llm = mock_ocr_client

        result = client._get_structured_llm()

        assert result is mock_structured_llm

    def test_creates_new_when_temperature_override(self, mock_ocr_client):
        """Should create new LLM when temperature is overridden."""
        client, mock_structured_llm = mock_ocr_client

        # Setup the mock chain
        mock_bound = MagicMock()
        mock_new_structured = MagicMock()
        client._langchain.client.bind.return_value = mock_bound
        mock_bound.with_structured_output.return_value = mock_new_structured

        result = client._get_structured_llm(temperature=0.8)

        client._langchain.client.bind.assert_called_once_with(temperature=0.8)
        mock_bound.with_structured_output.assert_called_once_with(DocumentExtraction)
        assert result is mock_new_structured

    def test_creates_new_when_max_tokens_override(self, mock_ocr_client):
        """Should create new LLM when max_output_tokens is overridden."""
        client, _ = mock_ocr_client

        mock_bound = MagicMock()
        mock_new_structured = MagicMock()
        client._langchain.client.bind.return_value = mock_bound
        mock_bound.with_structured_output.return_value = mock_new_structured

        client._get_structured_llm(max_output_tokens=8192)

        client._langchain.client.bind.assert_called_once_with(max_output_tokens=8192)


# =============================================================================
# Test: Singleton Access Functions
# =============================================================================


class TestSingletonFunctions:
    """Tests for singleton access functions."""

    def test_get_llm_ocr_client_returns_same_instance(self):
        """get_llm_ocr_client should return the same instance."""
        with patch("src.ocr.llm_ocr_client.LangChainClient") as mock_lc:
            mock_instance = MagicMock()
            mock_instance.model_id = "gemini-1.5-flash"
            mock_lc.return_value = mock_instance

            client1 = get_llm_ocr_client()
            client2 = get_llm_ocr_client()

            assert client1 is client2

    def test_reset_llm_ocr_client_clears_singleton(self):
        """reset_llm_ocr_client should clear the singleton."""
        with patch("src.ocr.llm_ocr_client.LangChainClient") as mock_lc:
            mock_instance = MagicMock()
            mock_instance.model_id = "gemini-1.5-flash"
            mock_lc.return_value = mock_instance

            client1 = get_llm_ocr_client()
            reset_llm_ocr_client()
            client2 = get_llm_ocr_client()

            # After reset, should be a new instance
            assert client1 is not client2

    def test_get_llm_ocr_client_uses_parameters_on_first_call(self):
        """Parameters should be used when creating the singleton."""
        with patch("src.ocr.llm_ocr_client.LangChainClient") as mock_lc:
            mock_instance = MagicMock()
            mock_instance.model_id = "gemini-1.5-flash"
            mock_lc.return_value = mock_instance

            client = get_llm_ocr_client(temperature=0.7, max_output_tokens=2048)

            assert client._temperature == 0.7
            assert client._max_output_tokens == 2048
