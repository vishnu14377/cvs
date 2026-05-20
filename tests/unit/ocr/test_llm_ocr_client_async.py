"""
Unit tests for LlmOcrClientAsync.

Tests async LLM OCR client behavior matching the sync version but with:
- Concurrent document processing
- Semaphore limits
- Async operations
- LangChain structured output
"""

from unittest.mock import MagicMock, patch

import pytest
from src.ocr.data_models.llm_response import DocumentExtraction, PageExtraction
from src.ocr.llm_ocr_client_async import LlmOcrClientAsync


@pytest.fixture
def sample_pdf_uri():
    """Sample GCS PDF URI."""
    return "gs://test-bucket/documents/sample.pdf"


@pytest.fixture
def sample_extraction():
    """Sample DocumentExtraction Pydantic object."""
    return DocumentExtraction(
        pages=[
            PageExtraction(index=1, extracted_text="Page 1 content"),
            PageExtraction(index=2, extracted_text="Page 2 content"),
        ]
    )


@pytest.fixture
def mock_langchain_client():
    """Mock LangChain client."""
    mock_client = MagicMock()
    mock_client.model_id = "gemini-1.5-pro-002"
    mock_client.client = MagicMock()

    # Mock with_structured_output to return a mock LLM
    mock_structured_llm = MagicMock()
    mock_client.with_structured_output.return_value = mock_structured_llm

    return mock_client


class TestLlmOcrClientAsyncInit:
    """Test client initialization."""

    @patch("src.ocr.llm_ocr_client_async.LangChainClient")
    def test_init_default_config(self, mock_langchain_class):
        """Test initialization with default configuration."""
        mock_langchain = MagicMock()
        mock_langchain.model_id = "gemini-1.5-pro-002"
        mock_langchain.with_structured_output.return_value = MagicMock()
        mock_langchain_class.return_value = mock_langchain

        client = LlmOcrClientAsync()

        assert client._temperature == 0.1
        assert client._langchain is not None
        assert client._structured_llm is not None
        assert client._max_concurrent == 5
        assert client._executor is not None

    @patch("src.ocr.llm_ocr_client_async.LangChainClient")
    def test_init_custom_concurrent_limit(self, mock_langchain_class):
        """Test initialization with custom concurrency limit."""
        mock_langchain = MagicMock()
        mock_langchain.model_id = "gemini-1.5-pro-002"
        mock_langchain.with_structured_output.return_value = MagicMock()
        mock_langchain_class.return_value = mock_langchain

        client = LlmOcrClientAsync(max_concurrent_requests=10)

        assert client._max_concurrent == 10

    @patch("src.ocr.llm_ocr_client_async.LangChainClient")
    def test_init_custom_temperature(self, mock_langchain_class):
        """Test initialization with custom temperature."""
        mock_langchain = MagicMock()
        mock_langchain.model_id = "gemini-1.5-pro-002"
        mock_langchain.with_structured_output.return_value = MagicMock()
        mock_langchain_class.return_value = mock_langchain

        client = LlmOcrClientAsync(temperature=0.5)

        assert client._temperature == 0.5


class TestLlmOcrClientAsyncProcessDocument:
    """Test single document processing."""

    @pytest.mark.asyncio
    @patch("src.ocr.llm_ocr_client_async.LangChainClient")
    async def test_process_document_success(
        self, mock_langchain_class, sample_pdf_uri, sample_extraction
    ):
        """Test successful PDF processing."""
        # Setup mock
        mock_langchain = MagicMock()
        mock_langchain.model_id = "gemini-1.5-pro-002"
        mock_structured_llm = MagicMock()
        mock_structured_llm.invoke.return_value = sample_extraction
        mock_langchain.with_structured_output.return_value = mock_structured_llm
        mock_langchain_class.return_value = mock_langchain

        client = LlmOcrClientAsync(max_concurrent_requests=1)

        result = await client.process_document(sample_pdf_uri)

        assert result["success"] is True
        assert len(result["pages"]) == 2
        assert result["pages"][0]["index"] == 1
        assert result["pages"][0]["extracted_text"] == "Page 1 content"
        assert result["pages"][1]["index"] == 2
        assert result["pages"][1]["extracted_text"] == "Page 2 content"
        assert result["error"] is None

    @pytest.mark.asyncio
    @patch("src.ocr.llm_ocr_client_async.LangChainClient")
    async def test_process_document_invalid_gcs_uri(self, mock_langchain_class):
        """Test processing with invalid GCS URI."""
        mock_langchain = MagicMock()
        mock_langchain.model_id = "gemini-1.5-pro-002"
        mock_langchain.with_structured_output.return_value = MagicMock()
        mock_langchain_class.return_value = mock_langchain

        client = LlmOcrClientAsync()

        result = await client.process_document("not-a-gcs-uri.pdf")

        assert result["success"] is False
        assert "Invalid GCS URI" in result["error"]
        assert result["pages"] == []

    @pytest.mark.asyncio
    @patch("src.ocr.llm_ocr_client_async.LangChainClient")
    async def test_process_document_invalid_file_type(self, mock_langchain_class):
        """Test processing with non-PDF file."""
        mock_langchain = MagicMock()
        mock_langchain.model_id = "gemini-1.5-pro-002"
        mock_langchain.with_structured_output.return_value = MagicMock()
        mock_langchain_class.return_value = mock_langchain

        client = LlmOcrClientAsync()

        result = await client.process_document("gs://bucket/file.txt")

        assert result["success"] is False
        assert "Invalid file type" in result["error"]
        assert result["pages"] == []

    @pytest.mark.asyncio
    @patch("src.ocr.llm_ocr_client_async.LangChainClient")
    async def test_process_document_validation_error(self, mock_langchain_class, sample_pdf_uri):
        """Test handling of Pydantic validation errors."""
        from pydantic import ValidationError

        mock_langchain = MagicMock()
        mock_langchain.model_id = "gemini-1.5-pro-002"
        mock_structured_llm = MagicMock()
        mock_structured_llm.invoke.side_effect = ValidationError.from_exception_data("test", [])
        mock_langchain.with_structured_output.return_value = mock_structured_llm
        mock_langchain_class.return_value = mock_langchain

        client = LlmOcrClientAsync()

        result = await client.process_document(sample_pdf_uri)

        assert result["success"] is False
        assert "Response validation failed" in result["error"]

    @pytest.mark.asyncio
    @patch("src.ocr.llm_ocr_client_async.LangChainClient")
    async def test_process_document_api_error(self, mock_langchain_class, sample_pdf_uri):
        """Test handling of API errors."""
        mock_langchain = MagicMock()
        mock_langchain.model_id = "gemini-1.5-pro-002"
        mock_structured_llm = MagicMock()
        mock_structured_llm.invoke.side_effect = Exception("API error")
        mock_langchain.with_structured_output.return_value = mock_structured_llm
        mock_langchain_class.return_value = mock_langchain

        client = LlmOcrClientAsync()

        result = await client.process_document(sample_pdf_uri)

        assert result["success"] is False
        assert "Gemini prediction failed" in result["error"]
        assert "API error" in result["error"]

    @pytest.mark.asyncio
    @patch("src.ocr.llm_ocr_client_async.LangChainClient")
    async def test_process_document_with_custom_prompt(
        self, mock_langchain_class, sample_pdf_uri, sample_extraction
    ):
        """Test processing with custom prompt."""
        mock_langchain = MagicMock()
        mock_langchain.model_id = "gemini-1.5-pro-002"
        mock_structured_llm = MagicMock()
        mock_structured_llm.invoke.return_value = sample_extraction
        mock_langchain.with_structured_output.return_value = mock_structured_llm
        mock_langchain_class.return_value = mock_langchain

        client = LlmOcrClientAsync()
        custom_prompt = "Extract only headers"

        result = await client.process_document(sample_pdf_uri, prompt=custom_prompt)

        assert result["success"] is True
        # Verify invoke was called
        mock_structured_llm.invoke.assert_called_once()
        messages = mock_structured_llm.invoke.call_args[0][0]
        # Check that custom prompt is in the messages
        assert any(custom_prompt in str(msg.content) for msg in messages)

    @pytest.mark.asyncio
    @patch("src.ocr.llm_ocr_client_async.LangChainClient")
    async def test_process_document_with_temperature_override(
        self, mock_langchain_class, sample_pdf_uri, sample_extraction
    ):
        """Test processing with temperature override."""
        mock_langchain = MagicMock()
        mock_langchain.model_id = "gemini-1.5-pro-002"
        mock_llm_base = MagicMock()
        mock_bound_llm = MagicMock()
        mock_structured_llm = MagicMock()

        # Setup chain: client -> bind() -> with_structured_output()
        mock_langchain.client = mock_llm_base
        mock_llm_base.bind.return_value = mock_bound_llm
        mock_bound_llm.with_structured_output.return_value = mock_structured_llm
        mock_structured_llm.invoke.return_value = sample_extraction

        # Default structured LLM for initialization
        mock_langchain.with_structured_output.return_value = MagicMock()
        mock_langchain_class.return_value = mock_langchain

        client = LlmOcrClientAsync()

        result = await client.process_document(sample_pdf_uri, temperature=0.5)

        assert result["success"] is True
        # Verify bind was called with temperature
        mock_llm_base.bind.assert_called_once()
        bind_kwargs = mock_llm_base.bind.call_args.kwargs
        assert bind_kwargs["temperature"] == 0.5


class TestLlmOcrClientAsyncConcurrency:
    """Test concurrent processing."""

    @pytest.mark.asyncio
    @patch("src.ocr.llm_ocr_client_async.LangChainClient")
    async def test_process_multiple_documents_success(
        self, mock_langchain_class, sample_extraction
    ):
        """Test successful concurrent processing."""
        mock_langchain = MagicMock()
        mock_langchain.model_id = "gemini-1.5-pro-002"
        mock_structured_llm = MagicMock()
        mock_structured_llm.invoke.return_value = sample_extraction
        mock_langchain.with_structured_output.return_value = mock_structured_llm
        mock_langchain_class.return_value = mock_langchain

        client = LlmOcrClientAsync(max_concurrent_requests=3)

        pdf_uris = [
            "gs://bucket/doc1.pdf",
            "gs://bucket/doc2.pdf",
            "gs://bucket/doc3.pdf",
        ]

        results = await client.process_multiple_documents(pdf_uris)

        assert len(results) == 3
        assert all(r["success"] for r in results)
        assert all(len(r["pages"]) == 2 for r in results)
        assert mock_structured_llm.invoke.call_count == 3

    @pytest.mark.asyncio
    @patch("src.ocr.llm_ocr_client_async.LangChainClient")
    async def test_semaphore_limits_concurrent_requests(
        self, mock_langchain_class, sample_extraction
    ):
        """Test that semaphore limits concurrent API calls."""
        max_concurrent = 2
        concurrent_tracker = {"current": 0, "max_reached": 0}

        def mock_invoke(*args, **kwargs):
            concurrent_tracker["current"] += 1
            concurrent_tracker["max_reached"] = max(
                concurrent_tracker["max_reached"], concurrent_tracker["current"]
            )
            import time

            time.sleep(0.05)
            concurrent_tracker["current"] -= 1
            return sample_extraction

        mock_langchain = MagicMock()
        mock_langchain.model_id = "gemini-1.5-pro-002"
        mock_structured_llm = MagicMock()
        mock_structured_llm.invoke.side_effect = mock_invoke
        mock_langchain.with_structured_output.return_value = mock_structured_llm
        mock_langchain_class.return_value = mock_langchain

        client = LlmOcrClientAsync(max_concurrent_requests=max_concurrent)

        pdf_uris = [f"gs://bucket/doc{i}.pdf" for i in range(5)]

        results = await client.process_multiple_documents(pdf_uris)

        assert len(results) == 5
        assert all(r["success"] for r in results)
        assert concurrent_tracker["max_reached"] <= max_concurrent

    @pytest.mark.asyncio
    @patch("src.ocr.llm_ocr_client_async.LangChainClient")
    async def test_process_multiple_documents_partial_failure(
        self, mock_langchain_class, sample_extraction
    ):
        """Test batch processing with some failures."""
        call_count = {"value": 0}

        def mock_invoke(*args, **kwargs):
            call_count["value"] += 1
            if call_count["value"] == 2:
                raise Exception("API error on second call")
            return sample_extraction

        mock_langchain = MagicMock()
        mock_langchain.model_id = "gemini-1.5-pro-002"
        mock_structured_llm = MagicMock()
        mock_structured_llm.invoke.side_effect = mock_invoke
        mock_langchain.with_structured_output.return_value = mock_structured_llm
        mock_langchain_class.return_value = mock_langchain

        client = LlmOcrClientAsync()

        pdf_uris = ["gs://bucket/doc1.pdf", "gs://bucket/doc2.pdf", "gs://bucket/doc3.pdf"]

        results = await client.process_multiple_documents(pdf_uris)

        assert len(results) == 3
        assert results[0]["success"] is True
        assert results[1]["success"] is False
        assert "API error" in results[1]["error"]
        assert results[2]["success"] is True

    @pytest.mark.asyncio
    @patch("src.ocr.llm_ocr_client_async.LangChainClient")
    async def test_process_multiple_documents_all_failures(self, mock_langchain_class):
        """Test batch processing with all failures."""
        mock_langchain = MagicMock()
        mock_langchain.model_id = "gemini-1.5-pro-002"
        mock_structured_llm = MagicMock()
        mock_structured_llm.invoke.side_effect = Exception("API down")
        mock_langchain.with_structured_output.return_value = mock_structured_llm
        mock_langchain_class.return_value = mock_langchain

        client = LlmOcrClientAsync()

        pdf_uris = ["gs://bucket/doc1.pdf", "gs://bucket/doc2.pdf"]

        results = await client.process_multiple_documents(pdf_uris)

        assert len(results) == 2
        assert all(not r["success"] for r in results)
        assert all("API down" in r["error"] for r in results)


class TestLlmOcrClientAsyncHelpers:
    """Test helper methods."""

    @patch("src.ocr.llm_ocr_client_async.LangChainClient")
    def test_build_messages_default_prompt(self, mock_langchain_class):
        """Test message building with default prompt."""
        mock_langchain = MagicMock()
        mock_langchain.model_id = "gemini-1.5-pro-002"
        mock_langchain.with_structured_output.return_value = MagicMock()
        mock_langchain_class.return_value = mock_langchain

        client = LlmOcrClientAsync()

        gcs_uri = "gs://bucket/test.pdf"
        messages = client._build_messages(gcs_uri)

        assert len(messages) == 2
        # System message
        assert messages[0].content == client._system_prompt
        # Human message with media and text
        assert isinstance(messages[1].content, list)
        assert messages[1].content[0]["type"] == "media"
        assert messages[1].content[0]["file_uri"] == gcs_uri
        assert messages[1].content[1]["type"] == "text"

    @patch("src.ocr.llm_ocr_client_async.LangChainClient")
    def test_build_messages_custom_prompt(self, mock_langchain_class):
        """Test message building with custom prompt."""
        mock_langchain = MagicMock()
        mock_langchain.model_id = "gemini-1.5-pro-002"
        mock_langchain.with_structured_output.return_value = MagicMock()
        mock_langchain_class.return_value = mock_langchain

        client = LlmOcrClientAsync()

        gcs_uri = "gs://bucket/test.pdf"
        custom_prompt = "Extract only headers"
        messages = client._build_messages(gcs_uri, prompt=custom_prompt)

        assert len(messages) == 2
        assert messages[1].content[1]["text"] == custom_prompt

    @patch("src.ocr.llm_ocr_client_async.LangChainClient")
    def test_get_structured_llm_default(self, mock_langchain_class):
        """Test getting default structured LLM."""
        mock_langchain = MagicMock()
        mock_langchain.model_id = "gemini-1.5-pro-002"
        mock_structured_llm = MagicMock()
        mock_langchain.with_structured_output.return_value = mock_structured_llm
        mock_langchain_class.return_value = mock_langchain

        client = LlmOcrClientAsync()

        llm = client._get_structured_llm()

        assert llm == client._structured_llm

    @patch("src.ocr.llm_ocr_client_async.LangChainClient")
    def test_get_structured_llm_with_overrides(self, mock_langchain_class):
        """Test getting structured LLM with parameter overrides."""
        mock_langchain = MagicMock()
        mock_langchain.model_id = "gemini-1.5-pro-002"
        mock_llm_base = MagicMock()
        mock_bound_llm = MagicMock()

        mock_langchain.client = mock_llm_base
        mock_llm_base.bind.return_value = mock_bound_llm
        mock_bound_llm.with_structured_output.return_value = MagicMock()

        # Default for init
        mock_langchain.with_structured_output.return_value = MagicMock()
        mock_langchain_class.return_value = mock_langchain

        client = LlmOcrClientAsync()

        client._get_structured_llm(temperature=0.5, max_output_tokens=4096, timeout=60.0)

        mock_llm_base.bind.assert_called_once_with(
            temperature=0.5, max_output_tokens=4096, timeout=60.0
        )
        mock_bound_llm.with_structured_output.assert_called_once_with(DocumentExtraction)


class TestLlmOcrClientAsyncSingleton:
    """Test singleton pattern."""

    @patch("src.ocr.llm_ocr_client_async.LangChainClient")
    def test_get_llm_ocr_client_async_creates_singleton(self, mock_langchain_class):
        """Test that get_llm_ocr_client_async creates singleton."""
        from src.ocr.llm_ocr_client_async import (
            get_llm_ocr_client_async,
            reset_llm_ocr_client_async,
        )

        mock_langchain = MagicMock()
        mock_langchain.model_id = "gemini-1.5-pro-002"
        mock_langchain.with_structured_output.return_value = MagicMock()
        mock_langchain_class.return_value = mock_langchain

        # Reset first
        reset_llm_ocr_client_async()

        # Get client twice
        client1 = get_llm_ocr_client_async()
        client2 = get_llm_ocr_client_async()

        # Should be same instance
        assert client1 is client2

        # Cleanup
        reset_llm_ocr_client_async()

    @patch("src.ocr.llm_ocr_client_async.LangChainClient")
    def test_reset_llm_ocr_client_async(self, mock_langchain_class):
        """Test singleton reset."""
        from src.ocr.llm_ocr_client_async import (
            get_llm_ocr_client_async,
            reset_llm_ocr_client_async,
        )

        mock_langchain = MagicMock()
        mock_langchain.model_id = "gemini-1.5-pro-002"
        mock_langchain.with_structured_output.return_value = MagicMock()
        mock_langchain_class.return_value = mock_langchain

        # Get client
        client1 = get_llm_ocr_client_async()

        # Reset
        reset_llm_ocr_client_async()

        # Get again - should be new instance
        client2 = get_llm_ocr_client_async()

        assert client1 is not client2

        # Cleanup
        reset_llm_ocr_client_async()


class TestLlmOcrClientAsyncAlias:
    """Test LlmHandlerAsync alias."""

    @patch("src.ocr.llm_ocr_client_async.LangChainClient")
    def test_llm_handler_async_alias(self, mock_langchain_class):
        """Test that LlmHandlerAsync is an alias for LlmOcrClientAsync."""
        from src.ocr.llm_ocr_client_async import LlmHandlerAsync, LlmOcrClientAsync

        mock_langchain = MagicMock()
        mock_langchain.model_id = "gemini-1.5-pro-002"
        mock_langchain.with_structured_output.return_value = MagicMock()
        mock_langchain_class.return_value = mock_langchain

        assert LlmHandlerAsync is LlmOcrClientAsync

        # Can instantiate via alias
        handler = LlmHandlerAsync()
        assert isinstance(handler, LlmOcrClientAsync)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
