"""
Unit tests for MistralOcrClientAsync.

Tests async API client behavior including:
- Concurrent request handling
- Semaphore limits
- Error handling in batch operations
- Timeout scenarios
- Mock API responses
"""

import base64
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from src.ocr.mistral_ocr_client_async import MistralOcrClientAsync


@pytest.fixture
def mock_vertex_client():
    """Mock Vertex AI client."""
    client = MagicMock()
    client.generate = Mock(
        return_value={
            "candidates": [{"content": "mock OCR response"}],
            "usage_metadata": {"total_token_count": 100},
        }
    )
    return client


@pytest.fixture
def sample_pdf_bytes():
    """Sample PDF bytes for testing."""
    return b"%PDF-1.4\n%mock pdf content"


@pytest.fixture
def sample_pdf_base64(sample_pdf_bytes):
    """Base64 encoded PDF."""
    return base64.b64encode(sample_pdf_bytes).decode("utf-8")


@pytest.fixture
def expected_response():
    """Expected OCR response."""
    return {
        "candidates": [{"content": "Extracted text from PDF"}],
        "usage_metadata": {"total_token_count": 150},
    }


class AsyncContextManagerMock:
    """Helper class to mock async context managers like aiofiles.open()"""

    def __init__(self, return_value):
        self.return_value = return_value

    async def __aenter__(self):
        return self.return_value

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


class TestMistralOcrClientAsyncInit:
    """Test client initialization."""

    @patch("src.ocr.mistral_ocr_client_async.get_vertex_ai_client")
    def test_init_default_config(self, mock_get_client):
        """Test initialization with default configuration."""
        mock_get_client.return_value = MagicMock()
        client = MistralOcrClientAsync()

        assert client.model_id is not None
        assert client.model_publisher is not None
        assert client._vertex_client is not None
        assert client._max_concurrent == 5

    @patch("src.ocr.mistral_ocr_client_async.get_vertex_ai_client")
    def test_init_custom_concurrent_limit(self, mock_get_client):
        """Test initialization with custom concurrency limit."""
        mock_get_client.return_value = MagicMock()
        max_concurrent = 10
        client = MistralOcrClientAsync(max_concurrent_requests=max_concurrent)

        assert client._max_concurrent == max_concurrent

    @patch("src.ocr.mistral_ocr_client_async.get_vertex_ai_client")
    def test_init_custom_model_config(self, mock_get_client):
        """Test initialization with custom model configuration."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        model_id = "custom-mistral-model"
        project_id = "test-project"
        region = "us-central1"

        client = MistralOcrClientAsync(model_id=model_id, project_id=project_id, region=region)

        assert client.model_id == model_id
        mock_get_client.assert_called_once_with(project_id=project_id, region=region)


class TestMistralOcrClientAsyncProcessPdf:
    """Test single PDF processing."""

    @pytest.mark.asyncio
    @patch("src.ocr.mistral_ocr_client_async.get_vertex_ai_client")
    @patch("aiofiles.open")
    @patch("os.path.exists")
    async def test_process_pdf_success(
        self, mock_exists, mock_aiofiles_open, mock_get_client, sample_pdf_bytes, expected_response
    ):
        """Test successful PDF processing."""
        # Setup mocks
        mock_exists.return_value = True

        # Create async file mock
        mock_file = AsyncMock()
        mock_file.read = AsyncMock(return_value=sample_pdf_bytes)
        mock_file.write = AsyncMock()

        # Mock aiofiles.open to return async context manager
        mock_aiofiles_open.return_value = AsyncContextManagerMock(mock_file)

        mock_vertex_client = MagicMock()
        mock_vertex_client.generate = Mock(return_value=expected_response)
        mock_get_client.return_value = mock_vertex_client

        # Execute
        client = MistralOcrClientAsync(max_concurrent_requests=1)
        result = await client.process_pdf("test.pdf", save_response=True)

        # Assert
        assert result["success"] is True
        assert result["pdf_path"] == "test.pdf"
        assert result["response"] == expected_response
        assert result["error"] is None
        assert result["output_file"] == "test_mistral_response.json"

        # Verify API was called
        mock_vertex_client.generate.assert_called_once()
        call_kwargs = mock_vertex_client.generate.call_args.kwargs
        assert "payload" in call_kwargs
        assert call_kwargs["payload"]["model"] == client.model_id

    @pytest.mark.asyncio
    @patch("src.ocr.mistral_ocr_client_async.get_vertex_ai_client")
    @patch("os.path.exists")
    async def test_process_pdf_file_not_found(self, mock_exists, mock_get_client):
        """Test processing when PDF file doesn't exist."""
        mock_exists.return_value = False

        client = MistralOcrClientAsync()
        result = await client.process_pdf("nonexistent.pdf")

        assert result["success"] is False
        assert "File not found" in result["error"]
        assert result["response"] is None

    @pytest.mark.asyncio
    @patch("src.ocr.mistral_ocr_client_async.get_vertex_ai_client")
    @patch("aiofiles.open")
    @patch("os.path.exists")
    async def test_process_pdf_read_error(self, mock_exists, mock_aiofiles_open, mock_get_client):
        """Test handling of file read errors."""
        mock_exists.return_value = True
        mock_aiofiles_open.side_effect = OSError("Permission denied")

        client = MistralOcrClientAsync()
        result = await client.process_pdf("test.pdf")

        assert result["success"] is False
        assert "Failed to read PDF" in result["error"]

    @pytest.mark.asyncio
    @patch("src.ocr.mistral_ocr_client_async.get_vertex_ai_client")
    @patch("aiofiles.open")
    @patch("os.path.exists")
    async def test_process_pdf_api_error(
        self, mock_exists, mock_aiofiles_open, mock_get_client, sample_pdf_bytes
    ):
        """Test handling of API call failures."""
        # Setup mocks
        mock_exists.return_value = True

        mock_file = AsyncMock()
        mock_file.read = AsyncMock(return_value=sample_pdf_bytes)
        mock_aiofiles_open.return_value = AsyncContextManagerMock(mock_file)

        mock_vertex_client = MagicMock()
        mock_vertex_client.generate = Mock(side_effect=Exception("API timeout"))
        mock_get_client.return_value = mock_vertex_client

        # Execute
        client = MistralOcrClientAsync()
        result = await client.process_pdf("test.pdf")

        # Assert
        assert result["success"] is False
        assert "Prediction failed" in result["error"]
        assert "API timeout" in result["error"]

    @pytest.mark.asyncio
    @patch("src.ocr.mistral_ocr_client_async.get_vertex_ai_client")
    @patch("aiofiles.open")
    @patch("os.path.exists")
    async def test_process_pdf_without_saving(
        self, mock_exists, mock_aiofiles_open, mock_get_client, sample_pdf_bytes, expected_response
    ):
        """Test PDF processing without saving response."""
        # Setup mocks
        mock_exists.return_value = True

        mock_file = AsyncMock()
        mock_file.read = AsyncMock(return_value=sample_pdf_bytes)
        mock_aiofiles_open.return_value = AsyncContextManagerMock(mock_file)

        mock_vertex_client = MagicMock()
        mock_vertex_client.generate = Mock(return_value=expected_response)
        mock_get_client.return_value = mock_vertex_client

        # Execute
        client = MistralOcrClientAsync()
        result = await client.process_pdf("test.pdf", save_response=False)

        # Assert
        assert result["success"] is True
        assert result["output_file"] is None


class TestMistralOcrClientAsyncConcurrency:
    """Test concurrent processing and semaphore limits."""

    @pytest.mark.asyncio
    @patch("src.ocr.mistral_ocr_client_async.get_vertex_ai_client")
    @patch("aiofiles.open")
    @patch("os.path.exists")
    async def test_semaphore_limits_concurrent_requests(
        self, mock_exists, mock_aiofiles_open, mock_get_client, sample_pdf_bytes
    ):
        """Test that semaphore limits concurrent API calls."""
        max_concurrent = 2
        concurrent_tracker = {"current": 0, "max_reached": 0}

        def mock_generate(*args, **kwargs):
            """Track concurrent executions."""
            concurrent_tracker["current"] += 1
            concurrent_tracker["max_reached"] = max(
                concurrent_tracker["max_reached"], concurrent_tracker["current"]
            )
            import time

            time.sleep(0.05)  # Simulate API delay
            concurrent_tracker["current"] -= 1
            return {"candidates": [{"content": "mock response"}]}

        # Setup mocks
        mock_exists.return_value = True

        mock_file = AsyncMock()
        mock_file.read = AsyncMock(return_value=sample_pdf_bytes)
        mock_aiofiles_open.return_value = AsyncContextManagerMock(mock_file)

        mock_vertex_client = MagicMock()
        mock_vertex_client.generate = Mock(side_effect=mock_generate)
        mock_get_client.return_value = mock_vertex_client

        # Execute with 5 PDFs but max 2 concurrent
        client = MistralOcrClientAsync(max_concurrent_requests=max_concurrent)
        pdf_paths = [f"test{i}.pdf" for i in range(5)]
        results = await client.process_multiple_pdfs(pdf_paths, save_response=False)

        # Assert
        assert len(results) == 5
        assert all(r["success"] for r in results)
        assert concurrent_tracker["max_reached"] <= max_concurrent

    @pytest.mark.asyncio
    @patch("src.ocr.mistral_ocr_client_async.get_vertex_ai_client")
    @patch("aiofiles.open")
    @patch("os.path.exists")
    async def test_process_multiple_pdfs_success(
        self, mock_exists, mock_aiofiles_open, mock_get_client, sample_pdf_bytes
    ):
        """Test successful concurrent processing of multiple PDFs."""
        # Setup mocks
        mock_exists.return_value = True

        mock_file = AsyncMock()
        mock_file.read = AsyncMock(return_value=sample_pdf_bytes)
        mock_aiofiles_open.return_value = AsyncContextManagerMock(mock_file)

        mock_vertex_client = MagicMock()
        mock_vertex_client.generate = Mock(return_value={"candidates": [{"content": "mock"}]})
        mock_get_client.return_value = mock_vertex_client

        # Execute
        client = MistralOcrClientAsync(max_concurrent_requests=3)
        pdf_paths = ["test1.pdf", "test2.pdf", "test3.pdf"]
        results = await client.process_multiple_pdfs(pdf_paths, save_response=False)

        # Assert
        assert len(results) == 3
        assert all(r["success"] for r in results)
        assert results[0]["pdf_path"] == "test1.pdf"
        assert results[1]["pdf_path"] == "test2.pdf"
        assert results[2]["pdf_path"] == "test3.pdf"

    @pytest.mark.asyncio
    @patch("src.ocr.mistral_ocr_client_async.get_vertex_ai_client")
    @patch("aiofiles.open")
    @patch("os.path.exists")
    async def test_process_multiple_pdfs_partial_failure(
        self, mock_exists, mock_aiofiles_open, mock_get_client, sample_pdf_bytes
    ):
        """Test that batch processing continues when some PDFs fail."""

        # Setup mocks
        def exists_side_effect(path):
            return path != "nonexistent.pdf"

        mock_exists.side_effect = exists_side_effect

        mock_file = AsyncMock()
        mock_file.read = AsyncMock(return_value=sample_pdf_bytes)
        mock_aiofiles_open.return_value = AsyncContextManagerMock(mock_file)

        mock_vertex_client = MagicMock()
        mock_vertex_client.generate = Mock(return_value={"candidates": [{"content": "mock"}]})
        mock_get_client.return_value = mock_vertex_client

        # Execute
        client = MistralOcrClientAsync()
        pdf_paths = ["test1.pdf", "nonexistent.pdf", "test3.pdf"]
        results = await client.process_multiple_pdfs(pdf_paths, save_response=False)

        # Assert
        assert len(results) == 3
        assert results[0]["success"] is True
        assert results[1]["success"] is False
        assert "File not found" in results[1]["error"]
        assert results[2]["success"] is True

    @pytest.mark.asyncio
    @patch("src.ocr.mistral_ocr_client_async.get_vertex_ai_client")
    @patch("aiofiles.open")
    @patch("os.path.exists")
    async def test_process_multiple_pdfs_exception_handling(
        self, mock_exists, mock_aiofiles_open, mock_get_client, sample_pdf_bytes
    ):
        """Test exception handling in batch processing."""
        # Setup mocks
        mock_exists.return_value = True

        call_count = {"value": 0}

        def create_mock_file():
            async def read_side_effect():
                call_count["value"] += 1
                if call_count["value"] == 2:
                    raise RuntimeError("Unexpected error")
                return sample_pdf_bytes

            mock_file = AsyncMock()
            mock_file.read = AsyncMock(side_effect=read_side_effect)
            return mock_file

        def aiofiles_open_side_effect(*args, **kwargs):
            return AsyncContextManagerMock(create_mock_file())

        mock_aiofiles_open.side_effect = aiofiles_open_side_effect

        mock_vertex_client = MagicMock()
        mock_vertex_client.generate = Mock(return_value={"candidates": [{"content": "mock"}]})
        mock_get_client.return_value = mock_vertex_client

        # Execute
        client = MistralOcrClientAsync()
        pdf_paths = ["test1.pdf", "test2.pdf", "test3.pdf"]
        results = await client.process_multiple_pdfs(pdf_paths, save_response=False)

        # Assert
        assert len(results) == 3
        assert results[0]["success"] is True
        assert results[1]["success"] is False
        assert "Unexpected error" in results[1]["error"]
        assert results[2]["success"] is True


class TestMistralOcrClientAsyncTimeout:
    """Test timeout scenarios."""

    @pytest.mark.asyncio
    @patch("src.ocr.mistral_ocr_client_async.get_vertex_ai_client")
    @patch("aiofiles.open")
    @patch("os.path.exists")
    async def test_process_pdf_with_custom_timeout(
        self, mock_exists, mock_aiofiles_open, mock_get_client, sample_pdf_bytes
    ):
        """Test that custom timeout is passed to API call."""
        # Setup mocks
        mock_exists.return_value = True

        mock_file = AsyncMock()
        mock_file.read = AsyncMock(return_value=sample_pdf_bytes)
        mock_aiofiles_open.return_value = AsyncContextManagerMock(mock_file)

        mock_vertex_client = MagicMock()
        mock_vertex_client.generate = Mock(return_value={"candidates": [{"content": "mock"}]})
        mock_get_client.return_value = mock_vertex_client

        # Execute
        custom_timeout = 120.0
        client = MistralOcrClientAsync()
        result = await client.process_pdf("test.pdf", timeout=custom_timeout, save_response=False)

        # Assert
        assert result["success"] is True
        call_kwargs = mock_vertex_client.generate.call_args.kwargs
        assert call_kwargs["timeout"] == custom_timeout

    @pytest.mark.asyncio
    @patch("src.ocr.mistral_ocr_client_async.get_vertex_ai_client")
    @patch("aiofiles.open")
    @patch("os.path.exists")
    async def test_process_pdf_timeout_error(
        self, mock_exists, mock_aiofiles_open, mock_get_client, sample_pdf_bytes
    ):
        """Test handling of timeout errors."""
        # Setup mocks
        mock_exists.return_value = True

        mock_file = AsyncMock()
        mock_file.read = AsyncMock(return_value=sample_pdf_bytes)
        mock_aiofiles_open.return_value = AsyncContextManagerMock(mock_file)

        mock_vertex_client = MagicMock()
        mock_vertex_client.generate = Mock(side_effect=TimeoutError("Request timeout"))
        mock_get_client.return_value = mock_vertex_client

        # Execute
        client = MistralOcrClientAsync()
        result = await client.process_pdf("test.pdf", timeout=5.0)

        # Assert
        assert result["success"] is False
        assert "Prediction failed" in result["error"]
        assert "timeout" in result["error"].lower()


class TestMistralOcrClientAsyncPayload:
    """Test payload construction."""

    @pytest.mark.asyncio
    @patch("src.ocr.mistral_ocr_client_async.get_vertex_ai_client")
    @patch("aiofiles.open")
    @patch("os.path.exists")
    async def test_payload_structure(
        self, mock_exists, mock_aiofiles_open, mock_get_client, sample_pdf_bytes, sample_pdf_base64
    ):
        """Test that payload is correctly structured."""
        # Setup mocks
        mock_exists.return_value = True

        mock_file = AsyncMock()
        mock_file.read = AsyncMock(return_value=sample_pdf_bytes)
        mock_aiofiles_open.return_value = AsyncContextManagerMock(mock_file)

        mock_vertex_client = MagicMock()
        mock_vertex_client.generate = Mock(return_value={"candidates": [{"content": "mock"}]})
        mock_get_client.return_value = mock_vertex_client

        # Execute
        client = MistralOcrClientAsync()
        await client.process_pdf("test.pdf", save_response=False)

        # Assert payload structure
        call_kwargs = mock_vertex_client.generate.call_args.kwargs
        payload = call_kwargs["payload"]

        assert "model" in payload
        assert payload["model"] == client.model_id
        assert "document" in payload
        assert payload["document"]["type"] == "document_url"
        assert "document_url" in payload["document"]
        assert payload["document"]["document_url"].startswith("data:application/pdf;base64,")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
