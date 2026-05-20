"""
Unit tests for the Mistral OCR client.

Tests cover:
- Client initialization
- PDF processing workflow
- Error handling (file not found, read errors, API errors)
- Response saving logic

Run with: pytest tests/unit/test_mistral_ocr_client.py -v
"""

import base64
import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
from google.api_core import exceptions as google_exceptions
from google.api_core import gapic_v1
from src.core.vertex_ai_client import VertexAIClient
from src.ocr.mistral_ocr_client import MistralOcrClient

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_vertex_ai_client():
    """Create a mock VertexAIClient."""
    mock_client = MagicMock(spec=VertexAIClient)
    return mock_client


@pytest.fixture
def mock_mistral_client(mock_vertex_ai_client):
    """Create a MistralOcrClient with mocked dependencies."""
    with patch("src.ocr.mistral_ocr_client.get_vertex_ai_client") as mock_get_client:
        mock_get_client.return_value = mock_vertex_ai_client

        client = MistralOcrClient(
            model_id="test-model", project_id="test-project", region="us-central1"
        )
        yield client, mock_vertex_ai_client


@pytest.fixture
def sample_pdf_file():
    """Create a temporary PDF file for testing."""
    # Simple PDF header bytes (minimal valid PDF structure for testing)
    pdf_content = (
        b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\nxref\n0 1\n0000000000 65535 f \ntrailer\n<<>>\n%%EOF"
    )

    with tempfile.NamedTemporaryFile(mode="wb", suffix=".pdf", delete=False) as f:
        f.write(pdf_content)
        temp_path = f.name

    yield temp_path

    # Cleanup
    if os.path.exists(temp_path):
        os.remove(temp_path)
    # Also cleanup response file if created
    response_file = f"{os.path.splitext(temp_path)[0]}_mistral_response.json"
    if os.path.exists(response_file):
        os.remove(response_file)


@pytest.fixture(autouse=True)
def reset_vertex_singleton():
    """Reset the VertexAIClient singleton before and after each test."""
    VertexAIClient.reset_instance()
    yield
    VertexAIClient.reset_instance()


# =============================================================================
# Test: Client Initialization
# =============================================================================


class TestMistralOcrClientInitialization:
    """Tests for MistralOcrClient initialization."""

    def test_init_with_custom_model_id(self):
        """Client should initialize with custom model ID."""
        with patch("src.ocr.mistral_ocr_client.get_vertex_ai_client") as mock_get_client:
            mock_get_client.return_value = MagicMock()

            client = MistralOcrClient(model_id="custom-model-123")

            assert client.model_id == "custom-model-123"

    def test_init_with_default_model_id(self):
        """Client should use default model ID from config."""
        with patch("src.ocr.mistral_ocr_client.get_vertex_ai_client") as mock_get_client:
            mock_get_client.return_value = MagicMock()

            with patch("src.ocr.mistral_ocr_client.mistral_ocr_config") as mock_config:
                mock_config.MISTRAL_MODEL_ID = "default-ocr-model"

                client = MistralOcrClient()

                assert client.model_id == "default-ocr-model"

    def test_init_calls_get_vertex_ai_client(self):
        """Client should call get_vertex_ai_client with provided params."""
        with patch("src.ocr.mistral_ocr_client.get_vertex_ai_client") as mock_get_client:
            mock_get_client.return_value = MagicMock()

            MistralOcrClient(project_id="my-project", region="europe-west1")

            mock_get_client.assert_called_once_with(project_id="my-project", region="europe-west1")

    def test_vertex_client_property(self, mock_mistral_client):
        """Client should store the VertexAI client instance."""
        client, mock_vertex_client = mock_mistral_client

        assert client._vertex_client == mock_vertex_client


# =============================================================================
# Test: PDF Processing - Success Cases
# =============================================================================


class TestProcessPdfSuccess:
    """Tests for successful PDF processing."""

    def test_process_pdf_success(self, mock_mistral_client, sample_pdf_file):
        """Should successfully process a PDF file."""
        client, mock_vertex_client = mock_mistral_client

        # Mock successful response
        mock_vertex_client.generate.return_value = {
            "pages": [{"index": 0, "markdown": "Test content"}]
        }

        result = client.process_pdf(sample_pdf_file)

        assert result["success"] is True
        assert result["pdf_path"] == sample_pdf_file
        assert result["response"]["pages"][0]["markdown"] == "Test content"
        assert result["error"] is None

    def test_process_pdf_builds_correct_payload(self, mock_mistral_client, sample_pdf_file):
        """Should build correct payload with base64 encoded PDF."""
        client, mock_vertex_client = mock_mistral_client
        mock_vertex_client.generate.return_value = {"pages": []}

        client.process_pdf(sample_pdf_file)

        # Get the payload that was passed to generate
        call_args = mock_vertex_client.generate.call_args
        payload = call_args.kwargs.get("payload") or call_args[0][0]

        assert payload["model"] == "test-model"
        assert payload["document"]["type"] == "document_url"
        assert payload["document"]["document_url"].startswith("data:application/pdf;base64,")

    def test_process_pdf_saves_response_by_default(self, mock_mistral_client, sample_pdf_file):
        """Should save response to JSON file by default."""
        client, mock_vertex_client = mock_mistral_client
        mock_vertex_client.generate.return_value = {"pages": [{"text": "test"}]}

        result = client.process_pdf(sample_pdf_file, save_response=True)

        assert result["output_file"] is not None
        assert result["output_file"].endswith("_mistral_response.json")

        # Verify file was created with correct content
        with open(result["output_file"], encoding="utf-8") as f:
            saved_data = json.load(f)
        assert saved_data == {"pages": [{"text": "test"}]}

    def test_process_pdf_no_save_response(self, mock_mistral_client, sample_pdf_file):
        """Should not save response when save_response=False."""
        client, mock_vertex_client = mock_mistral_client
        mock_vertex_client.generate.return_value = {"pages": []}

        result = client.process_pdf(sample_pdf_file, save_response=False)

        assert result["success"] is True
        assert result["output_file"] is None

    def test_process_pdf_uses_custom_timeout(self, mock_mistral_client, sample_pdf_file):
        """Should use custom timeout when provided."""
        client, mock_vertex_client = mock_mistral_client
        mock_vertex_client.generate.return_value = {"pages": []}

        client.process_pdf(sample_pdf_file, timeout=500.0)

        call_args = mock_vertex_client.generate.call_args
        assert call_args.kwargs.get("timeout") == 500.0

    def test_process_pdf_passes_retry_param(self, mock_mistral_client, sample_pdf_file):
        """Should pass retry parameter to generate method."""
        client, mock_vertex_client = mock_mistral_client
        mock_vertex_client.generate.return_value = {"pages": []}

        client.process_pdf(sample_pdf_file, retry=gapic_v1.method.DEFAULT)

        call_args = mock_vertex_client.generate.call_args
        assert call_args.kwargs.get("retry") == gapic_v1.method.DEFAULT


# =============================================================================
# Test: PDF Processing - File Errors
# =============================================================================


class TestProcessPdfFileErrors:
    """Tests for file-related error handling."""

    def test_file_not_found(self, mock_mistral_client):
        """Should return error when file does not exist."""
        client, _ = mock_mistral_client

        result = client.process_pdf("/nonexistent/path/to/file.pdf")

        assert result["success"] is False
        assert "File not found" in result["error"]
        assert result["pdf_path"] == "/nonexistent/path/to/file.pdf"
        assert result["response"] is None

    def test_file_read_error(self, mock_mistral_client):
        """Should handle file read errors gracefully."""
        client, _ = mock_mistral_client

        with tempfile.NamedTemporaryFile(mode="w", suffix=".pdf", delete=False) as f:
            f.write("dummy content")
            temp_path = f.name

        try:
            # Mock open to raise an exception
            with (
                patch("builtins.open", side_effect=PermissionError("Access denied")),
                # Need to also patch os.path.exists to return True
                patch("os.path.exists", return_value=True),
            ):
                result = client.process_pdf(temp_path)

            assert result["success"] is False
            assert "Failed to read PDF" in result["error"]
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_file_read_ioerror(self, mock_mistral_client):
        """Should handle IOError during file reading."""
        client, _ = mock_mistral_client

        with (
            patch("os.path.exists", return_value=True),
            patch("builtins.open", side_effect=OSError("Disk error")),
        ):
            result = client.process_pdf("/some/path.pdf")

        assert result["success"] is False
        assert "Failed to read PDF" in result["error"]
        assert "Disk error" in result["error"]


# =============================================================================
# Test: PDF Processing - API Errors
# =============================================================================


class TestProcessPdfApiErrors:
    """Tests for API error handling."""

    def test_api_error_returns_failure(self, mock_mistral_client, sample_pdf_file):
        """Should return error result when API call fails."""
        client, mock_vertex_client = mock_mistral_client

        mock_vertex_client.generate.side_effect = google_exceptions.InternalServerError(
            "Server error"
        )

        result = client.process_pdf(sample_pdf_file)

        assert result["success"] is False
        assert "Prediction failed" in result["error"]
        assert "Server error" in result["error"]
        assert result["response"] is None

    def test_service_unavailable_error(self, mock_mistral_client, sample_pdf_file):
        """Should handle ServiceUnavailable errors."""
        client, mock_vertex_client = mock_mistral_client

        mock_vertex_client.generate.side_effect = google_exceptions.ServiceUnavailable(
            "Service temporarily unavailable"
        )

        result = client.process_pdf(sample_pdf_file)

        assert result["success"] is False
        assert "Prediction failed" in result["error"]

    def test_deadline_exceeded_error(self, mock_mistral_client, sample_pdf_file):
        """Should handle DeadlineExceeded errors."""
        client, mock_vertex_client = mock_mistral_client

        mock_vertex_client.generate.side_effect = google_exceptions.DeadlineExceeded(
            "Request timed out"
        )

        result = client.process_pdf(sample_pdf_file)

        assert result["success"] is False
        assert "Prediction failed" in result["error"]

    def test_invalid_argument_error(self, mock_mistral_client, sample_pdf_file):
        """Should handle InvalidArgument errors."""
        client, mock_vertex_client = mock_mistral_client

        mock_vertex_client.generate.side_effect = google_exceptions.InvalidArgument(
            "Invalid document format"
        )

        result = client.process_pdf(sample_pdf_file)

        assert result["success"] is False
        assert "Prediction failed" in result["error"]

    def test_resource_exhausted_error(self, mock_mistral_client, sample_pdf_file):
        """Should handle ResourceExhausted (quota) errors."""
        client, mock_vertex_client = mock_mistral_client

        mock_vertex_client.generate.side_effect = google_exceptions.ResourceExhausted(
            "Quota exceeded"
        )

        result = client.process_pdf(sample_pdf_file)

        assert result["success"] is False
        assert "Prediction failed" in result["error"]


# =============================================================================
# Test: Logging
# =============================================================================


class TestProcessPdfLogging:
    """Tests for logging behavior."""

    def test_logs_processing_start(self, mock_mistral_client, sample_pdf_file, caplog):
        """Should log when processing starts."""
        import logging

        caplog.set_level(logging.INFO)

        client, mock_vertex_client = mock_mistral_client
        mock_vertex_client.generate.return_value = {"pages": []}

        client.process_pdf(sample_pdf_file)

        assert "Processing PDF" in caplog.text
        assert sample_pdf_file in caplog.text

    def test_logs_success(self, mock_mistral_client, sample_pdf_file, caplog):
        """Should log success message."""
        import logging

        caplog.set_level(logging.INFO)

        client, mock_vertex_client = mock_mistral_client
        mock_vertex_client.generate.return_value = {"pages": []}

        client.process_pdf(sample_pdf_file)

        assert "Mistral OCR succeeded" in caplog.text

    def test_logs_file_not_found_error(self, mock_mistral_client, caplog):
        """Should log error when file not found."""
        import logging

        caplog.set_level(logging.ERROR)

        client, _ = mock_mistral_client

        client.process_pdf("/nonexistent/file.pdf")

        assert "File not found" in caplog.text

    def test_logs_api_error(self, mock_mistral_client, sample_pdf_file, caplog):
        """Should log error when API call fails."""
        import logging

        caplog.set_level(logging.ERROR)

        client, mock_vertex_client = mock_mistral_client
        mock_vertex_client.generate.side_effect = google_exceptions.InternalServerError(
            "Server error"
        )

        client.process_pdf(sample_pdf_file)

        assert "Prediction failed" in caplog.text

    def test_logs_saved_response_path(self, mock_mistral_client, sample_pdf_file, caplog):
        """Should log the saved response file path."""
        import logging

        caplog.set_level(logging.INFO)

        client, mock_vertex_client = mock_mistral_client
        mock_vertex_client.generate.return_value = {"pages": []}

        result = client.process_pdf(sample_pdf_file, save_response=True)

        assert "Response saved to" in caplog.text
        if result["output_file"]:
            assert result["output_file"] in caplog.text or "_mistral_response.json" in caplog.text


# =============================================================================
# Test: Base64 Encoding
# =============================================================================


class TestBase64Encoding:
    """Tests for PDF base64 encoding."""

    def test_correct_base64_encoding(self, mock_mistral_client, sample_pdf_file):
        """Should correctly base64 encode the PDF content."""
        client, mock_vertex_client = mock_mistral_client
        mock_vertex_client.generate.return_value = {"pages": []}

        # Read the actual PDF content
        with open(sample_pdf_file, "rb") as f:
            pdf_bytes = f.read()
        expected_base64 = base64.b64encode(pdf_bytes).decode("utf-8")

        client.process_pdf(sample_pdf_file)

        # Check the payload
        call_args = mock_vertex_client.generate.call_args
        payload = call_args.kwargs.get("payload") or call_args[0][0]
        document_url = payload["document"]["document_url"]

        # Extract base64 from data URL
        actual_base64 = document_url.replace("data:application/pdf;base64,", "")

        assert actual_base64 == expected_base64

    def test_handles_large_pdf(self, mock_mistral_client):
        """Should handle encoding of larger PDF files."""
        client, mock_vertex_client = mock_mistral_client
        mock_vertex_client.generate.return_value = {"pages": []}

        # Create a larger test file (1MB of data)
        large_content = b"%PDF-1.4\n" + (b"X" * 1024 * 1024)

        with tempfile.NamedTemporaryFile(mode="wb", suffix=".pdf", delete=False) as f:
            f.write(large_content)
            temp_path = f.name

        try:
            result = client.process_pdf(temp_path, save_response=False)

            assert result["success"] is True

            # Verify the call was made
            mock_vertex_client.generate.assert_called_once()
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


# =============================================================================
# Test: Response File Output
# =============================================================================


class TestResponseFileOutput:
    """Tests for response file saving."""

    def test_output_file_naming_convention(self, mock_mistral_client, sample_pdf_file):
        """Should follow naming convention for output file."""
        client, mock_vertex_client = mock_mistral_client
        mock_vertex_client.generate.return_value = {"pages": []}

        result = client.process_pdf(sample_pdf_file, save_response=True)

        expected_suffix = "_mistral_response.json"
        assert result["output_file"].endswith(expected_suffix)

        # Should have same base name
        base_name = os.path.splitext(sample_pdf_file)[0]
        assert result["output_file"] == f"{base_name}{expected_suffix}"

    def test_output_file_contains_valid_json(self, mock_mistral_client, sample_pdf_file):
        """Output file should contain valid JSON."""
        client, mock_vertex_client = mock_mistral_client
        mock_response = {
            "pages": [
                {"index": 0, "markdown": "# Title\n\nContent here"},
                {"index": 1, "markdown": "## Section\n\nMore content"},
            ],
            "metadata": {"total_pages": 2},
        }
        mock_vertex_client.generate.return_value = mock_response

        result = client.process_pdf(sample_pdf_file, save_response=True)

        with open(result["output_file"], encoding="utf-8") as f:
            saved_data = json.load(f)

        assert saved_data == mock_response

    def test_output_file_unicode_content(self, mock_mistral_client, sample_pdf_file):
        """Should handle Unicode content in response."""
        client, mock_vertex_client = mock_mistral_client
        mock_response = {"pages": [{"text": "日本語テスト 中文测试 emoji: 🎉"}]}
        mock_vertex_client.generate.return_value = mock_response

        result = client.process_pdf(sample_pdf_file, save_response=True)

        with open(result["output_file"], encoding="utf-8") as f:
            saved_data = json.load(f)

        assert saved_data["pages"][0]["text"] == "日本語テスト 中文测试 emoji: 🎉"


# =============================================================================
# Test: Result Dictionary Structure
# =============================================================================


class TestResultStructure:
    """Tests for result dictionary structure."""

    def test_success_result_structure(self, mock_mistral_client, sample_pdf_file):
        """Successful result should have all expected keys."""
        client, mock_vertex_client = mock_mistral_client
        mock_vertex_client.generate.return_value = {"pages": []}

        result = client.process_pdf(sample_pdf_file)

        assert "success" in result
        assert "pdf_path" in result
        assert "response" in result
        assert "output_file" in result
        assert "error" in result

    def test_failure_result_structure(self, mock_mistral_client):
        """Failed result should have all expected keys."""
        client, _ = mock_mistral_client

        result = client.process_pdf("/nonexistent/file.pdf")

        assert "success" in result
        assert "pdf_path" in result
        assert "response" in result
        assert "output_file" in result
        assert "error" in result

    def test_success_result_values(self, mock_mistral_client, sample_pdf_file):
        """Successful result should have correct values."""
        client, mock_vertex_client = mock_mistral_client
        mock_vertex_client.generate.return_value = {"pages": [{"text": "test"}]}

        result = client.process_pdf(sample_pdf_file, save_response=False)

        assert result["success"] is True
        assert result["pdf_path"] == sample_pdf_file
        assert result["response"] == {"pages": [{"text": "test"}]}
        assert result["output_file"] is None
        assert result["error"] is None

    def test_failure_result_values(self, mock_mistral_client):
        """Failed result should have correct values."""
        client, _ = mock_mistral_client

        result = client.process_pdf("/nonexistent/file.pdf")

        assert result["success"] is False
        assert result["pdf_path"] == "/nonexistent/file.pdf"
        assert result["response"] is None
        assert result["output_file"] is None
        assert result["error"] is not None


# =============================================================================
# Test: Integration with VertexAI Client
# =============================================================================


class TestVertexAIIntegration:
    """Tests for integration with VertexAI client."""

    def test_passes_model_id_to_vertex_client(self, mock_mistral_client, sample_pdf_file):
        """Should pass model_id to generate method."""
        client, mock_vertex_client = mock_mistral_client
        mock_vertex_client.generate.return_value = {"pages": []}

        client.process_pdf(sample_pdf_file)

        call_args = mock_vertex_client.generate.call_args
        assert call_args.kwargs.get("model_id") == "test-model"

    def test_uses_stored_vertex_client(self, mock_mistral_client, sample_pdf_file):
        """Should use the stored VertexAI client instance."""
        client, mock_vertex_client = mock_mistral_client
        mock_vertex_client.generate.return_value = {"pages": []}

        client.process_pdf(sample_pdf_file)

        mock_vertex_client.generate.assert_called_once()


# =============================================================================
# Test: Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_pdf_response(self, mock_mistral_client, sample_pdf_file):
        """Should handle empty response from API."""
        client, mock_vertex_client = mock_mistral_client
        mock_vertex_client.generate.return_value = {"pages": []}

        result = client.process_pdf(sample_pdf_file)

        assert result["success"] is True
        assert result["response"]["pages"] == []

    def test_pdf_path_with_spaces(self, mock_mistral_client):
        """Should handle PDF paths with spaces."""
        client, mock_vertex_client = mock_mistral_client
        mock_vertex_client.generate.return_value = {"pages": []}

        # Create a temp file with space in name
        with tempfile.NamedTemporaryFile(
            mode="wb", suffix=".pdf", prefix="test file ", delete=False
        ) as f:
            f.write(b"%PDF-1.4\n%%EOF")
            temp_path = f.name

        try:
            result = client.process_pdf(temp_path, save_response=False)

            assert result["success"] is True
            assert result["pdf_path"] == temp_path
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_pdf_path_with_unicode(self, mock_mistral_client):
        """Should handle PDF paths with Unicode characters."""
        client, mock_vertex_client = mock_mistral_client
        mock_vertex_client.generate.return_value = {"pages": []}

        # Create a temp file with unicode in name
        with tempfile.NamedTemporaryFile(
            mode="wb", suffix=".pdf", prefix="测试文件_", delete=False
        ) as f:
            f.write(b"%PDF-1.4\n%%EOF")
            temp_path = f.name

        try:
            result = client.process_pdf(temp_path, save_response=False)

            assert result["success"] is True
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_response_with_nested_structures(self, mock_mistral_client, sample_pdf_file):
        """Should handle complex nested response structures."""
        client, mock_vertex_client = mock_mistral_client
        complex_response = {
            "pages": [
                {
                    "index": 0,
                    "dimensions": {"width": 612, "height": 792},
                    "blocks": [
                        {"type": "text", "content": "Hello", "bbox": [0, 0, 100, 20]},
                        {"type": "table", "rows": [["a", "b"], ["c", "d"]]},
                    ],
                }
            ],
            "metadata": {"model": "mistral-ocr", "version": "1.0", "processing_time_ms": 1234},
        }
        mock_vertex_client.generate.return_value = complex_response

        result = client.process_pdf(sample_pdf_file, save_response=False)

        assert result["success"] is True
        assert result["response"] == complex_response
        assert result["response"]["pages"][0]["blocks"][1]["rows"] == [["a", "b"], ["c", "d"]]
