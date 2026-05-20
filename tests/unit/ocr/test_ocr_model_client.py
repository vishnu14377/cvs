"""Unit tests for ocr_model_client module.

Tests the unified OCR model client that abstracts Mistral and LLM backends.
All responses are normalized to contain only 'index' and 'extracted_text' per page.
Includes tests for automatic fallback from Mistral to LLM when primary fails.
"""

from unittest.mock import MagicMock, patch

import pytest
from src.ocr.llm_ocr_client import LLMOcrClient
from src.ocr.mistral_ocr_client import MistralOcrClient
from src.ocr.ocr_model_client import (
    OcrModelClient,
    get_ocr_model_client,
)


class TestOcrModelClientInit:
    """Tests for OcrModelClient initialization."""

    @patch("src.ocr.ocr_model_client.MistralOcrClient")
    def test_init_default_is_mistral(self, mock_mistral_class):
        client = OcrModelClient()
        assert client.model_type == "mistral"
        mock_mistral_class.assert_called_once()

    @patch("src.ocr.ocr_model_client.MistralOcrClient")
    def test_init_with_mistral_type(self, mock_mistral_class):
        client = OcrModelClient(model_type="mistral")
        assert client.model_type == "mistral"
        mock_mistral_class.assert_called_once()

    @patch("src.ocr.ocr_model_client.LLMOcrClient")
    def test_init_with_llm_type(self, mock_llm_class):
        client = OcrModelClient(model_type="llm")
        assert client.model_type == "llm"
        mock_llm_class.assert_called_once()

    def test_init_with_mistral_client_instance(self):
        mock_mistral = MagicMock(spec=MistralOcrClient)
        client = OcrModelClient(client=mock_mistral)
        assert client.model_type == "mistral"
        assert client._primary_client is mock_mistral

    def test_init_with_llm_client_instance(self):
        mock_llm = MagicMock(spec=LLMOcrClient)
        client = OcrModelClient(client=mock_llm)
        assert client.model_type == "llm"
        assert client._primary_client is mock_llm

    def test_init_client_overrides_model_type(self):
        """When a client is provided, model_type parameter is ignored."""
        mock_llm = MagicMock(spec=LLMOcrClient)
        # Even though model_type="mistral", it should detect LLM from client
        client = OcrModelClient(model_type="mistral", client=mock_llm)
        assert client.model_type == "llm"

    @patch("src.ocr.ocr_model_client.MistralOcrClient")
    def test_init_fallback_enabled_by_default(self, mock_mistral_class):
        client = OcrModelClient()
        assert client.enable_fallback is True

    @patch("src.ocr.ocr_model_client.MistralOcrClient")
    def test_init_fallback_can_be_disabled(self, mock_mistral_class):
        client = OcrModelClient(enable_fallback=False)
        assert client.enable_fallback is False


class TestOcrModelClientProcessPdfMistral:
    """Tests for OcrModelClient.process_pdf with Mistral backend."""

    @pytest.fixture
    def mock_mistral_client(self):
        return MagicMock(spec=MistralOcrClient)

    @pytest.fixture
    def client(self, mock_mistral_client):
        return OcrModelClient(client=mock_mistral_client)

    def test_process_pdf_success(self, client, mock_mistral_client):
        """Test successful PDF processing with Mistral - normalizes markdown to extracted_text."""
        mock_mistral_client.process_pdf.return_value = {
            "success": True,
            "response": {
                "pages": [
                    {"index": 0, "markdown": "Page 1 content"},
                    {"index": 1, "markdown": "Page 2 content"},
                ]
            },
        }

        result = client.process_pdf("test.pdf")

        assert result.success is True
        assert len(result.pages) == 2
        assert result.pages[0].index == 0
        assert result.pages[0].extracted_text == "Page 1 content"
        assert result.pages[1].index == 1
        assert result.pages[1].extracted_text == "Page 2 content"
        assert result.error is None
        assert result.model_used == "mistral"
        assert result.fallback_used is False
        assert result.primary_error is None

    def test_process_pdf_failure(self, client, mock_mistral_client):
        mock_mistral_client.process_pdf.return_value = {
            "success": False,
            "error": "OCR failed",
            "response": None,
        }

        # Note: With fallback enabled by default, this will try LLM
        # To test pure Mistral failure, disable fallback
        client_no_fallback = OcrModelClient(client=mock_mistral_client, enable_fallback=False)
        result = client_no_fallback.process_pdf("test.pdf")

        assert result.success is False
        assert result.pages == []
        assert result.error == "OCR failed"
        assert result.model_used == "mistral"

    def test_process_pdf_passes_timeout(self, client, mock_mistral_client):
        mock_mistral_client.process_pdf.return_value = {"success": True, "response": {"pages": []}}

        client.process_pdf("test.pdf", timeout=60.0, save_response=True)

        mock_mistral_client.process_pdf.assert_called_once_with(
            pdf_path="test.pdf",
            timeout=60.0,
            save_response=True,
        )

    def test_process_pdf_empty_pages(self, client, mock_mistral_client):
        mock_mistral_client.process_pdf.return_value = {"success": True, "response": {"pages": []}}

        result = client.process_pdf("test.pdf")

        assert result.success is True
        assert result.pages == []


class TestOcrModelClientProcessPdfLLM:
    """Tests for OcrModelClient.process_pdf with LLM backend."""

    @pytest.fixture
    def mock_llm_client(self):
        return MagicMock(spec=LLMOcrClient)

    @pytest.fixture
    def client(self, mock_llm_client):
        return OcrModelClient(client=mock_llm_client)

    def test_process_pdf_success(self, client, mock_llm_client):
        """Test successful PDF processing with LLM - returns pages with extracted_text."""
        mock_llm_client.process_document.return_value = {
            "success": True,
            "pages": [
                {"index": 0, "extracted_text": "Page 1 content"},
                {"index": 1, "extracted_text": "Page 2 content"},
            ],
        }

        result = client.process_pdf("test.pdf")

        assert result.success is True
        assert len(result.pages) == 2
        assert result.pages[0].index == 0
        assert result.pages[0].extracted_text == "Page 1 content"
        assert result.pages[1].index == 1
        assert result.pages[1].extracted_text == "Page 2 content"
        assert result.error is None
        assert result.model_used == "llm"
        assert result.fallback_used is False
        assert result.primary_error is None

    def test_process_pdf_failure(self, client, mock_llm_client):
        mock_llm_client.process_document.return_value = {
            "success": False,
            "error": "LLM service error",
            "pages": [],
        }

        # Disable fallback to test pure LLM failure
        client_no_fallback = OcrModelClient(client=mock_llm_client, enable_fallback=False)
        result = client_no_fallback.process_pdf("test.pdf")

        assert result.success is False
        assert result.pages == []
        assert result.error == "LLM service error"
        assert result.model_used == "llm"

    def test_process_pdf_passes_timeout(self, client, mock_llm_client):
        mock_llm_client.process_document.return_value = {
            "success": True,
            "pages": [],
        }

        client.process_pdf("test.pdf", timeout=120.0, save_response=False)

        mock_llm_client.process_document.assert_called_once_with(
            file_path="test.pdf",
            timeout=120.0,
            save_response=False,
        )

    def test_process_pdf_empty_pages(self, client, mock_llm_client):
        mock_llm_client.process_document.return_value = {
            "success": True,
            "pages": [],
        }

        result = client.process_pdf("test.pdf")

        assert result.success is True
        assert result.pages == []


class TestGetOcrModelClient:
    """Tests for get_ocr_model_client factory function."""

    @patch("src.ocr.ocr_model_client.MistralOcrClient")
    def test_creates_default_mistral_client(self, mock_mistral_class):
        client = get_ocr_model_client()
        assert isinstance(client, OcrModelClient)
        assert client.model_type == "mistral"

    @patch("src.ocr.ocr_model_client.LLMOcrClient")
    def test_creates_llm_client(self, mock_llm_class):
        client = get_ocr_model_client(model_type="llm")
        assert client.model_type == "llm"

    def test_creates_client_with_instance(self):
        mock_client = MagicMock(spec=MistralOcrClient)
        client = get_ocr_model_client(client=mock_client)
        assert client._primary_client is mock_client

    @patch("src.ocr.ocr_model_client.MistralOcrClient")
    def test_creates_client_with_fallback_enabled(self, mock_mistral_class):
        client = get_ocr_model_client(enable_fallback=True)
        assert client.enable_fallback is True

    @patch("src.ocr.ocr_model_client.MistralOcrClient")
    def test_creates_client_with_fallback_disabled(self, mock_mistral_class):
        client = get_ocr_model_client(enable_fallback=False)
        assert client.enable_fallback is False


class TestOcrModelClientFallback:
    """Tests for OcrModelClient fallback behavior."""

    @pytest.fixture
    def mock_mistral_client(self):
        return MagicMock(spec=MistralOcrClient)

    @pytest.fixture
    def mock_llm_client(self):
        return MagicMock(spec=LLMOcrClient)

    def test_fallback_to_llm_on_mistral_failure(self, mock_mistral_client, mock_llm_client):
        """Test automatic fallback to LLM when Mistral fails."""
        # Mistral fails
        mock_mistral_client.process_pdf.return_value = {
            "success": False,
            "error": "Mistral API timeout",
            "response": None,
        }

        # LLM succeeds
        mock_llm_client.process_document.return_value = {
            "success": True,
            "pages": [
                {"index": 0, "extracted_text": "Fallback content"},
            ],
        }

        client = OcrModelClient(client=mock_mistral_client, enable_fallback=True)

        # Patch the fallback client creation
        with patch.object(client, "_get_fallback_client", return_value=mock_llm_client):
            result = client.process_pdf("test.pdf")

        assert result.success is True
        assert len(result.pages) == 1
        assert result.pages[0].extracted_text == "Fallback content"
        assert result.model_used == "llm"
        assert result.fallback_used is True
        assert result.primary_error == "Mistral API timeout"

    def test_fallback_to_mistral_on_llm_failure(self, mock_mistral_client, mock_llm_client):
        """Test automatic fallback to Mistral when LLM fails."""
        # LLM fails
        mock_llm_client.process_document.return_value = {
            "success": False,
            "error": "LLM quota exceeded",
            "pages": [],
        }

        # Mistral succeeds
        mock_mistral_client.process_pdf.return_value = {
            "success": True,
            "response": {
                "pages": [
                    {"index": 0, "markdown": "Mistral fallback content"},
                ]
            },
        }

        client = OcrModelClient(client=mock_llm_client, enable_fallback=True)

        # Patch the fallback client creation
        with patch.object(client, "_get_fallback_client", return_value=mock_mistral_client):
            result = client.process_pdf("test.pdf")

        assert result.success is True
        assert len(result.pages) == 1
        assert result.pages[0].extracted_text == "Mistral fallback content"
        assert result.model_used == "mistral"
        assert result.fallback_used is True
        assert result.primary_error == "LLM quota exceeded"

    def test_no_fallback_when_disabled(self, mock_mistral_client):
        """Test that fallback does not occur when disabled."""
        mock_mistral_client.process_pdf.return_value = {
            "success": False,
            "error": "Mistral failed",
            "response": None,
        }

        client = OcrModelClient(client=mock_mistral_client, enable_fallback=False)
        result = client.process_pdf("test.pdf")

        assert result.success is False
        assert result.error == "Mistral failed"
        assert result.model_used == "mistral"
        assert result.fallback_used is False
        assert result.primary_error is None

    def test_both_models_fail(self, mock_mistral_client, mock_llm_client):
        """Test behavior when both primary and fallback models fail."""
        # Mistral fails
        mock_mistral_client.process_pdf.return_value = {
            "success": False,
            "error": "Mistral API down",
            "response": None,
        }

        # LLM also fails
        mock_llm_client.process_document.return_value = {
            "success": False,
            "error": "LLM API down",
            "pages": [],
        }

        client = OcrModelClient(client=mock_mistral_client, enable_fallback=True)

        with patch.object(client, "_get_fallback_client", return_value=mock_llm_client):
            result = client.process_pdf("test.pdf")

        assert result.success is False
        assert result.error == "LLM API down"
        assert result.model_used == "llm"
        assert result.fallback_used is True
        assert result.primary_error == "Mistral API down"

    def test_no_fallback_when_primary_succeeds(self, mock_mistral_client, mock_llm_client):
        """Test that fallback is not attempted when primary model succeeds."""
        mock_mistral_client.process_pdf.return_value = {
            "success": True,
            "response": {
                "pages": [
                    {"index": 0, "markdown": "Primary content"},
                ]
            },
        }

        client = OcrModelClient(client=mock_mistral_client, enable_fallback=True)
        result = client.process_pdf("test.pdf")

        assert result.success is True
        assert result.pages[0].extracted_text == "Primary content"
        assert result.model_used == "mistral"
        assert result.fallback_used is False
        assert result.primary_error is None
        # Verify LLM client was never created/called
        mock_llm_client.process_document.assert_not_called()

    def test_fallback_client_lazy_initialization(self, mock_mistral_client):
        """Test that fallback client is only created when needed."""
        mock_mistral_client.process_pdf.return_value = {"success": True, "response": {"pages": []}}

        client = OcrModelClient(client=mock_mistral_client, enable_fallback=True)

        # Initially no fallback client
        assert client._fallback_client is None

        # Process with success - no fallback needed
        client.process_pdf("test.pdf")

        # Still no fallback client
        assert client._fallback_client is None

    @patch("src.ocr.ocr_model_client.LLMOcrClient")
    @patch("src.ocr.ocr_model_client.MistralOcrClient")
    def test_fallback_client_created_on_failure(self, mock_mistral_class, mock_llm_class):
        """Test that fallback client is created on primary failure."""
        mock_mistral_instance = MagicMock()
        mock_mistral_instance.process_pdf.return_value = {
            "success": False,
            "error": "Mistral failed",
            "response": None,
        }
        mock_mistral_class.return_value = mock_mistral_instance

        mock_llm_instance = MagicMock()
        mock_llm_instance.process_document.return_value = {
            "success": True,
            "pages": [{"index": 0, "extracted_text": "Fallback"}],
        }
        mock_llm_class.return_value = mock_llm_instance

        # Create client with default model_type="mistral"
        client = OcrModelClient(model_type="mistral", enable_fallback=True)

        # Initially no fallback client
        assert client._fallback_client is None

        client.process_pdf("test.pdf")

        # Fallback client was created (it's the mock_llm_instance)
        assert client._fallback_client is mock_llm_instance
        mock_llm_class.assert_called_once()

    def test_fallback_handles_exception_in_primary(self, mock_mistral_client, mock_llm_client):
        """Test fallback when primary model raises an exception."""
        mock_mistral_client.process_pdf.side_effect = Exception("Connection error")

        mock_llm_client.process_document.return_value = {
            "success": True,
            "pages": [{"index": 0, "extracted_text": "Recovered content"}],
        }

        client = OcrModelClient(client=mock_mistral_client, enable_fallback=True)

        with patch.object(client, "_get_fallback_client", return_value=mock_llm_client):
            result = client.process_pdf("test.pdf")

        assert result.success is True
        assert result.pages[0].extracted_text == "Recovered content"
        assert result.model_used == "llm"
        assert result.fallback_used is True
        assert "mistral processing exception" in result.primary_error.lower()
