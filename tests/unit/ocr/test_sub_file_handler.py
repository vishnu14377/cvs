"""
Unit tests for SubFileHandler.

Tests the sub-file processing pipeline including:
- Filename parsing for page range extraction
- Page number mapping from sub-file index to original document
- Integration with OcrModelClient (mocked)
- Metadata creation and page mapping

Run with: pytest tests/unit/test_sub_file_handler.py -v
"""

from unittest.mock import MagicMock, patch

import pytest
from src.ocr.data_models.normalized_response import (
    NormalizedOcrResponse,
    NormalizedPage,
)
from src.ocr.data_models.sub_file_models import (
    map_page_to_original,
)
from src.ocr.ocr_model_client import OcrModelClient
from src.ocr.sub_file_handler import (
    SubFileHandler,
    get_sub_file_handler,
    parse_filename_page_range,
)

# =============================================================================
# Tests: parse_filename_page_range
# =============================================================================


class TestParseFilenamePageRange:
    """Tests for parse_filename_page_range function."""

    def test_simple_filename(self):
        """Parse simple filename with page range."""
        doc_name, start, end = parse_filename_page_range("test_document_p1-10.pdf")

        assert doc_name == "test_document"
        assert start == 1
        assert end == 10

    def test_complex_filename(self):
        """Parse complex filename with underscores and commas."""
        doc_name, start, end = parse_filename_page_range(
            "COLLINS,ALEXANDER_5010280528530_FLC_REDACTED_p19-36.pdf"
        )

        assert doc_name == "COLLINS,ALEXANDER_5010280528530_FLC_REDACTED"
        assert start == 19
        assert end == 36

    def test_single_page(self):
        """Parse filename with single page range (same start and end)."""
        doc_name, start, end = parse_filename_page_range("document_p5-5.pdf")

        assert doc_name == "document"
        assert start == 5
        assert end == 5

    def test_with_full_path(self):
        """Parse filename when full path is provided."""
        doc_name, start, end = parse_filename_page_range("/path/to/docs/report_p1-20.pdf")

        assert doc_name == "report"
        assert start == 1
        assert end == 20

    def test_uppercase_pdf_extension(self):
        """Parse filename with uppercase PDF extension."""
        doc_name, start, end = parse_filename_page_range("document_p1-5.PDF")

        assert doc_name == "document"
        assert start == 1
        assert end == 5

    def test_invalid_filename_no_page_range(self):
        """Raise error for filename without page range."""
        with pytest.raises(ValueError) as exc_info:
            parse_filename_page_range("document.pdf")

        assert "does not match expected pattern" in str(exc_info.value)

    def test_invalid_filename_wrong_format(self):
        """Raise error for filename with wrong format."""
        with pytest.raises(ValueError):
            parse_filename_page_range("document_pages1-10.pdf")

    def test_invalid_start_page_zero(self):
        """Raise error when start page is zero."""
        with pytest.raises(ValueError) as exc_info:
            parse_filename_page_range("document_p0-10.pdf")

        assert "Start page must be >= 1" in str(exc_info.value)

    def test_invalid_end_less_than_start(self):
        """Raise error when end page is less than start page."""
        with pytest.raises(ValueError) as exc_info:
            parse_filename_page_range("document_p10-5.pdf")

        assert "End page" in str(exc_info.value)
        assert "must be >= start page" in str(exc_info.value)


# =============================================================================
# Tests: map_page_to_original
# =============================================================================


class TestMapPageToOriginal:
    """Tests for map_page_to_original function."""

    def test_first_page(self):
        """Map first page (index 0) to base page number."""
        result = map_page_to_original(sub_file_index=0, base_page_number=19)
        assert result == 19

    def test_middle_page(self):
        """Map middle page to correct original page number."""
        result = map_page_to_original(sub_file_index=5, base_page_number=19)
        assert result == 24

    def test_base_page_one(self):
        """Map when base page is 1."""
        result = map_page_to_original(sub_file_index=0, base_page_number=1)
        assert result == 1

        result = map_page_to_original(sub_file_index=9, base_page_number=1)
        assert result == 10


# =============================================================================
# Tests: SubFileHandler initialization
# =============================================================================


class TestSubFileHandlerInit:
    """Tests for SubFileHandler initialization."""

    @patch("src.ocr.sub_file_handler.OcrModelClient")
    def test_init_with_default_model(self, mock_ocr_client_class):
        """Initialize with default Mistral model."""
        handler = SubFileHandler(key="test_key")

        assert handler._key == "test_key"
        mock_ocr_client_class.assert_called_once_with(model_type="mistral")

    @patch("src.ocr.sub_file_handler.OcrModelClient")
    def test_init_with_llm_model(self, mock_ocr_client_class):
        """Initialize with LLM model type."""
        SubFileHandler(key="test_key", model_type="llm")

        mock_ocr_client_class.assert_called_once_with(model_type="llm")

    def test_init_with_custom_client(self):
        """Initialize with custom OcrModelClient instance."""
        mock_client = MagicMock(spec=OcrModelClient)
        mock_client.model_type = "llm"

        handler = SubFileHandler(key="test_key", ocr_client=mock_client)

        assert handler._ocr_client is mock_client

    def test_init_empty_key_raises(self):
        """Raise error for empty key."""
        with pytest.raises(ValueError) as exc_info:
            SubFileHandler(key="")

        assert "must not be empty" in str(exc_info.value)

    def test_init_key_with_slash_raises(self):
        """Raise error for key containing slashes."""
        with pytest.raises(ValueError) as exc_info:
            SubFileHandler(key="path/to/key")

        assert "must not contain slashes" in str(exc_info.value)


# =============================================================================
# Tests: SubFileHandler.process_sub_file
# =============================================================================


class TestSubFileHandlerProcessSubFile:
    """Tests for SubFileHandler.process_sub_file method."""

    @pytest.fixture
    def mock_ocr_client(self):
        """Create a mock OcrModelClient."""
        client = MagicMock(spec=OcrModelClient)
        client.model_type = "mistral"
        return client

    @pytest.fixture
    def handler(self, mock_ocr_client):
        """Create a SubFileHandler with mocked OCR client."""
        return SubFileHandler(key="test_key", ocr_client=mock_ocr_client)

    @pytest.fixture
    def temp_pdf_file(self, tmp_path):
        """Create a temporary PDF file with proper naming."""
        pdf_path = tmp_path / "test_document_p1-3.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake content")
        return str(pdf_path)

    def test_process_success(self, handler, mock_ocr_client, temp_pdf_file):
        """Test successful sub-file processing."""
        # Mock OCR response with NormalizedOcrResponse
        mock_ocr_client.process_pdf.return_value = NormalizedOcrResponse(
            success=True,
            pages=[
                NormalizedPage(index=0, extracted_text="Page 1 content"),
                NormalizedPage(index=1, extracted_text="Page 2 content"),
                NormalizedPage(index=2, extracted_text="Page 3 content"),
            ],
            error=None,
        )

        result = handler.process_sub_file(temp_pdf_file)

        assert result.success is True
        assert result.metadata.document_name == "test_document"
        assert result.metadata.base_page_number == 1
        assert result.metadata.end_page_number == 3
        assert len(result.pages) == 3

        # Check page mapping
        assert result.pages[0].sub_file_index == 0
        assert result.pages[0].original_page_number == 1
        assert result.pages[0].extracted_text == "Page 1 content"

        assert result.pages[2].sub_file_index == 2
        assert result.pages[2].original_page_number == 3

    def test_process_with_page_offset(self, handler, mock_ocr_client, tmp_path):
        """Test page number mapping with offset (pages 19-21)."""
        pdf_path = tmp_path / "document_p19-21.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake content")

        mock_ocr_client.process_pdf.return_value = NormalizedOcrResponse(
            success=True,
            pages=[
                NormalizedPage(index=0, extracted_text="Content of page 19"),
                NormalizedPage(index=1, extracted_text="Content of page 20"),
                NormalizedPage(index=2, extracted_text="Content of page 21"),
            ],
        )

        result = handler.process_sub_file(str(pdf_path))

        assert result.success is True
        assert result.metadata.base_page_number == 19
        assert result.metadata.end_page_number == 21

        # Check page mapping with offset
        assert result.pages[0].original_page_number == 19
        assert result.pages[1].original_page_number == 20
        assert result.pages[2].original_page_number == 21

    def test_process_ocr_failure(self, handler, mock_ocr_client, temp_pdf_file):
        """Test handling of OCR processing failure."""
        mock_ocr_client.process_pdf.return_value = NormalizedOcrResponse(
            success=False,
            pages=[],
            error="OCR service unavailable",
        )

        result = handler.process_sub_file(temp_pdf_file)

        assert result.success is False
        assert result.error == "OCR service unavailable"
        assert result.pages == []

    def test_process_invalid_filename(self, handler, mock_ocr_client, tmp_path):
        """Test handling of invalid filename pattern."""
        pdf_path = tmp_path / "invalid_name.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake content")

        result = handler.process_sub_file(str(pdf_path))

        assert result.success is False
        assert "does not match expected pattern" in result.error
        # OCR should not be called for invalid filename
        mock_ocr_client.process_pdf.assert_not_called()


# =============================================================================
# Tests: SubFileHandler._extract_pages_from_response
# =============================================================================


class TestExtractPagesFromResponse:
    """Tests for SubFileHandler._extract_pages_from_response method."""

    @pytest.fixture
    def handler(self):
        mock_client = MagicMock(spec=OcrModelClient)
        mock_client.model_type = "mistral"
        return SubFileHandler(key="test", ocr_client=mock_client)

    def test_extract_pages_basic(self, handler):
        """Extract pages from NormalizedOcrResponse."""
        ocr_result = NormalizedOcrResponse(
            success=True,
            pages=[
                NormalizedPage(index=0, extracted_text="First page"),
                NormalizedPage(index=1, extracted_text="Second page"),
            ],
        )

        pages = handler._extract_pages_from_response(ocr_result, base_page=10)

        assert len(pages) == 2
        assert pages[0].sub_file_index == 0
        assert pages[0].original_page_number == 10
        assert pages[0].extracted_text == "First page"

        assert pages[1].sub_file_index == 1
        assert pages[1].original_page_number == 11
        assert pages[1].extracted_text == "Second page"

    def test_extract_empty_pages(self, handler):
        """Extract from empty pages list."""
        ocr_result = NormalizedOcrResponse(success=True, pages=[])

        pages = handler._extract_pages_from_response(ocr_result, base_page=1)

        assert pages == []


# =============================================================================
# Tests: get_sub_file_handler factory
# =============================================================================


class TestGetSubFileHandler:
    """Tests for get_sub_file_handler factory function."""

    @patch("src.ocr.sub_file_handler.OcrModelClient")
    def test_creates_handler_with_defaults(self, mock_ocr_class):
        """Create handler with default settings."""
        handler = get_sub_file_handler(key="my_key")

        assert isinstance(handler, SubFileHandler)
        assert handler._key == "my_key"

    @patch("src.ocr.sub_file_handler.OcrModelClient")
    def test_creates_handler_with_llm(self, mock_ocr_class):
        """Create handler with LLM model type."""
        get_sub_file_handler(key="my_key", model_type="llm")

        mock_ocr_class.assert_called_with(model_type="llm")
