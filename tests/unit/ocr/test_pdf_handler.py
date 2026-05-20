"""
Unit tests for PDF handler module.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PyPDF2 import PdfWriter

from src.ocr.pdf_handler import (
    _validate_pdf_input,
    calculate_pages_by_size,
    split_pdf_by_pages,
    split_pdf_by_size,
)


def create_test_pdf(num_pages: int, output_path: str = None) -> str:
    """Create a temporary PDF file with specified number of pages.

    Args:
        num_pages: Number of pages to create
        output_path: Optional path for the PDF file

    Returns:
        Path to the created PDF file
    """
    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix=".pdf")
        os.close(fd)

    writer = PdfWriter()

    for _ in range(num_pages):
        writer.add_blank_page(width=612, height=792)

    with open(output_path, "wb") as f:
        writer.write(f)

    return output_path


def cleanup_test_files(file_paths: list):
    """Clean up test files."""
    for file_path in file_paths:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception:
            pass


@pytest.fixture
def mock_gcs_upload():
    """Fixture to mock GCS upload function."""
    with patch("ocr.pdf_handler.upload_to_gcs") as mock_upload:

        def upload_side_effect(local_path, gcs_path):
            return f"gs://test-bucket/{gcs_path}"

        mock_upload.side_effect = upload_side_effect
        yield mock_upload


TEST_SESSION_KEY = "test-session-1"


@pytest.fixture
def mock_config():
    """Fixture to mock ocr_config used by split functions."""
    mock_cfg = MagicMock()
    mock_cfg.GCS_BUCKET_NAME = "test-bucket"
    mock_cfg.GCS_WORKING_FOLDER = "ocr-base"
    with patch("core.config.ocr_config", mock_cfg):
        yield mock_cfg


@pytest.fixture
def temp_pdf_file():
    """Fixture to create a temporary PDF file."""
    pdf_path = create_test_pdf(10)
    yield pdf_path
    cleanup_test_files([pdf_path])


class TestCalculatePagesBySize:
    """Test cases for calculate_pages_by_size function."""

    def test_normal_case(self):
        result = calculate_pages_by_size(20.0, 100, 5.0)
        assert result == 25

    def test_small_file(self):
        result = calculate_pages_by_size(2.0, 10, 5.0)
        assert result == 10

    def test_large_file(self):
        result = calculate_pages_by_size(25.0, 200, 5.0)
        assert result == 40

    def test_edge_case_zero_mb(self):
        result = calculate_pages_by_size(0.0, 10, 5.0)
        assert result == 10

    def test_edge_case_zero_pages(self):
        result = calculate_pages_by_size(10.0, 0, 5.0)
        assert result == 1

    def test_custom_size_limit(self):
        result = calculate_pages_by_size(20.0, 100, 10.0)
        assert result == 50

    def test_fractional_result(self):
        result = calculate_pages_by_size(15.0, 100, 5.0)
        assert result == 34


class TestValidatePdfInput:
    """Test cases for _validate_pdf_input function."""

    def test_valid_pdf_path(self, temp_pdf_file):
        _validate_pdf_input(temp_pdf_file)

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError, match="File not found"):
            _validate_pdf_input("/nonexistent/file.pdf")

    def test_non_pdf_extension(self):
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            temp_file = f.name

        try:
            with pytest.raises(ValueError, match="File must be a PDF"):
                _validate_pdf_input(temp_file)
        finally:
            cleanup_test_files([temp_file])

    def test_case_insensitive_extension(self, temp_pdf_file):
        with tempfile.NamedTemporaryFile(suffix=".PDF", delete=False) as f:
            temp_file = f.name

        try:
            writer = PdfWriter()
            writer.add_blank_page(width=612, height=792)
            with open(temp_file, "wb") as pdf_file:
                writer.write(pdf_file)

            _validate_pdf_input(temp_file)
        finally:
            cleanup_test_files([temp_file])


class TestSplitPdfBySize:
    """Test cases for split_pdf_by_size function."""

    def test_successful_split(self, mock_gcs_upload, mock_config):
        """Test successful PDF split by size."""
        pdf_path = create_test_pdf(20)

        try:
            result = split_pdf_by_size(
                pdf_path,
                TEST_SESSION_KEY,
                size_limit_mb=5.0,
            )

            assert mock_gcs_upload.called
            assert len(result) > 0
            # Check that results contain the session key and tmp folder
            assert all(f"{TEST_SESSION_KEY}/tmp/" in path for path in result)
            assert any("_p1-" in path for path in result)
        finally:
            cleanup_test_files([pdf_path])

    def test_file_not_found(self, mock_gcs_upload, mock_config):
        """Test that FileNotFoundError is raised for non-existent files."""
        with pytest.raises(FileNotFoundError, match="File not found"):
            split_pdf_by_size("/nonexistent/file.pdf", TEST_SESSION_KEY)

    def test_non_pdf_file(self, mock_gcs_upload, mock_config):
        """Test that ValueError is raised for non-PDF files."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            temp_file = f.name

        try:
            with pytest.raises(ValueError, match="File must be a PDF"):
                split_pdf_by_size(temp_file, TEST_SESSION_KEY)
        finally:
            cleanup_test_files([temp_file])

    def test_naming_convention(self, mock_gcs_upload, mock_config):
        """Test that split PDFs follow the naming convention."""
        pdf_path = create_test_pdf(15)
        original_name = Path(pdf_path).stem

        try:
            result = split_pdf_by_size(pdf_path, TEST_SESSION_KEY, size_limit_mb=5.0)

            for gcs_path in result:
                filename = gcs_path.split("/")[-1]
                assert filename.startswith(f"{original_name}_p")
                assert filename.endswith(".pdf")
                assert "_p" in filename
        finally:
            cleanup_test_files([pdf_path])

    def test_gcs_paths_format(self, mock_gcs_upload, mock_config):
        """Test that GCS paths have the correct format."""
        pdf_path = create_test_pdf(10)

        try:
            result = split_pdf_by_size(pdf_path, TEST_SESSION_KEY)

            for path in result:
                assert path.startswith("gs://")
                assert f"/{TEST_SESSION_KEY}/tmp/" in path
        finally:
            cleanup_test_files([pdf_path])

    def test_temporary_files_cleaned_up(self, mock_gcs_upload, mock_config):
        """Test that temporary files are cleaned up after splitting."""
        pdf_path = create_test_pdf(10)

        try:
            result = split_pdf_by_size(pdf_path, TEST_SESSION_KEY)
            assert len(result) > 0
        finally:
            cleanup_test_files([pdf_path])

    def test_small_file_single_chunk(self, mock_gcs_upload, mock_config):
        """Test that a small file results in a single chunk."""
        pdf_path = create_test_pdf(5)

        try:
            result = split_pdf_by_size(pdf_path, TEST_SESSION_KEY, size_limit_mb=100.0)
            assert len(result) == 1
        finally:
            cleanup_test_files([pdf_path])

    def test_empty_unique_key(self, mock_gcs_upload, mock_config):
        """Test that ValueError is raised for empty unique_key."""
        pdf_path = create_test_pdf(5)
        try:
            with pytest.raises(ValueError, match="unique_key"):
                split_pdf_by_size(pdf_path, "")
        finally:
            cleanup_test_files([pdf_path])

    def test_unique_key_with_slash(self, mock_gcs_upload, mock_config):
        """Test that ValueError is raised for unique_key containing slashes."""
        pdf_path = create_test_pdf(5)
        try:
            with pytest.raises(ValueError, match="slashes"):
                split_pdf_by_size(pdf_path, "a/b")
        finally:
            cleanup_test_files([pdf_path])


class TestSplitPdfByPages:
    """Test cases for split_pdf_by_pages function."""

    def test_successful_split(self, mock_gcs_upload, mock_config):
        """Test successful PDF split by page count."""
        pdf_path = create_test_pdf(20)

        try:
            result = split_pdf_by_pages(pdf_path, 5, TEST_SESSION_KEY)

            assert len(result) == 4
            assert mock_gcs_upload.called
            assert all(f"{TEST_SESSION_KEY}/tmp/" in path for path in result)
        finally:
            cleanup_test_files([pdf_path])

    def test_invalid_pages_per_chunk_zero(self, mock_gcs_upload, mock_config):
        """Test that ValueError is raised for pages_per_chunk of 0."""
        pdf_path = create_test_pdf(10)

        try:
            with pytest.raises(ValueError, match="pages_per_chunk must be greater than 0"):
                split_pdf_by_pages(pdf_path, 0, TEST_SESSION_KEY)
        finally:
            cleanup_test_files([pdf_path])

    def test_invalid_pages_per_chunk_negative(self, mock_gcs_upload, mock_config):
        """Test that ValueError is raised for negative pages_per_chunk."""
        pdf_path = create_test_pdf(10)

        try:
            with pytest.raises(ValueError, match="pages_per_chunk must be greater than 0"):
                split_pdf_by_pages(pdf_path, -1, TEST_SESSION_KEY)
        finally:
            cleanup_test_files([pdf_path])

    def test_file_not_found(self, mock_gcs_upload, mock_config):
        """Test that FileNotFoundError is raised for non-existent files."""
        with pytest.raises(FileNotFoundError, match="File not found"):
            split_pdf_by_pages("/nonexistent/file.pdf", 5, TEST_SESSION_KEY)

    def test_non_pdf_file(self, mock_gcs_upload, mock_config):
        """Test that ValueError is raised for non-PDF files."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            temp_file = f.name

        try:
            with pytest.raises(ValueError, match="File must be a PDF"):
                split_pdf_by_pages(temp_file, 5, TEST_SESSION_KEY)
        finally:
            cleanup_test_files([temp_file])

    def test_naming_convention(self, mock_gcs_upload, mock_config):
        """Test that split PDFs follow the naming convention."""
        pdf_path = create_test_pdf(15)
        original_name = Path(pdf_path).stem

        try:
            result = split_pdf_by_pages(pdf_path, 5, TEST_SESSION_KEY)

            for gcs_path in result:
                filename = gcs_path.split("/")[-1]
                assert filename.startswith(f"{original_name}_p")
                assert filename.endswith(".pdf")
                assert "_p" in filename
        finally:
            cleanup_test_files([pdf_path])

    def test_gcs_paths_format(self, mock_gcs_upload, mock_config):
        """Test that GCS paths have the correct format."""
        pdf_path = create_test_pdf(10)

        try:
            result = split_pdf_by_pages(pdf_path, 3, TEST_SESSION_KEY)

            for path in result:
                assert path.startswith("gs://")
                assert f"/{TEST_SESSION_KEY}/tmp/" in path
        finally:
            cleanup_test_files([pdf_path])

    def test_pages_not_divisible_evenly(self, mock_gcs_upload, mock_config):
        """Test splitting when pages don't divide evenly."""
        pdf_path = create_test_pdf(7)

        try:
            result = split_pdf_by_pages(pdf_path, 3, TEST_SESSION_KEY)

            assert len(result) == 3

            last_filename = result[-1].split("/")[-1]
            assert "p7-7" in last_filename
        finally:
            cleanup_test_files([pdf_path])

    def test_single_page_chunk(self, mock_gcs_upload, mock_config):
        """Test splitting into single-page chunks."""
        pdf_path = create_test_pdf(5)

        try:
            result = split_pdf_by_pages(pdf_path, 1, TEST_SESSION_KEY)

            assert len(result) == 5
        finally:
            cleanup_test_files([pdf_path])

    def test_large_chunk_size(self, mock_gcs_upload, mock_config):
        """Test with chunk size larger than total pages."""
        pdf_path = create_test_pdf(5)

        try:
            result = split_pdf_by_pages(pdf_path, 10, TEST_SESSION_KEY)

            assert len(result) == 1
        finally:
            cleanup_test_files([pdf_path])

    def test_temporary_files_cleaned_up(self, mock_gcs_upload, mock_config):
        """Test that temporary files are cleaned up after splitting."""
        pdf_path = create_test_pdf(10)

        try:
            split_pdf_by_pages(pdf_path, 3, TEST_SESSION_KEY)
        finally:
            cleanup_test_files([pdf_path])
