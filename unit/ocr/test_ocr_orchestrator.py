"""Unit tests for OcrOrchestrator."""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ocr.data_models.orchestrator_models import (
    OcrOrchestrationResult,
    SubFileProcessingResult,
)
from ocr.ocr_orchestrator import (
    OcrOrchestrator,
    get_ocr_orchestrator,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_sub_file_handler():
    """Mock the sub-file handler."""
    with patch("ocr.ocr_orchestrator.get_sub_file_handler") as mock:
        handler = MagicMock()
        mock.return_value = handler
        yield handler


@pytest.fixture
def mock_gcs_functions():
    """Mock all GCS client functions."""
    with (
        patch("ocr.ocr_orchestrator.is_gcs_uri") as mock_is_gcs,
        patch("ocr.ocr_orchestrator.download_from_gcs") as mock_download,
        patch("ocr.ocr_orchestrator.download_folder_files") as mock_download_folder,
    ):
        yield {
            "is_gcs_uri": mock_is_gcs,
            "download_from_gcs": mock_download,
            "download_folder_files": mock_download_folder,
        }


@pytest.fixture
def mock_local_directory_handler():
    """Mock local directory handler functions."""
    with (
        patch("ocr.ocr_orchestrator.get_local_temp_path") as mock_temp,
        patch("ocr.ocr_orchestrator.cleanup_local_data") as mock_cleanup,
    ):
        mock_temp.return_value = Path("/tmp/test_session/downloads")
        mock_cleanup.return_value = True
        yield {
            "get_local_temp_path": mock_temp,
            "cleanup_local_data": mock_cleanup,
        }


@pytest.fixture
def mock_pdf_splitting():
    """Mock PDF splitting functions."""
    with (
        patch("ocr.ocr_orchestrator.split_pdf_by_pages") as mock_pages,
        patch("ocr.ocr_orchestrator.split_pdf_by_size") as mock_size,
    ):
        yield {
            "split_pdf_by_pages": mock_pages,
            "split_pdf_by_size": mock_size,
        }


# =============================================================================
# Tests: OcrOrchestrator Initialization
# =============================================================================


class TestOcrOrchestratorInit:
    """Tests for OcrOrchestrator initialization."""

    def test_init_with_valid_key(self):
        """Test successful initialization with valid session_id."""
        with patch("ocr.ocr_orchestrator.get_sub_file_handler"):
            orchestrator = OcrOrchestrator(session_id="test-session-123")
            assert orchestrator.session_id == "test-session-123"

    def test_init_with_empty_key_raises_error(self):
        """Test that empty session_id raises ValueError."""
        with pytest.raises(ValueError, match="session_id must not be empty"):
            OcrOrchestrator(session_id="")

    def test_init_with_whitespace_key_raises_error(self):
        """Test that whitespace-only key raises ValueError."""
        with pytest.raises(ValueError, match="session_id must not be empty"):
            OcrOrchestrator(session_id="   ")

    def test_init_with_slash_in_key_raises_error(self):
        """Test that key with slashes raises ValueError."""
        with pytest.raises(ValueError, match="must not contain slashes"):
            OcrOrchestrator(session_id="test/session")

    def test_init_with_backslash_in_key_raises_error(self):
        """Test that key with backslashes raises ValueError."""
        with pytest.raises(ValueError, match="must not contain slashes"):
            OcrOrchestrator(session_id="test\\session")

    def test_init_with_dotdot_in_key_raises_error(self):
        """Test that key with '..' raises ValueError."""
        with pytest.raises(ValueError, match="must not contain slashes"):
            OcrOrchestrator(session_id="test..session")

    def test_init_with_custom_model_type(self):
        """Test initialization with custom model type."""
        with patch("ocr.ocr_orchestrator.get_sub_file_handler") as mock_handler:
            OcrOrchestrator(
                session_id="test-key",
                model_type="llm",
            )
            mock_handler.assert_called_once_with(
                key="test-key",
                model_type="llm",
            )

    def test_init_with_custom_size_limit(self):
        """Test initialization with custom size limit."""
        with patch("ocr.ocr_orchestrator.get_sub_file_handler"):
            orchestrator = OcrOrchestrator(
                session_id="test-key",
                size_limit_mb=10.0,
            )
            assert orchestrator._size_limit_mb == 10.0

    def test_init_with_pages_per_chunk(self):
        """Test initialization with pages per chunk."""
        with patch("ocr.ocr_orchestrator.get_sub_file_handler"):
            orchestrator = OcrOrchestrator(
                session_id="test-key",
                pages_per_chunk=15,
            )
            assert orchestrator._pages_per_chunk == 15


# =============================================================================
# Tests: OcrOrchestrator Properties
# =============================================================================


class TestOcrOrchestratorProperties:
    """Tests for OcrOrchestrator properties."""

    def test_tmp_folder_path(self):
        """Test tmp folder path property."""
        with (
            patch("ocr.ocr_orchestrator.get_sub_file_handler"),
            patch("ocr.ocr_orchestrator.ocr_config") as mock_config,
        ):
            mock_config.GCS_TEMP_FOLDER = "tmp"
            orchestrator = OcrOrchestrator(session_id="my-session")
            assert orchestrator.tmp_folder_path == "my-session/tmp"

    def test_extracted_text_folder_path(self):
        """Test extracted text folder path property."""
        with (
            patch("ocr.ocr_orchestrator.get_sub_file_handler"),
            patch("ocr.ocr_orchestrator.ocr_config") as mock_config,
        ):
            mock_config.GCS_EXTRACTED_TEXT_FOLDER = "extracted_text"
            orchestrator = OcrOrchestrator(session_id="my-session")
            assert orchestrator.extracted_text_folder_path == "my-session/extracted_text"


# =============================================================================
# Tests: OcrOrchestrator.run() Method
# =============================================================================


class TestOcrOrchestratorRun:
    """Tests for OcrOrchestrator.run() method."""

    @pytest.fixture
    def mock_orchestrator(self):
        """Create a mocked orchestrator for testing."""
        with patch("ocr.ocr_orchestrator.get_sub_file_handler") as mock_handler:
            mock_sub_handler = MagicMock()
            mock_handler.return_value = mock_sub_handler

            orchestrator = OcrOrchestrator(session_id="test-session")
            orchestrator._sub_file_handler = mock_sub_handler

            yield orchestrator, mock_sub_handler

    def test_run_with_local_file_not_found(self, mock_orchestrator):
        """Test run with non-existent local file."""
        orchestrator, _ = mock_orchestrator

        result = orchestrator.run("/nonexistent/file.pdf")

        assert not result.success
        assert "not found" in result.error.lower()

    def test_run_with_gcs_uri_downloads_file(self, mock_orchestrator):
        """Test that GCS URI triggers download."""
        orchestrator, mock_sub_handler = mock_orchestrator

        with (
            patch("ocr.ocr_orchestrator.is_gcs_uri", return_value=True),
            patch("ocr.ocr_orchestrator.download_from_gcs") as mock_download,
            patch("ocr.ocr_orchestrator.split_pdf_by_size") as mock_split,
        ):
            # Setup mocks
            mock_download.return_value = "/tmp/downloaded.pdf"
            mock_split.return_value = ["gs://bucket/chunk1.pdf"]
            mock_sub_handler.run.return_value = {
                "success": True,
                "result": MagicMock(pages=[]),
                "gcs_uri": "gs://bucket/result.json",
            }

            orchestrator.run("gs://bucket/source.pdf")

            mock_download.assert_called_once()

    def test_run_splits_pdf_by_size_when_llm_model(self, mock_orchestrator):
        """Test that PDF is split by size when using LLM model."""
        # Need to recreate orchestrator with llm model type
        with patch("ocr.ocr_orchestrator.get_sub_file_handler") as mock_handler:
            mock_sub_handler = MagicMock()
            mock_handler.return_value = mock_sub_handler

            orchestrator = OcrOrchestrator(
                session_id="test-session",
                model_type="llm",
            )
            orchestrator._sub_file_handler = mock_sub_handler

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(b"PDF content")
            temp_pdf = f.name

        try:
            with (
                patch("ocr.ocr_orchestrator.is_gcs_uri", return_value=False),
                patch("ocr.ocr_orchestrator.split_pdf_by_size") as mock_split,
            ):
                mock_split.return_value = ["gs://bucket/chunk1.pdf"]
                mock_sub_handler.run.return_value = {
                    "success": True,
                    "result": MagicMock(pages=[MagicMock()]),
                    "gcs_uri": "gs://bucket/result.json",
                }

                orchestrator.run(temp_pdf)

                mock_split.assert_called_once()
        finally:
            os.unlink(temp_pdf)

    def test_run_splits_pdf_by_pages_when_pages_per_chunk_set(self):
        """Test that PDF is split by page count when pages_per_chunk is set."""
        with patch("ocr.ocr_orchestrator.get_sub_file_handler") as mock_handler:
            mock_sub_handler = MagicMock()
            mock_handler.return_value = mock_sub_handler

            orchestrator = OcrOrchestrator(
                session_id="test-session",
                pages_per_chunk=10,
            )
            orchestrator._sub_file_handler = mock_sub_handler

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(b"PDF content")
            temp_pdf = f.name

        try:
            with (
                patch("ocr.ocr_orchestrator.is_gcs_uri", return_value=False),
                patch("ocr.ocr_orchestrator.split_pdf_by_pages") as mock_split,
            ):
                mock_split.return_value = ["gs://bucket/chunk1.pdf"]
                mock_sub_handler.run.return_value = {
                    "success": True,
                    "result": MagicMock(pages=[MagicMock()]),
                    "gcs_uri": "gs://bucket/result.json",
                }

                orchestrator.run(temp_pdf)

                mock_split.assert_called_once_with(
                    pdf_path=temp_pdf,
                    pages_per_chunk=10,
                    unique_key="test-session",
                )
        finally:
            os.unlink(temp_pdf)

    def test_run_processes_all_sub_files(self, mock_orchestrator):
        """Test that all sub-files are processed."""
        orchestrator, mock_sub_handler = mock_orchestrator

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(b"PDF content")
            temp_pdf = f.name

        try:
            with (
                patch("ocr.ocr_orchestrator.is_gcs_uri", return_value=False),
                patch("ocr.ocr_orchestrator.split_pdf_by_pages") as mock_split,
            ):
                # Setup 3 sub-files
                sub_files = [
                    "gs://bucket/doc_p1-10.pdf",
                    "gs://bucket/doc_p11-20.pdf",
                    "gs://bucket/doc_p21-25.pdf",
                ]
                mock_split.return_value = sub_files

                # Create mock result with new fallback fields
                mock_result = MagicMock()
                mock_result.pages = [MagicMock(), MagicMock()]
                mock_result.model_used = "mistral"
                mock_result.fallback_used = False
                mock_result.primary_error = None

                mock_sub_handler.run.return_value = {
                    "success": True,
                    "result": mock_result,
                    "gcs_uri": "gs://bucket/result.json",
                }

                result = orchestrator.run(temp_pdf)

                # Verify all 3 sub-files were processed
                assert mock_sub_handler.run.call_count == 3
                assert result.total_sub_files == 3
                assert result.successful_sub_files == 3
        finally:
            os.unlink(temp_pdf)

    def test_run_aggregates_results(self, mock_orchestrator):
        """Test that results are properly aggregated."""
        orchestrator, mock_sub_handler = mock_orchestrator

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(b"PDF content")
            temp_pdf = f.name

        try:
            with (
                patch("ocr.ocr_orchestrator.is_gcs_uri", return_value=False),
                patch("ocr.ocr_orchestrator.split_pdf_by_pages") as mock_split,
            ):
                sub_files = [
                    "gs://bucket/doc_p1-10.pdf",
                    "gs://bucket/doc_p11-15.pdf",
                ]
                mock_split.return_value = sub_files

                # Create mock results with new fallback fields
                mock_result1 = MagicMock()
                mock_result1.pages = [MagicMock() for _ in range(10)]
                mock_result1.model_used = "mistral"
                mock_result1.fallback_used = False
                mock_result1.primary_error = None

                mock_result2 = MagicMock()
                mock_result2.pages = [MagicMock() for _ in range(5)]
                mock_result2.model_used = "mistral"
                mock_result2.fallback_used = False
                mock_result2.primary_error = None

                # First call: 10 pages, second call: 5 pages
                mock_sub_handler.run.side_effect = [
                    {
                        "success": True,
                        "result": mock_result1,
                        "gcs_uri": "gs://bucket/result1.json",
                    },
                    {
                        "success": True,
                        "result": mock_result2,
                        "gcs_uri": "gs://bucket/result2.json",
                    },
                ]

                result = orchestrator.run(temp_pdf)

                assert result.success
                assert result.total_pages == 15
                assert len(result.extracted_text_uris) == 2
        finally:
            os.unlink(temp_pdf)

    def test_run_handles_partial_failure(self, mock_orchestrator):
        """Test handling of partial sub-file failures."""
        orchestrator, mock_sub_handler = mock_orchestrator

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(b"PDF content")
            temp_pdf = f.name

        try:
            with (
                patch("ocr.ocr_orchestrator.is_gcs_uri", return_value=False),
                patch("ocr.ocr_orchestrator.split_pdf_by_pages") as mock_split,
            ):
                sub_files = [
                    "gs://bucket/doc_p1-10.pdf",
                    "gs://bucket/doc_p11-20.pdf",
                ]
                mock_split.return_value = sub_files

                # Create mock results with new fallback fields
                mock_result_success = MagicMock()
                mock_result_success.pages = [MagicMock() for _ in range(10)]
                mock_result_success.model_used = "mistral"
                mock_result_success.fallback_used = False
                mock_result_success.primary_error = None

                mock_result_failure = MagicMock()
                mock_result_failure.pages = []
                mock_result_failure.model_used = "llm"  # Tried fallback
                mock_result_failure.fallback_used = True
                mock_result_failure.primary_error = "Mistral failed"

                # First succeeds, second fails
                mock_sub_handler.run.side_effect = [
                    {
                        "success": True,
                        "result": mock_result_success,
                        "gcs_uri": "gs://bucket/result1.json",
                    },
                    {
                        "success": False,
                        "result": mock_result_failure,
                        "gcs_uri": None,
                        "error": "OCR failed",
                    },
                ]

                result = orchestrator.run(temp_pdf)

                assert not result.success  # Overall failure
                assert result.successful_sub_files == 1
                assert result.failed_sub_files == 1
                assert result.total_pages == 10
        finally:
            os.unlink(temp_pdf)


# =============================================================================
# Tests: get_ocr_orchestrator Factory Function
# =============================================================================


class TestGetOcrOrchestrator:
    """Tests for get_ocr_orchestrator factory function."""

    def test_get_ocr_orchestrator_returns_instance(self):
        """Test that factory function returns OcrOrchestrator instance."""
        with patch("ocr.ocr_orchestrator.get_sub_file_handler"):
            orchestrator = get_ocr_orchestrator(session_id="test-key")
            assert isinstance(orchestrator, OcrOrchestrator)

    def test_get_ocr_orchestrator_passes_parameters(self):
        """Test that factory function passes all parameters."""
        with patch("ocr.ocr_orchestrator.get_sub_file_handler"):
            orchestrator = get_ocr_orchestrator(
                session_id="test-key",
                model_type="llm",
                size_limit_mb=8.0,
                pages_per_chunk=20,
            )

            assert orchestrator._session_id == "test-key"
            assert orchestrator._model_type == "llm"
            assert orchestrator._size_limit_mb == 8.0
            assert orchestrator._pages_per_chunk == 20


# =============================================================================
# Tests: SubFileProcessingResult Dataclass
# =============================================================================


class TestSubFileProcessingResult:
    """Tests for SubFileProcessingResult dataclass."""

    def test_successful_result(self):
        """Test creating a successful processing result."""
        result = SubFileProcessingResult(
            gcs_uri="gs://bucket/file.pdf",
            success=True,
            extracted_text_uri="gs://bucket/result.json",
            page_count=10,
        )

        assert result.success
        assert result.page_count == 10
        assert result.error is None

    def test_failed_result(self):
        """Test creating a failed processing result."""
        result = SubFileProcessingResult(
            gcs_uri="gs://bucket/file.pdf",
            success=False,
            error="Processing failed",
        )

        assert not result.success
        assert result.page_count == 0
        assert result.error == "Processing failed"


# =============================================================================
# Tests: OcrOrchestrationResult Dataclass
# =============================================================================


class TestOcrOrchestrationResult:
    """Tests for OcrOrchestrationResult dataclass."""

    def test_default_values(self):
        """Test default values of orchestration result."""
        result = OcrOrchestrationResult(
            session_id="test-key",
            source_uri="gs://bucket/source.pdf",
            success=True,
        )

        assert result.total_sub_files == 0
        assert result.successful_sub_files == 0
        assert result.failed_sub_files == 0
        assert result.total_pages == 0
        assert result.sub_file_results == []
        assert result.extracted_text_uris == []
        assert result.error is None


# =============================================================================
# Tests: GCS Folder Detection and Parallel Downloads
# =============================================================================


class TestGcsFolderDetection:
    """Tests for GCS folder detection logic in run()."""

    def test_uri_ending_with_slash_is_folder(
        self,
        mock_sub_file_handler,
        mock_gcs_functions,
        mock_local_directory_handler,
        mock_pdf_splitting,
    ):
        """URI ending with / is treated as folder."""
        mock_gcs_functions["is_gcs_uri"].return_value = True
        mock_gcs_functions["download_folder_files"].return_value = ["/tmp/doc.pdf"]
        mock_pdf_splitting["split_pdf_by_pages"].return_value = ["gs://bucket/tmp/chunk.pdf"]

        mock_result = MagicMock()
        mock_result.pages = []
        mock_sub_file_handler.run.return_value = {
            "success": True,
            "result": mock_result,
            "gcs_uri": "gs://bucket/extracted/doc.json",
        }

        orchestrator = OcrOrchestrator(session_id="session123")
        orchestrator.run("gs://bucket/folder/")

        mock_gcs_functions["download_folder_files"].assert_called_once()
        mock_gcs_functions["download_from_gcs"].assert_not_called()

    def test_uri_without_extension_is_folder(
        self,
        mock_sub_file_handler,
        mock_gcs_functions,
        mock_local_directory_handler,
        mock_pdf_splitting,
    ):
        """URI without file extension is treated as folder."""
        mock_gcs_functions["is_gcs_uri"].return_value = True
        mock_gcs_functions["download_folder_files"].return_value = ["/tmp/doc.pdf"]
        mock_pdf_splitting["split_pdf_by_pages"].return_value = ["gs://bucket/tmp/chunk.pdf"]

        mock_result = MagicMock()
        mock_result.pages = []
        mock_sub_file_handler.run.return_value = {
            "success": True,
            "result": mock_result,
            "gcs_uri": "gs://bucket/extracted/doc.json",
        }

        orchestrator = OcrOrchestrator(session_id="session123")
        orchestrator.run("gs://bucket/folder")

        mock_gcs_functions["download_folder_files"].assert_called_once()

    def test_uri_with_pdf_extension_is_file(
        self,
        mock_sub_file_handler,
        mock_gcs_functions,
        mock_local_directory_handler,
        mock_pdf_splitting,
    ):
        """URI with .pdf extension is treated as single file."""
        mock_gcs_functions["is_gcs_uri"].return_value = True
        mock_gcs_functions["download_from_gcs"].return_value = "/tmp/doc.pdf"
        mock_pdf_splitting["split_pdf_by_pages"].return_value = ["gs://bucket/tmp/chunk.pdf"]

        mock_result = MagicMock()
        mock_result.pages = []
        mock_sub_file_handler.run.return_value = {
            "success": True,
            "result": mock_result,
            "gcs_uri": "gs://bucket/extracted/doc.json",
        }

        orchestrator = OcrOrchestrator(session_id="session123")
        orchestrator.run("gs://bucket/folder/document.pdf")

        mock_gcs_functions["download_from_gcs"].assert_called_once()
        mock_gcs_functions["download_folder_files"].assert_not_called()


class TestParallelFolderDownload:
    """Tests for parallel folder download using download_folder_files."""

    def test_folder_download_uses_correct_parameters(
        self,
        mock_sub_file_handler,
        mock_gcs_functions,
        mock_local_directory_handler,
        mock_pdf_splitting,
    ):
        """Verify download_folder_files is called with correct parameters."""
        mock_gcs_functions["is_gcs_uri"].return_value = True
        mock_gcs_functions["download_folder_files"].return_value = ["/tmp/doc.pdf"]
        mock_pdf_splitting["split_pdf_by_pages"].return_value = ["gs://bucket/tmp/chunk.pdf"]

        mock_result = MagicMock()
        mock_result.pages = []
        mock_result.model_used = "mistral"
        mock_result.fallback_used = False
        mock_result.primary_error = None
        mock_sub_file_handler.run.return_value = {
            "success": True,
            "result": mock_result,
            "gcs_uri": "gs://bucket/extracted/doc.json",
        }

        orchestrator = OcrOrchestrator(session_id="session123", max_workers=8)
        orchestrator.run("gs://bucket/folder/")

        # Verify call arguments
        call_kwargs = mock_gcs_functions["download_folder_files"].call_args
        assert call_kwargs.kwargs["folder_uri"] == "gs://bucket/folder/"
        assert call_kwargs.kwargs["file_extension"] == ".pdf"
        assert call_kwargs.kwargs["max_workers"] == 8

    def test_multiple_pdfs_downloaded_from_folder(
        self,
        mock_sub_file_handler,
        mock_gcs_functions,
        mock_local_directory_handler,
        mock_pdf_splitting,
    ):
        """Multiple PDFs from folder are all processed."""
        mock_gcs_functions["is_gcs_uri"].return_value = True
        mock_gcs_functions["download_folder_files"].return_value = [
            "/tmp/doc1.pdf",
            "/tmp/doc2.pdf",
            "/tmp/doc3.pdf",
        ]
        mock_pdf_splitting["split_pdf_by_pages"].return_value = ["gs://bucket/tmp/chunk.pdf"]

        mock_result = MagicMock()
        mock_result.pages = [MagicMock()]
        mock_result.model_used = "mistral"
        mock_result.fallback_used = False
        mock_result.primary_error = None
        mock_sub_file_handler.run.return_value = {
            "success": True,
            "result": mock_result,
            "gcs_uri": "gs://bucket/extracted/doc.json",
        }

        orchestrator = OcrOrchestrator(session_id="session123")
        result = orchestrator.run("gs://bucket/folder/")

        # All 3 PDFs should be split
        assert mock_pdf_splitting["split_pdf_by_pages"].call_count == 3
        assert result.success

    def test_empty_folder_raises_error(
        self,
        mock_sub_file_handler,
        mock_gcs_functions,
        mock_local_directory_handler,
    ):
        """Empty folder returns error result."""
        mock_gcs_functions["is_gcs_uri"].return_value = True
        mock_gcs_functions["download_folder_files"].return_value = []

        orchestrator = OcrOrchestrator(session_id="session123")
        result = orchestrator.run("gs://bucket/empty_folder/")

        assert result.success is False
        assert "No PDF files found" in result.error


class TestCleanupBehavior:
    """Tests for cleanup behavior using local_directory_handler."""

    def test_cleanup_called_on_success(
        self,
        mock_sub_file_handler,
        mock_gcs_functions,
        mock_local_directory_handler,
        mock_pdf_splitting,
    ):
        """Cleanup is called after successful processing."""
        mock_gcs_functions["is_gcs_uri"].return_value = True
        mock_gcs_functions["download_folder_files"].return_value = ["/tmp/doc.pdf"]
        mock_pdf_splitting["split_pdf_by_pages"].return_value = ["gs://bucket/tmp/chunk.pdf"]

        mock_result = MagicMock()
        mock_result.pages = []
        mock_sub_file_handler.run.return_value = {
            "success": True,
            "result": mock_result,
            "gcs_uri": "gs://bucket/extracted/doc.json",
        }

        orchestrator = OcrOrchestrator(session_id="session123")
        orchestrator.run("gs://bucket/folder/")

        mock_local_directory_handler["cleanup_local_data"].assert_called_once_with("session123")

    def test_cleanup_called_on_error(
        self,
        mock_sub_file_handler,
        mock_gcs_functions,
        mock_local_directory_handler,
    ):
        """Cleanup is called even when processing fails."""
        mock_gcs_functions["is_gcs_uri"].return_value = True
        mock_gcs_functions["download_folder_files"].side_effect = Exception("Download failed")

        orchestrator = OcrOrchestrator(session_id="session123")
        result = orchestrator.run("gs://bucket/folder/")

        assert result.success is False
        # Cleanup should still be called in finally block
        mock_local_directory_handler["cleanup_local_data"].assert_called_once_with("session123")

    def test_local_temp_path_used_for_downloads(
        self,
        mock_sub_file_handler,
        mock_gcs_functions,
        mock_local_directory_handler,
        mock_pdf_splitting,
    ):
        """get_local_temp_path is used to determine download directory."""
        mock_gcs_functions["is_gcs_uri"].return_value = True
        mock_gcs_functions["download_folder_files"].return_value = ["/tmp/doc.pdf"]
        mock_pdf_splitting["split_pdf_by_pages"].return_value = ["gs://bucket/tmp/chunk.pdf"]

        mock_result = MagicMock()
        mock_result.pages = []
        mock_sub_file_handler.run.return_value = {
            "success": True,
            "result": mock_result,
            "gcs_uri": "gs://bucket/extracted/doc.json",
        }

        orchestrator = OcrOrchestrator(session_id="session123")
        orchestrator.run("gs://bucket/folder/")

        mock_local_directory_handler["get_local_temp_path"].assert_called_once_with(
            "session123", "downloads"
        )


class TestSplitPdfBehavior:
    """Tests for _split_pdf method."""

    def test_mistral_model_uses_split_by_pages(
        self,
        mock_sub_file_handler,
        mock_pdf_splitting,
    ):
        """Mistral model splits by page count."""
        mock_pdf_splitting["split_pdf_by_pages"].return_value = [
            "gs://bucket/session/tmp/doc_p1-10.pdf",
            "gs://bucket/session/tmp/doc_p11-20.pdf",
        ]

        orchestrator = OcrOrchestrator(
            session_id="session123",
            model_type="mistral",
            pages_per_chunk=10,
        )

        result = orchestrator._split_pdf("/path/to/doc.pdf")

        mock_pdf_splitting["split_pdf_by_pages"].assert_called_once()
        mock_pdf_splitting["split_pdf_by_size"].assert_not_called()
        assert len(result) == 2

    def test_llm_model_uses_split_by_size(
        self,
        mock_sub_file_handler,
        mock_pdf_splitting,
    ):
        """LLM model splits by file size."""
        mock_pdf_splitting["split_pdf_by_size"].return_value = [
            "gs://bucket/session/tmp/doc_p1-5.pdf",
            "gs://bucket/session/tmp/doc_p6-10.pdf",
        ]

        orchestrator = OcrOrchestrator(
            session_id="session123",
            model_type="llm",
            size_limit_mb=5.0,
        )

        result = orchestrator._split_pdf("/path/to/doc.pdf")

        mock_pdf_splitting["split_pdf_by_size"].assert_called_once()
        mock_pdf_splitting["split_pdf_by_pages"].assert_not_called()
        assert len(result) == 2
