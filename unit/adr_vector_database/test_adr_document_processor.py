"""
Unit tests for ADR Document Processor.

Tests the AdrDocumentProcessor class and process_adr_document function
that orchestrate OCR processing and vector database ingestion.
"""

from unittest.mock import MagicMock, patch

import pytest
from src.adr_document_processor import (
    AdrDocumentProcessor,
    AdrProcessingResult,
    process_adr_document,
)
from src.adr_vector_database.data_models import (
    BatchIngestionResult,
)
from src.ocr.data_models.orchestrator_models import (
    FallbackStats,
    OcrOrchestrationResult,
)


class TestAdrProcessingResult:
    """Tests for AdrProcessingResult dataclass."""

    def test_default_values(self):
        """Test default values are set correctly."""
        result = AdrProcessingResult(
            session_id="test-session",
            source_uri="gs://bucket/file.pdf",
        )

        assert result.session_id == "test-session"
        assert result.source_uri == "gs://bucket/file.pdf"
        assert result.success is False
        assert result.ocr_success is False
        assert result.ocr_total_pages == 0
        assert result.ingestion_success is False
        assert result.ingestion_total_chunks == 0

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = AdrProcessingResult(
            session_id="test-session",
            source_uri="gs://bucket/file.pdf",
            success=True,
            ocr_success=True,
            ocr_total_pages=10,
            ingestion_success=True,
            ingestion_total_chunks=25,
        )

        result_dict = result.to_dict()

        assert result_dict["session_id"] == "test-session"
        assert result_dict["source_uri"] == "gs://bucket/file.pdf"
        assert result_dict["success"] is True
        assert result_dict["ocr"]["success"] is True
        assert result_dict["ocr"]["total_pages"] == 10
        assert result_dict["ingestion"]["success"] is True
        assert result_dict["ingestion"]["total_chunks"] == 25

    def test_to_dict_with_fallback_stats(self):
        """Test to_dict with fallback stats."""
        fallback_stats = FallbackStats(
            total_processed=5,
            primary_success_count=3,
            fallback_success_count=2,
        )

        result = AdrProcessingResult(
            session_id="test-session",
            source_uri="gs://bucket/file.pdf",
            ocr_fallback_stats=fallback_stats,
        )

        result_dict = result.to_dict()

        assert result_dict["ocr"]["fallback_stats"] is not None
        assert result_dict["ocr"]["fallback_stats"]["total_processed"] == 5


class TestAdrDocumentProcessor:
    """Tests for AdrDocumentProcessor class."""

    def test_init_valid_session_id(self):
        """Test initialization with valid session ID."""
        processor = AdrDocumentProcessor(session_id="test-123")

        assert processor.session_id == "test-123"

    def test_init_empty_session_id_raises(self):
        """Test initialization with empty session ID raises error."""
        with pytest.raises(ValueError, match="session_id must not be empty"):
            AdrDocumentProcessor(session_id="")

    def test_init_session_id_with_slashes_raises(self):
        """Test initialization with slashes in session ID raises error."""
        with pytest.raises(ValueError, match="session_id must not contain slashes"):
            AdrDocumentProcessor(session_id="test/123")

    def test_init_session_id_with_dotdot_raises(self):
        """Test initialization with '..' in session ID raises error."""
        with pytest.raises(ValueError, match="session_id must not contain slashes or '..'"):
            AdrDocumentProcessor(session_id="test..123")

    def test_init_strips_whitespace(self):
        """Test session ID whitespace is stripped."""
        processor = AdrDocumentProcessor(session_id="  test-123  ")

        assert processor.session_id == "test-123"

    @patch("src.adr_document_processor.OcrOrchestrator")
    @patch("src.adr_document_processor.ingest_session")
    def test_process_full_pipeline_success(self, mock_ingest, mock_orchestrator_class):
        """Test successful full pipeline processing."""
        # Setup OCR mock
        mock_ocr_result = OcrOrchestrationResult(
            session_id="test-session",
            source_uri="gs://bucket/file.pdf",
            success=True,
            total_sub_files=2,
            successful_sub_files=2,
            failed_sub_files=0,
            total_pages=10,
            extracted_text_uris=[
                "gs://bucket/extracted/file1.json",
                "gs://bucket/extracted/file2.json",
            ],
        )
        mock_orchestrator = MagicMock()
        mock_orchestrator.run.return_value = mock_ocr_result
        mock_orchestrator_class.return_value = mock_orchestrator

        # Setup ingestion mock
        mock_ingestion_result = BatchIngestionResult(
            session_id="test-session",
            total_documents=2,
            successful_documents=2,
            failed_documents=0,
            total_chunks_stored=25,
        )
        mock_ingest.return_value = mock_ingestion_result

        # Run processor
        processor = AdrDocumentProcessor(session_id="test-session")
        result = processor.process(gcs_uri="gs://bucket/file.pdf")

        # Verify result
        assert result.success is True
        assert result.ocr_success is True
        assert result.ocr_total_pages == 10
        assert result.ingestion_success is True
        assert result.ingestion_total_chunks == 25

    @patch("src.adr_document_processor.OcrOrchestrator")
    @patch("src.adr_document_processor.ingest_session")
    def test_process_ocr_failure_skips_ingestion(self, mock_ingest, mock_orchestrator_class):
        """Test that OCR failure with no successful sub-files skips ingestion."""
        # Setup OCR mock with complete failure
        mock_ocr_result = OcrOrchestrationResult(
            session_id="test-session",
            source_uri="gs://bucket/file.pdf",
            success=False,
            total_sub_files=2,
            successful_sub_files=0,
            failed_sub_files=2,
            total_pages=0,
            error="OCR failed",
        )
        mock_orchestrator = MagicMock()
        mock_orchestrator.run.return_value = mock_ocr_result
        mock_orchestrator_class.return_value = mock_orchestrator

        # Run processor
        processor = AdrDocumentProcessor(session_id="test-session")
        result = processor.process(gcs_uri="gs://bucket/file.pdf")

        # Verify result
        assert result.success is False
        assert result.ocr_success is False
        assert "OCR failed" in result.error

        # Verify ingestion was not called
        mock_ingest.assert_not_called()

    @patch("src.adr_document_processor.OcrOrchestrator")
    @patch("src.adr_document_processor.ingest_session")
    def test_process_skip_ocr(self, mock_ingest, mock_orchestrator_class):
        """Test processing with skip_ocr=True."""
        # Setup ingestion mock
        mock_ingestion_result = BatchIngestionResult(
            session_id="test-session",
            total_documents=2,
            successful_documents=2,
            total_chunks_stored=25,
        )
        mock_ingest.return_value = mock_ingestion_result

        # Run processor
        processor = AdrDocumentProcessor(session_id="test-session")
        result = processor.process(gcs_uri="gs://bucket/file.pdf", skip_ocr=True)

        # Verify OCR was not called
        mock_orchestrator_class.assert_not_called()

        # Verify ingestion was called
        mock_ingest.assert_called_once()

        # Verify result
        assert result.success is True
        assert result.ocr_success is True  # Assumed successful when skipped
        assert result.ingestion_success is True

    @patch("src.adr_document_processor.OcrOrchestrator")
    @patch("src.adr_document_processor.ingest_session")
    def test_process_skip_ingestion(self, mock_ingest, mock_orchestrator_class):
        """Test processing with skip_ingestion=True."""
        # Setup OCR mock
        mock_ocr_result = OcrOrchestrationResult(
            session_id="test-session",
            source_uri="gs://bucket/file.pdf",
            success=True,
            total_sub_files=2,
            successful_sub_files=2,
            total_pages=10,
        )
        mock_orchestrator = MagicMock()
        mock_orchestrator.run.return_value = mock_ocr_result
        mock_orchestrator_class.return_value = mock_orchestrator

        # Run processor
        processor = AdrDocumentProcessor(session_id="test-session")
        result = processor.process(gcs_uri="gs://bucket/file.pdf", skip_ingestion=True)

        # Verify OCR was called
        mock_orchestrator_class.assert_called_once()

        # Verify ingestion was not called
        mock_ingest.assert_not_called()

        # Verify result
        assert result.success is True
        assert result.ocr_success is True
        assert result.ingestion_success is True  # Assumed successful when skipped

    @patch("src.adr_document_processor.OcrOrchestrator")
    @patch("src.adr_document_processor.ingest_session")
    def test_process_handles_exception(self, mock_ingest, mock_orchestrator_class):
        """Test that exceptions are handled gracefully."""
        # Setup OCR mock to raise exception
        mock_orchestrator = MagicMock()
        mock_orchestrator.run.side_effect = Exception("Unexpected error")
        mock_orchestrator_class.return_value = mock_orchestrator

        # Run processor
        processor = AdrDocumentProcessor(session_id="test-session")
        result = processor.process(gcs_uri="gs://bucket/file.pdf")

        # Verify result
        assert result.success is False
        assert result.error == "Unexpected error"


class TestProcessAdrDocumentFunction:
    """Tests for the process_adr_document convenience function."""

    @patch("src.adr_document_processor.AdrDocumentProcessor")
    def test_creates_processor_with_args(self, mock_processor_class):
        """Test that function creates processor with correct arguments."""
        mock_processor = MagicMock()
        mock_result = AdrProcessingResult(
            session_id="test-session",
            source_uri="gs://bucket/file.pdf",
            success=True,
        )
        mock_processor.process.return_value = mock_result
        mock_processor_class.return_value = mock_processor

        result = process_adr_document(
            session_id="test-session",
            gcs_uri="gs://bucket/file.pdf",
            model_type="mistral",
            max_workers=8,
            collection_name="test_collection",
            chunk_size=1000,
        )

        # Verify processor was created with correct args
        mock_processor_class.assert_called_once_with(
            session_id="test-session",
            model_type="mistral",
            size_limit_mb=5.0,
            pages_per_chunk=None,
            max_workers=8,
            collection_name="test_collection",
            chunk_size=1000,
            chunk_overlap=None,
            batch_size=None,
        )

        # Verify process was called with correct args
        mock_processor.process.assert_called_once_with(
            gcs_uri="gs://bucket/file.pdf",
            timeout=None,
            additional_metadata=None,
            skip_ocr=False,
            skip_ingestion=False,
        )

        assert result == mock_result

    @patch("src.adr_document_processor.AdrDocumentProcessor")
    def test_passes_skip_flags(self, mock_processor_class):
        """Test that skip flags are passed correctly."""
        mock_processor = MagicMock()
        mock_result = AdrProcessingResult(
            session_id="test-session",
            source_uri="gs://bucket/file.pdf",
        )
        mock_processor.process.return_value = mock_result
        mock_processor_class.return_value = mock_processor

        process_adr_document(
            session_id="test-session",
            gcs_uri="gs://bucket/file.pdf",
            skip_ocr=True,
            skip_ingestion=True,
        )

        # Verify process was called with skip flags
        mock_processor.process.assert_called_once_with(
            gcs_uri="gs://bucket/file.pdf",
            timeout=None,
            additional_metadata=None,
            skip_ocr=True,
            skip_ingestion=True,
        )
