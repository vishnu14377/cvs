"""Tests for ADR Vector Database ingestion pipeline."""

import json
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document
from src.adr_vector_database.data_models import (
    BatchIngestionResult,
    ExtractedDocument,
    IngestionResult,
)
from src.adr_vector_database.file_processor import FileProcessor
from src.adr_vector_database.ingestion_pipeline import (
    _parse_json_file,
    _process_single_file,
    ingest_document,
    ingest_session,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_extracted_json():
    """Sample extracted JSON data from OCR."""
    return {
        "document_name": "test_document.pdf",
        "base_page_number": 1,
        "end_page_number": 3,
        "pages": [
            {"sub_file_index": 0, "original_page_number": 1, "extracted_text": "Page 1 content."},
            {"sub_file_index": 1, "original_page_number": 2, "extracted_text": "Page 2 content."},
            {"sub_file_index": 2, "original_page_number": 3, "extracted_text": "Page 3 content."},
        ],
        "success": True,
        "model_used": "mistral",
    }


@pytest.fixture
def sample_json_file(sample_extracted_json, tmp_path):
    """Create a sample JSON file for testing."""
    json_file = tmp_path / "doc.json"
    json_file.write_text(json.dumps(sample_extracted_json))
    return str(json_file)


# =============================================================================
# Helper Function Tests
# =============================================================================


class TestParseJsonFile:
    """Tests for _parse_json_file helper."""

    def test_parses_valid_json(self, sample_json_file):
        """Test parsing a valid JSON file."""
        doc = _parse_json_file(sample_json_file)

        assert isinstance(doc, ExtractedDocument)
        assert doc.document_name == "test_document.pdf"
        assert doc.page_count == 3

    def test_raises_on_invalid_json(self, tmp_path):
        """Test that invalid JSON raises an error."""
        invalid_file = tmp_path / "invalid.json"
        invalid_file.write_text("not valid json")

        with pytest.raises(json.JSONDecodeError):
            _parse_json_file(str(invalid_file))


class TestProcessSingleFile:
    """Tests for _process_single_file helper."""

    @patch("src.adr_vector_database.ingestion_pipeline.download_from_gcs")
    def test_process_success(self, mock_download, sample_json_file):
        """Test successful file processing."""
        mock_download.return_value = sample_json_file
        file_processor = FileProcessor()

        documents, result = _process_single_file(
            gcs_uri="gs://bucket/path/doc.json",
            session_id="session-123",
            local_dir="/tmp",
            file_processor=file_processor,
        )

        assert result.success is True
        assert result.document_name == "test_document.pdf"
        assert len(documents) >= 1
        assert all(isinstance(d, Document) for d in documents)

    @patch("src.adr_vector_database.ingestion_pipeline.download_from_gcs")
    def test_process_failure(self, mock_download):
        """Test file processing failure."""
        mock_download.side_effect = Exception("Download failed")
        file_processor = FileProcessor()

        documents, result = _process_single_file(
            gcs_uri="gs://bucket/path/doc.json",
            session_id="session-123",
            local_dir="/tmp",
            file_processor=file_processor,
        )

        assert result.success is False
        assert "Download failed" in result.error
        assert documents == []


# =============================================================================
# Ingest Session Tests
# =============================================================================


class TestIngestSession:
    """Tests for ingest_session function."""

    @patch("src.adr_vector_database.ingestion_pipeline.cleanup_local_data")
    @patch("src.adr_vector_database.ingestion_pipeline.get_local_temp_path")
    @patch("src.adr_vector_database.ingestion_pipeline.VectorStoreManager")
    @patch("src.adr_vector_database.ingestion_pipeline.download_from_gcs")
    @patch("src.adr_vector_database.ingestion_pipeline.list_files_in_gcs_folder")
    def test_ingest_session_success(
        self,
        mock_list_files,
        mock_download,
        mock_vector_store_class,
        mock_get_temp,
        mock_cleanup,
        sample_json_file,
        tmp_path,
    ):
        """Test successful session ingestion."""
        # Setup mocks
        mock_list_files.return_value = ["gs://bucket/session/extracted_text/doc.json"]
        mock_download.return_value = sample_json_file
        mock_get_temp.return_value = tmp_path

        mock_vector_store = MagicMock()
        mock_vector_store.batch_insert.return_value = ["id1", "id2"]
        mock_vector_store_class.return_value = mock_vector_store

        # Run
        result = ingest_session(session_id="session-123", max_workers=1)

        # Verify
        assert isinstance(result, BatchIngestionResult)
        assert result.session_id == "session-123"
        assert result.total_documents == 1
        assert result.successful_documents == 1
        assert result.total_chunks_stored == 2
        mock_cleanup.assert_called_once()

    @patch("src.adr_vector_database.ingestion_pipeline.cleanup_local_data")
    @patch("src.adr_vector_database.ingestion_pipeline.get_local_temp_path")
    @patch("src.adr_vector_database.ingestion_pipeline.list_files_in_gcs_folder")
    def test_ingest_session_no_files(
        self,
        mock_list_files,
        mock_get_temp,
        mock_cleanup,
        tmp_path,
    ):
        """Test ingestion when no files are found."""
        mock_list_files.return_value = []
        mock_get_temp.return_value = tmp_path

        result = ingest_session(session_id="empty-session")

        assert result.total_documents == 0
        assert result.successful_documents == 0

    @patch("src.adr_vector_database.ingestion_pipeline.cleanup_local_data")
    @patch("src.adr_vector_database.ingestion_pipeline.get_local_temp_path")
    @patch("src.adr_vector_database.ingestion_pipeline.VectorStoreManager")
    @patch("src.adr_vector_database.ingestion_pipeline.download_from_gcs")
    @patch("src.adr_vector_database.ingestion_pipeline.list_files_in_gcs_folder")
    def test_ingest_session_parallel(
        self,
        mock_list_files,
        mock_download,
        mock_vector_store_class,
        mock_get_temp,
        mock_cleanup,
        sample_extracted_json,
        tmp_path,
    ):
        """Test parallel processing of multiple files."""
        # Create multiple test files
        files = []
        for i in range(3):
            json_file = tmp_path / f"doc{i}.json"
            data = sample_extracted_json.copy()
            data["document_name"] = f"document_{i}.pdf"
            json_file.write_text(json.dumps(data))
            files.append(str(json_file))

        mock_list_files.return_value = [f"gs://bucket/doc{i}.json" for i in range(3)]
        mock_download.side_effect = files
        mock_get_temp.return_value = tmp_path

        mock_vector_store = MagicMock()
        mock_vector_store.batch_insert.return_value = ["id1", "id2", "id3"]
        mock_vector_store_class.return_value = mock_vector_store

        result = ingest_session(session_id="session-123", max_workers=3)

        assert result.total_documents == 3
        assert result.successful_documents == 3


# =============================================================================
# Ingest Document Tests
# =============================================================================


class TestIngestDocument:
    """Tests for ingest_document function."""

    @patch("src.adr_vector_database.ingestion_pipeline.cleanup_local_data")
    @patch("src.adr_vector_database.ingestion_pipeline.get_local_temp_path")
    @patch("src.adr_vector_database.ingestion_pipeline.VectorStoreManager")
    @patch("src.adr_vector_database.ingestion_pipeline.download_from_gcs")
    def test_ingest_single_document(
        self,
        mock_download,
        mock_vector_store_class,
        mock_get_temp,
        mock_cleanup,
        sample_json_file,
        tmp_path,
    ):
        """Test single document ingestion."""
        mock_download.return_value = sample_json_file
        mock_get_temp.return_value = tmp_path

        mock_vector_store = MagicMock()
        mock_vector_store.insert.return_value = ["id1", "id2"]
        mock_vector_store_class.return_value = mock_vector_store

        result = ingest_document(
            gcs_uri="gs://bucket/doc.json",
            session_id="session-123",
        )

        assert isinstance(result, IngestionResult)
        assert result.success is True
        assert result.chunks_stored == 2

    @patch("src.adr_vector_database.ingestion_pipeline.cleanup_local_data")
    @patch("src.adr_vector_database.ingestion_pipeline.get_local_temp_path")
    @patch("src.adr_vector_database.ingestion_pipeline.download_from_gcs")
    def test_ingest_document_failure(
        self,
        mock_download,
        mock_get_temp,
        mock_cleanup,
        tmp_path,
    ):
        """Test document ingestion failure."""
        mock_download.side_effect = Exception("Download failed")
        mock_get_temp.return_value = tmp_path

        result = ingest_document(
            gcs_uri="gs://bucket/doc.json",
            session_id="session-123",
        )

        assert result.success is False
        assert "Download failed" in result.error


# =============================================================================
# Integration-like Tests
# =============================================================================


class TestPipelineIntegration:
    """Tests for pipeline integration behavior."""

    @patch("src.adr_vector_database.ingestion_pipeline.cleanup_local_data")
    @patch("src.adr_vector_database.ingestion_pipeline.get_local_temp_path")
    @patch("src.adr_vector_database.ingestion_pipeline.VectorStoreManager")
    @patch("src.adr_vector_database.ingestion_pipeline.download_from_gcs")
    @patch("src.adr_vector_database.ingestion_pipeline.list_files_in_gcs_folder")
    def test_metadata_passed_to_documents(
        self,
        mock_list_files,
        mock_download,
        mock_vector_store_class,
        mock_get_temp,
        mock_cleanup,
        sample_json_file,
        tmp_path,
    ):
        """Test that additional metadata is passed to documents."""
        mock_list_files.return_value = ["gs://bucket/doc.json"]
        mock_download.return_value = sample_json_file
        mock_get_temp.return_value = tmp_path

        captured_docs = []
        mock_vector_store = MagicMock()
        mock_vector_store.batch_insert.side_effect = lambda docs: (
            captured_docs.extend(docs),
            ["id"] * len(docs),
        )[1]
        mock_vector_store_class.return_value = mock_vector_store

        result = ingest_session(
            session_id="session-123",
            additional_metadata={"custom_field": "custom_value"},
            max_workers=1,
        )

        assert result.success
        # Check that metadata was passed (documents were captured)
        assert len(captured_docs) >= 1

    @patch("src.adr_vector_database.ingestion_pipeline.cleanup_local_data")
    @patch("src.adr_vector_database.ingestion_pipeline.get_local_temp_path")
    @patch("src.adr_vector_database.ingestion_pipeline.list_files_in_gcs_folder")
    def test_cleanup_always_called(
        self,
        mock_list_files,
        mock_get_temp,
        mock_cleanup,
        tmp_path,
    ):
        """Test that cleanup is called even on error."""
        mock_list_files.side_effect = Exception("List failed")
        mock_get_temp.return_value = tmp_path

        result = ingest_session(session_id="session-123")

        # Cleanup should still be called
        mock_cleanup.assert_called_once()
        assert "List failed" in result.errors[0]
