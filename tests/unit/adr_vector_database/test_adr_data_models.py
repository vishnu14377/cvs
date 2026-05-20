"""Tests for ADR Vector Database data models."""

from src.adr_vector_database.data_models import (
    BatchIngestionResult,
    DocumentChunk,
    ExtractedDocument,
    ExtractedPage,
    IngestionResult,
)


class TestExtractedPage:
    """Tests for ExtractedPage model."""

    def test_create_extracted_page(self):
        """Test creating an ExtractedPage."""
        page = ExtractedPage(
            sub_file_index=0, original_page_number=1, extracted_text="Test content"
        )
        assert page.sub_file_index == 0
        assert page.original_page_number == 1
        assert page.extracted_text == "Test content"

    def test_extracted_page_default_text(self):
        """Test ExtractedPage with default empty text."""
        page = ExtractedPage(sub_file_index=0, original_page_number=1)
        assert page.extracted_text == ""

    def test_extracted_page_from_dict(self):
        """Test creating ExtractedPage from dictionary."""
        data = {"sub_file_index": 2, "original_page_number": 5, "extracted_text": "Page content"}
        page = ExtractedPage(**data)
        assert page.sub_file_index == 2
        assert page.original_page_number == 5
        assert page.extracted_text == "Page content"


class TestExtractedDocument:
    """Tests for ExtractedDocument model."""

    def test_create_extracted_document(self):
        """Test creating an ExtractedDocument."""
        pages = [
            ExtractedPage(sub_file_index=0, original_page_number=1, extracted_text="Page 1"),
            ExtractedPage(sub_file_index=1, original_page_number=2, extracted_text="Page 2"),
        ]
        doc = ExtractedDocument(
            document_name="test.pdf", base_page_number=1, end_page_number=2, pages=pages
        )
        assert doc.document_name == "test.pdf"
        assert doc.base_page_number == 1
        assert doc.end_page_number == 2
        assert len(doc.pages) == 2

    def test_page_count_property(self):
        """Test page_count property."""
        pages = [
            ExtractedPage(
                sub_file_index=i, original_page_number=i + 1, extracted_text=f"Page {i + 1}"
            )
            for i in range(5)
        ]
        doc = ExtractedDocument(
            document_name="test.pdf", base_page_number=1, end_page_number=5, pages=pages
        )
        assert doc.page_count == 5

    def test_get_combined_text(self):
        """Test get_combined_text method."""
        pages = [
            ExtractedPage(sub_file_index=0, original_page_number=1, extracted_text="First"),
            ExtractedPage(sub_file_index=1, original_page_number=2, extracted_text="Second"),
        ]
        doc = ExtractedDocument(
            document_name="test.pdf", base_page_number=1, end_page_number=2, pages=pages
        )
        combined = doc.get_combined_text()
        assert combined == "First\n\nSecond"

    def test_get_combined_text_custom_separator(self):
        """Test get_combined_text with custom separator."""
        pages = [
            ExtractedPage(sub_file_index=0, original_page_number=1, extracted_text="A"),
            ExtractedPage(sub_file_index=1, original_page_number=2, extracted_text="B"),
        ]
        doc = ExtractedDocument(
            document_name="test.pdf", base_page_number=1, end_page_number=2, pages=pages
        )
        combined = doc.get_combined_text(separator=" | ")
        assert combined == "A | B"

    def test_get_combined_text_skips_empty(self):
        """Test that get_combined_text skips empty pages."""
        pages = [
            ExtractedPage(sub_file_index=0, original_page_number=1, extracted_text="Content"),
            ExtractedPage(sub_file_index=1, original_page_number=2, extracted_text=""),
            ExtractedPage(sub_file_index=2, original_page_number=3, extracted_text="More"),
        ]
        doc = ExtractedDocument(
            document_name="test.pdf", base_page_number=1, end_page_number=3, pages=pages
        )
        combined = doc.get_combined_text()
        assert combined == "Content\n\nMore"

    def test_from_dict(self):
        """Test creating ExtractedDocument from dictionary."""
        data = {
            "document_name": "sample.pdf",
            "base_page_number": 10,
            "end_page_number": 15,
            "pages": [
                {"sub_file_index": 0, "original_page_number": 10, "extracted_text": "Page 10"},
                {"sub_file_index": 1, "original_page_number": 11, "extracted_text": "Page 11"},
            ],
            "success": True,
            "model_used": "mistral",
        }
        doc = ExtractedDocument.from_dict(data)
        assert doc.document_name == "sample.pdf"
        assert doc.base_page_number == 10
        assert doc.end_page_number == 15
        assert len(doc.pages) == 2
        assert doc.success is True
        assert doc.model_used == "mistral"

    def test_from_dict_with_defaults(self):
        """Test from_dict with missing optional fields."""
        data = {
            "pages": [{"sub_file_index": 0, "original_page_number": 1, "extracted_text": "Content"}]
        }
        doc = ExtractedDocument.from_dict(data)
        assert doc.document_name == "unknown"
        assert doc.base_page_number == 1
        assert doc.end_page_number == 1


class TestDocumentChunk:
    """Tests for DocumentChunk dataclass."""

    def test_create_document_chunk(self):
        """Test creating a DocumentChunk."""
        chunk = DocumentChunk(
            text="Chunk content",
            document_name="test.pdf",
            page_numbers=[1, 2],
            chunk_index=0,
            session_id="session-123",
        )
        assert chunk.text == "Chunk content"
        assert chunk.document_name == "test.pdf"
        assert chunk.page_numbers == [1, 2]
        assert chunk.chunk_index == 0
        assert chunk.session_id == "session-123"

    def test_to_metadata_dict(self):
        """Test to_metadata_dict method."""
        chunk = DocumentChunk(
            text="Content",
            document_name="doc.pdf",
            page_numbers=[5, 6, 7],
            chunk_index=3,
            session_id="sess-456",
            gcs_source_uri="gs://bucket/path/doc.json",
            metadata={"custom": "value"},
        )
        meta = chunk.to_metadata_dict()

        assert meta["document_name"] == "doc.pdf"
        assert meta["page_numbers"] == [5, 6, 7]
        assert meta["chunk_index"] == 3
        assert meta["session_id"] == "sess-456"
        assert meta["gcs_source_uri"] == "gs://bucket/path/doc.json"
        assert meta["page_start"] == 5
        assert meta["page_end"] == 7
        assert meta["custom"] == "value"

    def test_to_metadata_dict_empty_pages(self):
        """Test to_metadata_dict with empty page_numbers."""
        chunk = DocumentChunk(
            text="Content",
            document_name="doc.pdf",
            page_numbers=[],
            chunk_index=0,
            session_id="sess",
        )
        meta = chunk.to_metadata_dict()
        assert meta["page_start"] == 0
        assert meta["page_end"] == 0


class TestIngestionResult:
    """Tests for IngestionResult dataclass."""

    def test_create_ingestion_result(self):
        """Test creating an IngestionResult."""
        result = IngestionResult(
            document_name="test.pdf",
            session_id="session-123",
            success=True,
            chunks_created=10,
            chunks_stored=10,
            vector_ids=["id1", "id2"],
        )
        assert result.document_name == "test.pdf"
        assert result.success is True
        assert result.chunks_created == 10
        assert result.chunks_stored == 10

    def test_to_dict(self):
        """Test to_dict method."""
        result = IngestionResult(
            document_name="doc.pdf",
            session_id="sess",
            success=True,
            chunks_created=5,
            chunks_stored=5,
            vector_ids=["a", "b", "c", "d", "e"],
            gcs_source_uri="gs://bucket/doc.json",
        )
        d = result.to_dict()

        assert d["document_name"] == "doc.pdf"
        assert d["session_id"] == "sess"
        assert d["success"] is True
        assert d["chunks_created"] == 5
        assert d["chunks_stored"] == 5
        assert len(d["vector_ids"]) == 5

    def test_failed_result(self):
        """Test creating a failed IngestionResult."""
        result = IngestionResult(
            document_name="failed.pdf",
            session_id="sess",
            success=False,
            error="Connection failed",
        )
        assert result.success is False
        assert result.error == "Connection failed"
        assert result.chunks_stored == 0


class TestBatchIngestionResult:
    """Tests for BatchIngestionResult dataclass."""

    def test_create_batch_result(self):
        """Test creating a BatchIngestionResult."""
        result = BatchIngestionResult(
            session_id="session-123",
            total_documents=5,
            successful_documents=4,
            failed_documents=1,
            total_chunks_stored=100,
        )
        assert result.session_id == "session-123"
        assert result.total_documents == 5
        assert result.successful_documents == 4
        assert result.failed_documents == 1
        assert result.total_chunks_stored == 100

    def test_success_property_all_successful(self):
        """Test success property when all documents succeed."""
        result = BatchIngestionResult(
            session_id="sess",
            total_documents=3,
            successful_documents=3,
            failed_documents=0,
        )
        assert result.success is True

    def test_success_property_with_failures(self):
        """Test success property when some documents fail."""
        result = BatchIngestionResult(
            session_id="sess",
            total_documents=3,
            successful_documents=2,
            failed_documents=1,
        )
        assert result.success is False

    def test_to_dict(self):
        """Test to_dict method."""
        individual = IngestionResult(
            document_name="doc1.pdf",
            session_id="sess",
            success=True,
            chunks_stored=5,
        )
        result = BatchIngestionResult(
            session_id="sess",
            total_documents=1,
            successful_documents=1,
            total_chunks_stored=5,
            results=[individual],
        )
        d = result.to_dict()

        assert d["session_id"] == "sess"
        assert d["total_documents"] == 1
        assert d["success"] is True
        assert len(d["results"]) == 1
