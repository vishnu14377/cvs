"""Tests for ADR Vector Database file processor."""

import pytest
from langchain_core.documents import Document
from src.adr_vector_database.data_models import (
    DocumentChunk,
    ExtractedDocument,
    ExtractedPage,
)
from src.adr_vector_database.file_processor import (
    FileProcessor,
    get_file_processor,
    process_extracted_document,
)
from src.core.config import vectorstore_config

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_extracted_doc():
    """Sample extracted document for testing."""
    return ExtractedDocument(
        document_name="test_document.pdf",
        base_page_number=1,
        end_page_number=3,
        pages=[
            ExtractedPage(
                sub_file_index=0,
                original_page_number=1,
                extracted_text="Page 1 content with some medical information about the patient.",
            ),
            ExtractedPage(
                sub_file_index=1,
                original_page_number=2,
                extracted_text="Page 2 content with diagnosis and treatment details.",
            ),
            ExtractedPage(
                sub_file_index=2,
                original_page_number=3,
                extracted_text="Page 3 content with follow-up instructions.",
            ),
        ],
        model_used="mistral",
    )


# =============================================================================
# FileProcessor Initialization Tests
# =============================================================================


class TestFileProcessorInit:
    """Tests for FileProcessor initialization."""

    def test_default_initialization(self):
        """Test processor with default parameters."""
        processor = FileProcessor()
        assert processor.chunk_size == vectorstore_config.DEFAULT_CHUNK_SIZE
        assert processor.chunk_overlap == vectorstore_config.DEFAULT_CHUNK_OVERLAP

    def test_custom_parameters(self):
        """Test processor with custom parameters."""
        processor = FileProcessor(
            chunk_size=500,
            chunk_overlap=100,
        )
        assert processor.chunk_size == 500
        assert processor.chunk_overlap == 100

    def test_chunker_initialized(self):
        """Test that DocumentChunker is initialized."""
        processor = FileProcessor(chunk_size=1000, chunk_overlap=200)
        assert processor._chunker is not None
        assert processor._chunker.chunk_size == 1000
        assert processor._chunker.chunk_overlap == 200


# =============================================================================
# Chunking Tests
# =============================================================================


class TestChunkDocument:
    """Tests for _chunk_document method."""

    def test_chunk_document_returns_chunks(self, sample_extracted_doc):
        """Test that chunking returns DocumentChunk objects."""
        processor = FileProcessor()
        chunks = processor._chunk_document(
            extracted_doc=sample_extracted_doc,
            session_id="session-123",
            source_uri="gs://bucket/doc.json",
        )

        assert len(chunks) >= 1
        assert all(isinstance(c, DocumentChunk) for c in chunks)

    def test_chunk_document_includes_metadata(self, sample_extracted_doc):
        """Test that chunks include proper metadata."""
        processor = FileProcessor()
        chunks = processor._chunk_document(
            extracted_doc=sample_extracted_doc,
            session_id="session-123",
            source_uri="gs://bucket/doc.json",
        )

        for chunk in chunks:
            assert chunk.session_id == "session-123"
            assert chunk.document_name == "test_document.pdf"
            assert chunk.gcs_source_uri == "gs://bucket/doc.json"

    def test_chunk_document_with_additional_metadata(self, sample_extracted_doc):
        """Test that additional metadata is passed to chunks."""
        processor = FileProcessor()
        chunks = processor._chunk_document(
            extracted_doc=sample_extracted_doc,
            session_id="session-123",
            additional_metadata={"custom_field": "custom_value"},
        )

        for chunk in chunks:
            assert chunk.metadata.get("custom_field") == "custom_value"


# =============================================================================
# Conversion Tests
# =============================================================================


class TestConvertChunksToLangchain:
    """Tests for _convert_chunks_to_langchain method."""

    def test_converts_to_langchain_documents(self):
        """Test conversion to LangChain Documents."""
        processor = FileProcessor()
        chunks = [
            DocumentChunk(
                text="Test content",
                document_name="doc.pdf",
                page_numbers=[1],
                chunk_index=0,
                session_id="sess-123",
                gcs_source_uri="gs://bucket/doc.json",
            ),
        ]

        docs = processor._convert_chunks_to_langchain(chunks, model_used="mistral")

        assert len(docs) == 1
        assert isinstance(docs[0], Document)
        assert docs[0].page_content == "Test content"

    def test_includes_all_metadata(self):
        """Test that all metadata is included in conversion."""
        processor = FileProcessor()
        chunks = [
            DocumentChunk(
                text="Test content",
                document_name="doc.pdf",
                page_numbers=[1, 2],
                chunk_index=0,
                session_id="sess-123",
                gcs_source_uri="gs://bucket/doc.json",
                metadata={"extra": "data"},
            ),
        ]

        docs = processor._convert_chunks_to_langchain(chunks, model_used="mistral")

        metadata = docs[0].metadata
        assert metadata["session_id"] == "sess-123"
        assert metadata["document_name"] == "doc.pdf"
        assert metadata["page_numbers"] == [1, 2]
        assert metadata["chunk_index"] == 0
        assert metadata["source"] == "gs://bucket/doc.json"
        assert metadata["model_used"] == "mistral"
        assert metadata["extra"] == "data"


# =============================================================================
# Process File Tests
# =============================================================================


class TestProcessFile:
    """Tests for process_file method."""

    def test_process_file_returns_documents(self, sample_extracted_doc):
        """Test that process_file returns LangChain Documents."""
        processor = FileProcessor()
        documents = processor.process_file(
            extracted_doc=sample_extracted_doc,
            session_id="session-123",
            source_uri="gs://bucket/doc.json",
        )

        assert len(documents) >= 1
        assert all(isinstance(doc, Document) for doc in documents)

    def test_process_file_includes_metadata(self, sample_extracted_doc):
        """Test that returned documents include proper metadata."""
        processor = FileProcessor()
        documents = processor.process_file(
            extracted_doc=sample_extracted_doc,
            session_id="session-123",
            source_uri="gs://bucket/doc.json",
        )

        for doc in documents:
            assert doc.metadata["session_id"] == "session-123"
            assert doc.metadata["document_name"] == "test_document.pdf"
            assert doc.metadata["source"] == "gs://bucket/doc.json"
            assert doc.metadata["model_used"] == "mistral"

    def test_process_file_with_additional_metadata(self, sample_extracted_doc):
        """Test that additional metadata is passed through."""
        processor = FileProcessor()
        documents = processor.process_file(
            extracted_doc=sample_extracted_doc,
            session_id="session-123",
            additional_metadata={"priority": "high"},
        )

        for doc in documents:
            assert doc.metadata.get("priority") == "high"

    def test_process_file_empty_document(self):
        """Test processing a document with no content."""
        empty_doc = ExtractedDocument(
            document_name="empty.pdf",
            base_page_number=1,
            end_page_number=1,
            pages=[],
        )

        processor = FileProcessor()
        documents = processor.process_file(
            extracted_doc=empty_doc,
            session_id="session-123",
        )

        assert documents == []

    def test_process_file_long_content_creates_multiple_chunks(self):
        """Test that long content is split into multiple chunks."""
        long_text = "This is a test sentence. " * 200  # ~5000 chars
        long_doc = ExtractedDocument(
            document_name="long.pdf",
            base_page_number=1,
            end_page_number=1,
            pages=[
                ExtractedPage(
                    sub_file_index=0,
                    original_page_number=1,
                    extracted_text=long_text,
                ),
            ],
        )

        processor = FileProcessor(chunk_size=500, chunk_overlap=50)
        documents = processor.process_file(
            extracted_doc=long_doc,
            session_id="session-123",
        )

        # Should create multiple chunks
        assert len(documents) > 1

        # Each chunk should have sequential chunk_index
        for i, doc in enumerate(documents):
            assert doc.metadata["chunk_index"] == i


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_process_extracted_document(self, sample_extracted_doc):
        """Test process_extracted_document function."""
        documents = process_extracted_document(
            extracted_doc=sample_extracted_doc,
            session_id="test-session",
            chunk_size=500,
            chunk_overlap=100,
        )

        assert len(documents) >= 1
        assert all(isinstance(doc, Document) for doc in documents)

    def test_get_file_processor(self):
        """Test get_file_processor factory function."""
        processor = get_file_processor(
            chunk_size=500,
            chunk_overlap=100,
        )

        assert isinstance(processor, FileProcessor)
        assert processor.chunk_size == 500
        assert processor.chunk_overlap == 100
