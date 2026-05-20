"""Tests for ADR Vector Database document chunker."""

from src.adr_vector_database.chunker import (
    DocumentChunker,
    chunk_extracted_document,
)
from src.adr_vector_database.data_models import (
    DocumentChunk,
    ExtractedDocument,
    ExtractedPage,
)
from src.core.config import vectorstore_config


class TestDocumentChunkerInit:
    """Tests for DocumentChunker initialization."""

    def test_default_initialization(self):
        """Test chunker with default parameters."""
        chunker = DocumentChunker()
        assert chunker.chunk_size == vectorstore_config.DEFAULT_CHUNK_SIZE
        assert chunker.chunk_overlap == vectorstore_config.DEFAULT_CHUNK_OVERLAP

    def test_custom_parameters(self):
        """Test chunker with custom parameters."""
        chunker = DocumentChunker(chunk_size=500, chunk_overlap=100)
        assert chunker.chunk_size == 500
        assert chunker.chunk_overlap == 100

    def test_custom_separators(self):
        """Test chunker with custom separators."""
        custom_seps = ["\n", " "]
        chunker = DocumentChunker(separators=custom_seps)
        assert chunker.separators == custom_seps


class TestDocumentChunkerChunkDocument:
    """Tests for chunk_document method."""

    def test_chunk_single_page_document(self):
        """Test chunking a document with one page."""
        chunker = DocumentChunker(chunk_size=100, chunk_overlap=20)

        doc = ExtractedDocument(
            document_name="test.pdf",
            base_page_number=1,
            end_page_number=1,
            pages=[
                ExtractedPage(
                    sub_file_index=0,
                    original_page_number=1,
                    extracted_text="This is a short test document with some text content.",
                )
            ],
        )

        chunks = chunker.chunk_document(doc, session_id="session-123")

        assert len(chunks) >= 1
        assert all(isinstance(c, DocumentChunk) for c in chunks)
        assert all(c.session_id == "session-123" for c in chunks)
        assert all(c.document_name == "test.pdf" for c in chunks)

    def test_chunk_multi_page_document(self):
        """Test chunking a document with multiple pages."""
        chunker = DocumentChunker(chunk_size=100, chunk_overlap=20)

        pages = [
            ExtractedPage(
                sub_file_index=i,
                original_page_number=i + 1,
                extracted_text=f"Content for page {i + 1}. " * 10,
            )
            for i in range(3)
        ]

        doc = ExtractedDocument(
            document_name="multi.pdf", base_page_number=1, end_page_number=3, pages=pages
        )

        chunks = chunker.chunk_document(doc, session_id="sess")

        assert len(chunks) > 0
        # Each chunk should have page number metadata
        for chunk in chunks:
            assert len(chunk.page_numbers) > 0

    def test_chunk_preserves_page_numbers(self):
        """Test that chunks preserve original page numbers."""
        chunker = DocumentChunker(chunk_size=1000, chunk_overlap=100)

        doc = ExtractedDocument(
            document_name="test.pdf",
            base_page_number=10,
            end_page_number=12,
            pages=[
                ExtractedPage(
                    sub_file_index=0, original_page_number=10, extracted_text="Page 10 content"
                ),
                ExtractedPage(
                    sub_file_index=1, original_page_number=11, extracted_text="Page 11 content"
                ),
                ExtractedPage(
                    sub_file_index=2, original_page_number=12, extracted_text="Page 12 content"
                ),
            ],
        )

        chunks = chunker.chunk_document(doc, session_id="sess")

        # Verify page numbers are from the original document
        all_page_nums = set()
        for chunk in chunks:
            all_page_nums.update(chunk.page_numbers)

        assert 10 in all_page_nums
        assert 11 in all_page_nums
        assert 12 in all_page_nums

    def test_chunk_with_gcs_uri(self):
        """Test that chunks include GCS source URI."""
        chunker = DocumentChunker()

        doc = ExtractedDocument(
            document_name="test.pdf",
            base_page_number=1,
            end_page_number=1,
            pages=[
                ExtractedPage(sub_file_index=0, original_page_number=1, extracted_text="Content")
            ],
        )

        gcs_uri = "gs://bucket/path/to/doc.json"
        chunks = chunker.chunk_document(doc, session_id="sess", gcs_source_uri=gcs_uri)

        assert all(c.gcs_source_uri == gcs_uri for c in chunks)

    def test_chunk_with_additional_metadata(self):
        """Test that chunks include additional metadata."""
        chunker = DocumentChunker()

        doc = ExtractedDocument(
            document_name="test.pdf",
            base_page_number=1,
            end_page_number=1,
            pages=[
                ExtractedPage(sub_file_index=0, original_page_number=1, extracted_text="Content")
            ],
        )

        extra_meta = {"custom_field": "custom_value", "priority": 1}
        chunks = chunker.chunk_document(doc, session_id="sess", additional_metadata=extra_meta)

        for chunk in chunks:
            meta = chunk.to_metadata_dict()
            assert meta["custom_field"] == "custom_value"
            assert meta["priority"] == 1

    def test_chunk_empty_document(self):
        """Test chunking a document with no pages."""
        chunker = DocumentChunker()

        doc = ExtractedDocument(
            document_name="empty.pdf", base_page_number=1, end_page_number=1, pages=[]
        )

        chunks = chunker.chunk_document(doc, session_id="sess")
        assert len(chunks) == 0

    def test_chunk_document_with_empty_pages(self):
        """Test chunking a document where all pages are empty."""
        chunker = DocumentChunker()

        doc = ExtractedDocument(
            document_name="blank.pdf",
            base_page_number=1,
            end_page_number=2,
            pages=[
                ExtractedPage(sub_file_index=0, original_page_number=1, extracted_text=""),
                ExtractedPage(sub_file_index=1, original_page_number=2, extracted_text="   "),
            ],
        )

        chunks = chunker.chunk_document(doc, session_id="sess")
        assert len(chunks) == 0

    def test_chunk_indices_are_sequential(self):
        """Test that chunk indices are sequential."""
        chunker = DocumentChunker(chunk_size=50, chunk_overlap=10)

        doc = ExtractedDocument(
            document_name="test.pdf",
            base_page_number=1,
            end_page_number=1,
            pages=[
                ExtractedPage(
                    sub_file_index=0,
                    original_page_number=1,
                    extracted_text="A" * 200,  # Will create multiple chunks
                )
            ],
        )

        chunks = chunker.chunk_document(doc, session_id="sess")

        if len(chunks) > 1:
            indices = [c.chunk_index for c in chunks]
            assert indices == list(range(len(chunks)))


class TestDocumentChunkerChunkDocumentCombined:
    """Tests for chunk_document_combined method."""

    def test_combined_chunking(self):
        """Test combined chunking merges all pages."""
        chunker = DocumentChunker(chunk_size=500, chunk_overlap=50)

        pages = [
            ExtractedPage(sub_file_index=0, original_page_number=1, extracted_text="First page."),
            ExtractedPage(sub_file_index=1, original_page_number=2, extracted_text="Second page."),
            ExtractedPage(sub_file_index=2, original_page_number=3, extracted_text="Third page."),
        ]

        doc = ExtractedDocument(
            document_name="test.pdf", base_page_number=1, end_page_number=3, pages=pages
        )

        chunks = chunker.chunk_document_combined(doc, session_id="sess")

        assert len(chunks) >= 1
        # Combined chunks should reference all pages
        for chunk in chunks:
            assert 1 in chunk.page_numbers
            assert 2 in chunk.page_numbers
            assert 3 in chunk.page_numbers

    def test_combined_empty_document(self):
        """Test combined chunking with empty document."""
        chunker = DocumentChunker()

        doc = ExtractedDocument(
            document_name="empty.pdf", base_page_number=1, end_page_number=1, pages=[]
        )

        chunks = chunker.chunk_document_combined(doc, session_id="sess")
        assert len(chunks) == 0


class TestDocumentChunkerChunkPages:
    """Tests for chunk_pages method."""

    def test_chunk_pages_directly(self):
        """Test chunking a list of pages without document wrapper."""
        chunker = DocumentChunker()

        pages = [
            ExtractedPage(
                sub_file_index=0, original_page_number=5, extracted_text="Page five content"
            ),
            ExtractedPage(
                sub_file_index=1, original_page_number=6, extracted_text="Page six content"
            ),
        ]

        chunks = chunker.chunk_pages(
            pages=pages,
            document_name="custom_doc.pdf",
            session_id="sess-456",
        )

        assert len(chunks) > 0
        assert all(c.document_name == "custom_doc.pdf" for c in chunks)
        assert all(c.session_id == "sess-456" for c in chunks)


class TestChunkExtractedDocumentFunction:
    """Tests for the convenience function."""

    def test_convenience_function(self):
        """Test chunk_extracted_document convenience function."""
        doc = ExtractedDocument(
            document_name="test.pdf",
            base_page_number=1,
            end_page_number=1,
            pages=[
                ExtractedPage(
                    sub_file_index=0, original_page_number=1, extracted_text="Test content"
                )
            ],
        )

        chunks = chunk_extracted_document(
            document=doc,
            session_id="session-123",
            chunk_size=500,
            chunk_overlap=50,
        )

        assert len(chunks) > 0
        assert chunks[0].session_id == "session-123"

    def test_convenience_function_with_gcs_uri(self):
        """Test convenience function with GCS URI."""
        doc = ExtractedDocument(
            document_name="test.pdf",
            base_page_number=1,
            end_page_number=1,
            pages=[
                ExtractedPage(sub_file_index=0, original_page_number=1, extracted_text="Content")
            ],
        )

        gcs_uri = "gs://bucket/doc.json"
        chunks = chunk_extracted_document(
            document=doc,
            session_id="sess",
            gcs_source_uri=gcs_uri,
        )

        assert all(c.gcs_source_uri == gcs_uri for c in chunks)


class TestChunkMetadata:
    """Tests for chunk metadata generation."""

    def test_metadata_includes_page_range(self):
        """Test that metadata includes page start and end."""
        chunker = DocumentChunker()

        doc = ExtractedDocument(
            document_name="test.pdf",
            base_page_number=10,
            end_page_number=15,
            pages=[
                ExtractedPage(
                    sub_file_index=i, original_page_number=10 + i, extracted_text=f"Page {10 + i}"
                )
                for i in range(6)
            ],
        )

        chunks = chunker.chunk_document(doc, session_id="sess")

        for chunk in chunks:
            meta = chunk.to_metadata_dict()
            assert "page_start" in meta
            assert "page_end" in meta
            assert meta["page_start"] <= meta["page_end"]

    def test_metadata_includes_session_id(self):
        """Test that all chunks have session_id in metadata."""
        chunker = DocumentChunker()

        doc = ExtractedDocument(
            document_name="test.pdf",
            base_page_number=1,
            end_page_number=1,
            pages=[
                ExtractedPage(sub_file_index=0, original_page_number=1, extracted_text="Content")
            ],
        )

        chunks = chunker.chunk_document(doc, session_id="unique-session-id")

        for chunk in chunks:
            meta = chunk.to_metadata_dict()
            assert meta["session_id"] == "unique-session-id"
