"""
Document chunker for ADR Vector Database.

Provides text chunking functionality using LangChain's text splitters.
Supports configurable chunk sizes and overlap for optimal RAG retrieval.

Usage:
    from src.adr_vector_database.chunker import DocumentChunker

    chunker = DocumentChunker(chunk_size=1000, chunk_overlap=200)
    chunks = chunker.chunk_document(extracted_doc, session_id="session-123")
"""

from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.adr_vector_database.data_models import (
    DocumentChunk,
    ExtractedDocument,
    ExtractedPage,
)
from src.core.config import vectorstore_config
from src.core.logger import get_logger

logger = get_logger(__name__)


# Default separators for text splitting
DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


class DocumentChunker:
    """
    Chunks extracted documents into smaller pieces for embedding.

    Uses LangChain's RecursiveCharacterTextSplitter to intelligently
    split text at natural boundaries (paragraphs, sentences, etc.).

    Attributes:
        chunk_size: Maximum size of each chunk in characters.
        chunk_overlap: Number of characters to overlap between chunks.
        separators: List of separators to use for splitting (in priority order).

    Example:
        >>> chunker = DocumentChunker(chunk_size=1000, chunk_overlap=200)
        >>> doc = ExtractedDocument(
        ...     document_name="test.pdf",
        ...     base_page_number=1,
        ...     end_page_number=5,
        ...     pages=[ExtractedPage(sub_file_index=0, original_page_number=1, extracted_text="...")]
        ... )
        >>> chunks = chunker.chunk_document(doc, session_id="session-123")
        >>> print(len(chunks))
    """

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        separators: list[str] | None = None,
    ):
        """
        Initialize the document chunker.

        Args:
            chunk_size: Maximum characters per chunk. Defaults to config value.
            chunk_overlap: Characters to overlap between chunks. Defaults to config value.
            separators: Custom separators for splitting. Defaults to
                       ["\n\n", "\n", ". ", " ", ""].
        """
        self.chunk_size = chunk_size or vectorstore_config.DEFAULT_CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or vectorstore_config.DEFAULT_CHUNK_OVERLAP
        self.separators = separators or DEFAULT_SEPARATORS

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            length_function=len,
            is_separator_regex=False,
        )

        logger.debug(
            "DocumentChunker initialized: chunk_size=%d, chunk_overlap=%d",
            self.chunk_size,
            self.chunk_overlap,
        )

    def chunk_document(
        self,
        document: ExtractedDocument,
        session_id: str,
        gcs_source_uri: str | None = None,
        additional_metadata: dict | None = None,
    ) -> list[DocumentChunk]:
        """
        Chunk an extracted document into smaller pieces.

        Processes each page individually to maintain page-level metadata,
        then combines into chunks while tracking which pages each chunk spans.

        Args:
            document: The extracted document to chunk.
            session_id: Session ID for tracking/isolation.
            gcs_source_uri: Optional GCS URI of the source JSON.
            additional_metadata: Optional extra metadata to include.

        Returns:
            List of DocumentChunk objects ready for embedding.
        """
        if not document.pages:
            logger.warning("Document '%s' has no pages to chunk", document.document_name)
            return []

        chunks: list[DocumentChunk] = []
        chunk_index = 0
        extra_meta = additional_metadata or {}

        # Strategy: Chunk by page groups to maintain page-level granularity
        # This allows us to track which pages each chunk came from
        for page in document.pages:
            if not page.extracted_text or not page.extracted_text.strip():
                continue

            # Split the page text into chunks
            page_chunks = self._splitter.split_text(page.extracted_text)

            for chunk_text in page_chunks:
                if not chunk_text.strip():
                    continue

                chunk = DocumentChunk(
                    text=chunk_text,
                    document_name=document.document_name,
                    page_numbers=[page.original_page_number],
                    chunk_index=chunk_index,
                    session_id=session_id,
                    gcs_source_uri=gcs_source_uri,
                    metadata=extra_meta.copy(),
                )
                chunks.append(chunk)
                chunk_index += 1

        logger.info(
            "Chunked document '%s': %d pages -> %d chunks",
            document.document_name,
            document.page_count,
            len(chunks),
        )

        return chunks

    def chunk_document_combined(
        self,
        document: ExtractedDocument,
        session_id: str,
        gcs_source_uri: str | None = None,
        additional_metadata: dict | None = None,
    ) -> list[DocumentChunk]:
        """
        Chunk a document by combining all pages first, then splitting.

        This approach may create chunks that span multiple pages.
        Use this when you want larger context windows that cross page boundaries.

        Args:
            document: The extracted document to chunk.
            session_id: Session ID for tracking/isolation.
            gcs_source_uri: Optional GCS URI of the source JSON.
            additional_metadata: Optional extra metadata to include.

        Returns:
            List of DocumentChunk objects ready for embedding.
        """
        if not document.pages:
            logger.warning("Document '%s' has no pages to chunk", document.document_name)
            return []

        # Combine all pages with page markers
        combined_text = document.get_combined_text(separator="\n\n")
        if not combined_text.strip():
            logger.warning("Document '%s' has no text content", document.document_name)
            return []

        # Get all page numbers from the document
        all_page_numbers = [p.original_page_number for p in document.pages]
        extra_meta = additional_metadata or {}

        # Split the combined text
        text_chunks = self._splitter.split_text(combined_text)

        chunks: list[DocumentChunk] = []
        for idx, chunk_text in enumerate(text_chunks):
            if not chunk_text.strip():
                continue

            chunk = DocumentChunk(
                text=chunk_text,
                document_name=document.document_name,
                page_numbers=all_page_numbers,  # All pages since we combined
                chunk_index=idx,
                session_id=session_id,
                gcs_source_uri=gcs_source_uri,
                metadata=extra_meta.copy(),
            )
            chunks.append(chunk)

        logger.info(
            "Chunked document '%s' (combined): %d pages -> %d chunks",
            document.document_name,
            document.page_count,
            len(chunks),
        )

        return chunks

    def chunk_pages(
        self,
        pages: list[ExtractedPage],
        document_name: str,
        session_id: str,
        gcs_source_uri: str | None = None,
        additional_metadata: dict | None = None,
    ) -> list[DocumentChunk]:
        """
        Chunk a list of pages directly (without full document wrapper).

        Useful when you have pages from different sources or want to
        process a subset of pages.

        Args:
            pages: List of extracted pages to chunk.
            document_name: Name to use for the document.
            session_id: Session ID for tracking/isolation.
            gcs_source_uri: Optional GCS URI of the source.
            additional_metadata: Optional extra metadata to include.

        Returns:
            List of DocumentChunk objects ready for embedding.
        """
        # Create a temporary document wrapper
        doc = ExtractedDocument(
            document_name=document_name,
            base_page_number=min(p.original_page_number for p in pages) if pages else 1,
            end_page_number=max(p.original_page_number for p in pages) if pages else 1,
            pages=pages,
        )

        return self.chunk_document(
            document=doc,
            session_id=session_id,
            gcs_source_uri=gcs_source_uri,
            additional_metadata=additional_metadata,
        )


# Convenience function for quick chunking
def chunk_extracted_document(
    document: ExtractedDocument,
    session_id: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    gcs_source_uri: str | None = None,
) -> list[DocumentChunk]:
    """
    Convenience function to chunk an extracted document.

    Args:
        document: The extracted document to chunk.
        session_id: Session ID for tracking/isolation.
        chunk_size: Maximum characters per chunk. Defaults to config value.
        chunk_overlap: Characters to overlap between chunks. Defaults to config value.
        gcs_source_uri: Optional GCS URI of the source JSON.

    Returns:
        List of DocumentChunk objects ready for embedding.
    """
    chunker = DocumentChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return chunker.chunk_document(document, session_id, gcs_source_uri)
