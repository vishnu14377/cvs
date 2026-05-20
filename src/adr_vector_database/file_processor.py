"""
File Processor for ADR Vector Database.

Processes individual extracted JSON files by chunking the content
and converting to LangChain Documents ready for vector storage.

This module is called by the ingestion service for each file to be processed.
It returns LangChain Documents which can then be batch-inserted via the
VectorStoreManager.

Usage:
    from src.adr_vector_database.file_processor import FileProcessor

    processor = FileProcessor(chunk_size=1000, chunk_overlap=200)
    documents = processor.process_file(
        extracted_doc=doc,
        session_id="session-123",
        source_uri="gs://bucket/path/doc.json",
    )

    # Then use VectorStoreManager to store
    from src.adr_vector_database.vector_store import VectorStoreManager
    manager = VectorStoreManager()
    ids = manager.batch_insert(documents)
"""

from __future__ import annotations

from langchain_core.documents import Document
from src.adr_vector_database.chunker import DocumentChunker
from src.adr_vector_database.data_models import (
    DocumentChunk,
    ExtractedDocument,
)
from src.core.config import vectorstore_config
from src.core.logger import get_logger

logger = get_logger(__name__)


class FileProcessor:
    """
    Processes individual extracted documents into LangChain Documents.

    This class handles the processing pipeline for a single file:
    1. Receives an ExtractedDocument
    2. Chunks the document using DocumentChunker
    3. Converts chunks to LangChain Documents
    4. Returns the Documents for batch storage

    Example:
        >>> processor = FileProcessor(chunk_size=1000, chunk_overlap=200)
        >>> documents = processor.process_file(
        ...     extracted_doc=doc,
        ...     session_id="session-123",
        ...     source_uri="gs://bucket/path/doc.json",
        ... )
        >>> print(f"Created {len(documents)} documents")
    """

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ):
        """
        Initialize the file processor.

        Args:
            chunk_size: Maximum characters per chunk. Defaults to config value.
            chunk_overlap: Characters to overlap between chunks. Defaults to config value.
        """
        self.chunk_size = chunk_size or vectorstore_config.DEFAULT_CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or vectorstore_config.DEFAULT_CHUNK_OVERLAP

        # Initialize the chunker
        self._chunker = DocumentChunker(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

        logger.debug(
            "FileProcessor initialized: chunk_size=%d, overlap=%d",
            self.chunk_size,
            self.chunk_overlap,
        )

    def _chunk_document(
        self,
        extracted_doc: ExtractedDocument,
        session_id: str,
        source_uri: str | None = None,
        additional_metadata: dict | None = None,
    ) -> list[DocumentChunk]:
        """
        Chunk an extracted document using the DocumentChunker.

        Args:
            extracted_doc: The parsed extracted document.
            session_id: Session ID for filtering/isolation.
            source_uri: Source URI (GCS or local) for reference.
            additional_metadata: Optional extra metadata to include.

        Returns:
            List of DocumentChunk objects.
        """
        return self._chunker.chunk_document(
            document=extracted_doc,
            session_id=session_id,
            gcs_source_uri=source_uri,
            additional_metadata=additional_metadata,
        )

    def _convert_chunks_to_langchain(
        self,
        chunks: list[DocumentChunk],
        model_used: str | None = None,
    ) -> list[Document]:
        """
        Convert DocumentChunk objects to LangChain Document objects.

        Args:
            chunks: List of DocumentChunk objects.
            model_used: OCR model used for extraction (for metadata).

        Returns:
            List of LangChain Document objects ready for vector storage.
        """
        langchain_docs = []

        for chunk in chunks:
            doc = Document(
                page_content=chunk.text,
                metadata={
                    "session_id": chunk.session_id,
                    "document_name": chunk.document_name,
                    "page_numbers": chunk.page_numbers,
                    "chunk_index": chunk.chunk_index,
                    "source": chunk.gcs_source_uri or "",
                    "model_used": model_used or "unknown",
                    **(chunk.metadata or {}),
                },
            )
            langchain_docs.append(doc)

        return langchain_docs

    def process_file(
        self,
        extracted_doc: ExtractedDocument,
        session_id: str,
        source_uri: str | None = None,
        additional_metadata: dict | None = None,
    ) -> list[Document]:
        """
        Process a single extracted document and return LangChain Documents.

        Pipeline steps:
        1. Chunk the document using DocumentChunker
        2. Convert chunks to LangChain Documents
        3. Return Documents for batch storage

        Args:
            extracted_doc: The parsed extracted document to process.
            session_id: Session ID for tracking and isolation.
            source_uri: Source URI (GCS or local) for reference.
            additional_metadata: Optional extra metadata to store with chunks.

        Returns:
            List of LangChain Documents ready for vector storage.
        """
        logger.debug(
            "Processing document '%s' (session: %s)",
            extracted_doc.document_name,
            session_id,
        )

        # Step 1: Chunk the document
        doc_chunks = self._chunk_document(
            extracted_doc=extracted_doc,
            session_id=session_id,
            source_uri=source_uri,
            additional_metadata=additional_metadata,
        )

        if not doc_chunks:
            logger.warning(
                "No chunks created for document '%s'",
                extracted_doc.document_name,
            )
            return []

        # Step 2: Convert to LangChain Documents
        langchain_docs = self._convert_chunks_to_langchain(
            chunks=doc_chunks,
            model_used=extracted_doc.model_used,
        )

        logger.info(
            "Processed document '%s': %d chunks created (session: %s)",
            extracted_doc.document_name,
            len(langchain_docs),
            session_id,
        )

        return langchain_docs


# =============================================================================
# Convenience Functions
# =============================================================================


def process_extracted_document(
    extracted_doc: ExtractedDocument,
    session_id: str,
    source_uri: str | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    additional_metadata: dict | None = None,
) -> list[Document]:
    """
    Convenience function to process a single extracted document.

    Args:
        extracted_doc: The extracted document to process.
        session_id: Session ID for tracking.
        source_uri: Source URI for reference.
        chunk_size: Maximum characters per chunk. Defaults to config value.
        chunk_overlap: Characters to overlap between chunks. Defaults to config value.
        additional_metadata: Optional extra metadata.

    Returns:
        List of LangChain Documents ready for vector storage.
    """
    processor = FileProcessor(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return processor.process_file(
        extracted_doc=extracted_doc,
        session_id=session_id,
        source_uri=source_uri,
        additional_metadata=additional_metadata,
    )


def get_file_processor(
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> FileProcessor:
    """
    Factory function to create a FileProcessor instance.

    Args:
        chunk_size: Maximum characters per chunk. Defaults to config value.
        chunk_overlap: Characters to overlap between chunks. Defaults to config value.

    Returns:
        Configured FileProcessor instance.
    """
    return FileProcessor(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
