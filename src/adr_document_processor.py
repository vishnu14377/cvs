"""
ADR Document Processor.

This module provides a high-level processor for ADR (Appeal/Dispute Resolution) documents.
It orchestrates the complete pipeline:
1. OCR processing via OcrOrchestrator (PDF to extracted text)
2. Vector database ingestion via ingestion_pipeline (extracted text to embeddings)

Usage:
    from src.adr_document_processor import process_adr_document, AdrDocumentProcessor

    # Simple function call
    result = process_adr_document(
        session_id="session-123",
        gcs_uri="gs://bucket/path/to/document.pdf"
    )

    # Or using the class directly
    processor = AdrDocumentProcessor(
        session_id="session-123",
        model_type="mistral",
        max_workers=5
    )
    result = processor.process(gcs_uri="gs://bucket/path/to/document.pdf")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from src.adr_vector_database.data_models import BatchIngestionResult
from src.adr_vector_database.ingestion_pipeline import ingest_session
from src.core.logger import get_logger
from src.ocr.data_models.orchestrator_models import FallbackStats, OcrOrchestrationResult
from src.ocr.ocr_orchestrator import OcrOrchestrator

logger = get_logger(__name__)

# Type alias for OCR model types
OcrModelType = Literal["mistral", "llm"]


@dataclass
class AdrProcessingResult:
    """
    Result of the complete ADR document processing pipeline.

    Contains results from both OCR and ingestion stages.
    """

    session_id: str
    source_uri: str
    success: bool = False

    # OCR stage results
    ocr_success: bool = False
    ocr_total_pages: int = 0
    ocr_total_sub_files: int = 0
    ocr_successful_sub_files: int = 0
    ocr_failed_sub_files: int = 0
    ocr_extracted_text_uris: list[str] = field(default_factory=list)
    ocr_fallback_stats: FallbackStats | None = None
    ocr_error: str | None = None

    # Ingestion stage results
    ingestion_success: bool = False
    ingestion_total_documents: int = 0
    ingestion_successful_documents: int = 0
    ingestion_failed_documents: int = 0
    ingestion_total_chunks: int = 0
    ingestion_errors: list[str] = field(default_factory=list)

    # Overall error
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "session_id": self.session_id,
            "source_uri": self.source_uri,
            "success": self.success,
            "ocr": {
                "success": self.ocr_success,
                "total_pages": self.ocr_total_pages,
                "total_sub_files": self.ocr_total_sub_files,
                "successful_sub_files": self.ocr_successful_sub_files,
                "failed_sub_files": self.ocr_failed_sub_files,
                "extracted_text_uris": self.ocr_extracted_text_uris,
                "fallback_stats": self.ocr_fallback_stats.model_dump()
                if self.ocr_fallback_stats
                else None,
                "error": self.ocr_error,
            },
            "ingestion": {
                "success": self.ingestion_success,
                "total_documents": self.ingestion_total_documents,
                "successful_documents": self.ingestion_successful_documents,
                "failed_documents": self.ingestion_failed_documents,
                "total_chunks": self.ingestion_total_chunks,
                "errors": self.ingestion_errors,
            },
            "error": self.error,
        }


class AdrDocumentProcessor:
    """
    High-level processor for ADR documents.

    Orchestrates the complete pipeline from PDF to vector embeddings:
    1. OCR processing: PDF → extracted text JSON files
    2. Ingestion: extracted text → vector embeddings in PGVector
    """

    def __init__(
        self,
        session_id: str,
        model_type: OcrModelType = "mistral",
        size_limit_mb: float = 5.0,
        pages_per_chunk: int | None = None,
        max_workers: int = 5,
        collection_name: str | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        batch_size: int | None = None,
    ):
        """
        Initialize the ADR Document Processor.

        Args:
            session_id: Unique identifier for this processing session.
            model_type: Type of OCR model to use ("mistral" or "llm").
            size_limit_mb: Target size limit per chunk when splitting by size.
            pages_per_chunk: If specified, split by fixed page count instead of size.
            max_workers: Maximum number of parallel workers for processing.
            collection_name: PGVector collection name. Defaults to config value.
            chunk_size: Maximum characters per text chunk. Defaults to config value.
            chunk_overlap: Characters to overlap between chunks. Defaults to config value.
            batch_size: Batch size for vector store insertions. Defaults to config value.
        """
        # Validate session_id
        session_id = str(session_id).strip()
        if not session_id:
            raise ValueError("session_id must not be empty")
        if "/" in session_id or "\\" in session_id or ".." in session_id:
            raise ValueError("session_id must not contain slashes or '..'")

        self._session_id = session_id
        self._model_type = model_type
        self._size_limit_mb = size_limit_mb
        self._pages_per_chunk = pages_per_chunk
        self._max_workers = max_workers

        # Ingestion parameters
        self._collection_name = collection_name
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._batch_size = batch_size

        logger.info(
            "AdrDocumentProcessor initialized: session_id=%s, model=%s, max_workers=%d",
            self._session_id,
            self._model_type,
            self._max_workers,
        )

    @property
    def session_id(self) -> str:
        """Get the session ID."""
        return self._session_id

    def _run_ocr(
        self,
        gcs_uri: str,
        timeout: float | None = None,
    ) -> OcrOrchestrationResult:
        """
        Run the OCR processing stage.

        Args:
            gcs_uri: GCS URI to the source PDF (file or folder).
            timeout: Request timeout in seconds for each OCR operation.

        Returns:
            OcrOrchestrationResult with OCR processing outcome.
        """
        logger.info("Starting OCR processing for: %s", gcs_uri)

        orchestrator = OcrOrchestrator(
            session_id=self._session_id,
            model_type=self._model_type,
            size_limit_mb=self._size_limit_mb,
            pages_per_chunk=self._pages_per_chunk,
            max_workers=self._max_workers,
        )

        return orchestrator.run(source_uri=gcs_uri, timeout=timeout)

    def _run_ingestion(
        self,
        additional_metadata: dict | None = None,
    ) -> BatchIngestionResult:
        """
        Run the vector database ingestion stage.

        Args:
            additional_metadata: Extra metadata to add to all chunks.

        Returns:
            BatchIngestionResult with ingestion outcome.
        """
        logger.info("Starting ingestion for session: %s", self._session_id)

        return ingest_session(
            session_id=self._session_id,
            collection_name=self._collection_name,
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
            batch_size=self._batch_size,
            max_workers=self._max_workers,
            additional_metadata=additional_metadata,
        )

    def process(
        self,
        gcs_uri: str,
        timeout: float | None = None,
        additional_metadata: dict | None = None,
        skip_ocr: bool = False,
        skip_ingestion: bool = False,
    ) -> AdrProcessingResult:
        """
        Process an ADR document through the complete pipeline.

        Pipeline stages:
        1. OCR: Extract text from PDF using OcrOrchestrator
        2. Ingestion: Chunk and embed text into PGVector

        Args:
            gcs_uri: GCS URI to the source PDF (file or folder).
            timeout: Request timeout in seconds for each OCR operation.
            additional_metadata: Extra metadata to store with vector chunks.
            skip_ocr: If True, skip OCR and only run ingestion (assumes OCR already done).
            skip_ingestion: If True, skip ingestion and only run OCR.

        Returns:
            AdrProcessingResult with complete processing outcome.
        """
        logger.info("=" * 60)
        logger.info("Starting ADR document processing")
        logger.info("Session ID: %s", self._session_id)
        logger.info("Source URI: %s", gcs_uri)
        logger.info("Skip OCR: %s, Skip Ingestion: %s", skip_ocr, skip_ingestion)
        logger.info("=" * 60)

        result = AdrProcessingResult(
            session_id=self._session_id,
            source_uri=gcs_uri,
        )

        try:
            # Stage 1: OCR Processing
            if not skip_ocr:
                logger.info("-" * 40)
                logger.info("Stage 1: OCR Processing")

                ocr_result = self._run_ocr(gcs_uri, timeout)

                # Populate OCR results
                result.ocr_success = ocr_result.success
                result.ocr_total_pages = ocr_result.total_pages
                result.ocr_total_sub_files = ocr_result.total_sub_files
                result.ocr_successful_sub_files = ocr_result.successful_sub_files
                result.ocr_failed_sub_files = ocr_result.failed_sub_files
                result.ocr_extracted_text_uris = ocr_result.extracted_text_uris
                result.ocr_fallback_stats = ocr_result.fallback_stats
                result.ocr_error = ocr_result.error

                logger.info(
                    "OCR completed: %d/%d sub-files successful, %d pages",
                    ocr_result.successful_sub_files,
                    ocr_result.total_sub_files,
                    ocr_result.total_pages,
                )

                # Check if OCR failed entirely
                if not ocr_result.success and ocr_result.successful_sub_files == 0:
                    logger.error("OCR failed completely, skipping ingestion")
                    result.error = (
                        f"OCR failed: {ocr_result.error or 'No sub-files processed successfully'}"
                    )
                    return result
            else:
                logger.info("Skipping OCR stage (skip_ocr=True)")
                result.ocr_success = True  # Assume OCR was already done

            # Stage 2: Vector Database Ingestion
            if not skip_ingestion:
                logger.info("-" * 40)
                logger.info("Stage 2: Vector Database Ingestion")

                ingestion_result = self._run_ingestion(additional_metadata)

                # Populate ingestion results
                result.ingestion_success = ingestion_result.success
                result.ingestion_total_documents = ingestion_result.total_documents
                result.ingestion_successful_documents = ingestion_result.successful_documents
                result.ingestion_failed_documents = ingestion_result.failed_documents
                result.ingestion_total_chunks = ingestion_result.total_chunks_stored
                result.ingestion_errors = ingestion_result.errors

                logger.info(
                    "Ingestion completed: %d/%d documents successful, %d chunks stored",
                    ingestion_result.successful_documents,
                    ingestion_result.total_documents,
                    ingestion_result.total_chunks_stored,
                )
            else:
                logger.info("Skipping ingestion stage (skip_ingestion=True)")
                result.ingestion_success = True  # Assume ingestion will be done later

            # Determine overall success
            result.success = (skip_ocr or result.ocr_success) and (
                skip_ingestion or result.ingestion_success
            )

            logger.info("=" * 60)
            logger.info("ADR document processing completed")
            logger.info("Overall success: %s", result.success)
            logger.info("=" * 60)

        except Exception as e:
            logger.error("ADR document processing failed: %s", e)
            result.error = str(e)

        return result


def process_adr_document(
    session_id: str,
    gcs_uri: str,
    model_type: OcrModelType = "mistral",
    size_limit_mb: float = 5.0,
    pages_per_chunk: int | None = None,
    max_workers: int = 5,
    collection_name: str | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    batch_size: int | None = None,
    timeout: float | None = None,
    additional_metadata: dict | None = None,
    skip_ocr: bool = False,
    skip_ingestion: bool = False,
) -> AdrProcessingResult:
    """
    Process an ADR document through the complete pipeline.

    This is a convenience function that creates an AdrDocumentProcessor
    and runs the processing pipeline.

    Args:
        session_id: Unique identifier for this processing session.
        gcs_uri: GCS URI to the source PDF (file or folder).
        model_type: Type of OCR model to use ("mistral" or "llm").
        size_limit_mb: Target size limit per chunk when splitting by size.
        pages_per_chunk: If specified, split by fixed page count instead of size.
        max_workers: Maximum number of parallel workers for processing.
        collection_name: PGVector collection name. Defaults to config value.
        chunk_size: Maximum characters per text chunk. Defaults to config value.
        chunk_overlap: Characters to overlap between chunks. Defaults to config value.
        batch_size: Batch size for vector store insertions. Defaults to config value.
        timeout: Request timeout in seconds for each OCR operation.
        additional_metadata: Extra metadata to store with vector chunks.
        skip_ocr: If True, skip OCR and only run ingestion.
        skip_ingestion: If True, skip ingestion and only run OCR.

    Returns:
        AdrProcessingResult with complete processing outcome.

    Example:
        >>> result = process_adr_document(
        ...     session_id="session-123",
        ...     gcs_uri="gs://bucket/path/to/document.pdf"
        ... )
        >>> print(f"Success: {result.success}")
        >>> print(f"Pages processed: {result.ocr_total_pages}")
        >>> print(f"Chunks stored: {result.ingestion_total_chunks}")
    """
    processor = AdrDocumentProcessor(
        session_id=session_id,
        model_type=model_type,
        size_limit_mb=size_limit_mb,
        pages_per_chunk=pages_per_chunk,
        max_workers=max_workers,
        collection_name=collection_name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        batch_size=batch_size,
    )

    return processor.process(
        gcs_uri=gcs_uri,
        timeout=timeout,
        additional_metadata=additional_metadata,
        skip_ocr=skip_ocr,
        skip_ingestion=skip_ingestion,
    )


if __name__ == "__main__":
    # CLI for testing
    logger.info("ADR Document Processor CLI")

    # Example usage
    session_id = "test-session-12347"
    source_uri = "gs://care_connect_ai_initiatives/test_full_adrs/"

    print(f"\n{'=' * 60}")
    print("ADR Document Processor")
    print(f"{'=' * 60}")
    print(f"Session ID: {session_id}")
    print(f"Source URI: {source_uri}")
    print(f"{'=' * 60}\n")

    # Process the document
    result = process_adr_document(
        session_id=session_id,
        gcs_uri=source_uri,
        model_type="mistral",
        max_workers=5,
    )

    print(f"\n{'=' * 60}")
    print("Processing Result")
    print(f"{'=' * 60}")
    print(f"Overall Success: {result.success}")
    print("\nOCR Stage:")
    print(f"  Success: {result.ocr_success}")
    print(f"  Total Pages: {result.ocr_total_pages}")
    print(f"  Sub-files: {result.ocr_successful_sub_files}/{result.ocr_total_sub_files}")
    print("\nIngestion Stage:")
    print(f"  Success: {result.ingestion_success}")
    print(
        f"  Documents: {result.ingestion_successful_documents}/{result.ingestion_total_documents}"
    )
    print(f"  Chunks Stored: {result.ingestion_total_chunks}")

    if result.error:
        print(f"\nError: {result.error}")

    if result.ocr_extracted_text_uris:
        print("\nExtracted Text Files:")
        for uri in result.ocr_extracted_text_uris[:5]:  # Show first 5
            print(f"  - {uri}")
        if len(result.ocr_extracted_text_uris) > 5:
            print(f"  ... and {len(result.ocr_extracted_text_uris) - 5} more")

    print(f"{'=' * 60}")
