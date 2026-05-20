"""
ADR Vector Database Ingestion Pipeline.

A simplified, modular pipeline for ingesting extracted OCR documents
from GCS into PGVector for RAG retrieval.

Pipeline Flow:
1. List JSON files in {session_id}/extracted_text/ folder
2. Download and parse each JSON file (parallel)
3. Chunk documents using FileProcessor
4. Batch insert into vector store using VectorStoreManager

Usage:
    from src.adr_vector_database.ingestion_pipeline import ingest_session

    result = ingest_session(session_id="session-123")
    print(f"Stored {result.total_chunks_stored} chunks")
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_core.documents import Document
from src.adr_vector_database.data_models import (
    BatchIngestionResult,
    ExtractedDocument,
    IngestionResult,
)
from src.adr_vector_database.file_processor import FileProcessor
from src.adr_vector_database.vector_store import VectorStoreManager
from src.core.config import ocr_config, vectorstore_config
from src.core.gcs_client import download_from_gcs, list_files_in_gcs_folder
from src.core.local_directory_handler import cleanup_local_data, get_local_temp_path
from src.core.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# Helper Functions
# =============================================================================


def _parse_json_file(file_path: str) -> ExtractedDocument:
    """Parse an extracted JSON file into an ExtractedDocument."""
    with open(file_path, encoding="utf-8") as f:
        return ExtractedDocument.from_dict(json.load(f))


def _process_single_file(
    gcs_uri: str,
    session_id: str,
    local_dir: str,
    file_processor: FileProcessor,
    additional_metadata: dict | None = None,
) -> tuple[list[Document], IngestionResult]:
    """
    Download, parse, and chunk a single file.

    Returns:
        Tuple of (documents, ingestion_result)
    """
    try:
        # Download and parse
        local_path = download_from_gcs(gcs_uri, local_dir=local_dir)
        extracted_doc = _parse_json_file(local_path)

        # Process (chunk and convert to LangChain Documents)
        documents = file_processor.process_file(
            extracted_doc=extracted_doc,
            session_id=session_id,
            source_uri=gcs_uri,
            additional_metadata=additional_metadata,
        )

        result = IngestionResult(
            document_name=extracted_doc.document_name,
            session_id=session_id,
            success=True,
            chunks_created=len(documents),
            gcs_source_uri=gcs_uri,
        )
        return documents, result

    except Exception as e:
        logger.error("Failed to process '%s': %s", gcs_uri, e)
        result = IngestionResult(
            document_name=gcs_uri.split("/")[-1],
            session_id=session_id,
            success=False,
            error=str(e),
            gcs_source_uri=gcs_uri,
        )
        return [], result


# =============================================================================
# Main Pipeline Function
# =============================================================================


def ingest_session(
    session_id: str,
    collection_name: str | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    batch_size: int | None = None,
    max_workers: int | None = None,
    additional_metadata: dict | None = None,
    extracted_text_folder: str | None = None,
) -> BatchIngestionResult:
    """
    Ingest all extracted documents for a session from GCS.

    This is the main entry point for the ingestion pipeline. Files are
    processed in parallel using multiple threads.

    Args:
        session_id: Session ID (GCS folder prefix and metadata identifier).
        collection_name: PGVector collection name. Defaults to config value.
        chunk_size: Maximum characters per chunk. Defaults to config value.
        chunk_overlap: Characters to overlap between chunks. Defaults to config value.
        batch_size: Batch size for vector store insertions. Defaults to config value.
        max_workers: Number of parallel workers for file processing. Defaults to config value.
        additional_metadata: Extra metadata to add to all chunks.
        extracted_text_folder: Subfolder name for extracted text.

    Returns:
        BatchIngestionResult with details of the operation.

    Example:
        >>> result = ingest_session("session-123", max_workers=8)
        >>> print(f"Stored {result.total_chunks_stored} chunks")
    """
    # Apply config defaults
    chunk_size = chunk_size or vectorstore_config.DEFAULT_CHUNK_SIZE
    chunk_overlap = chunk_overlap or vectorstore_config.DEFAULT_CHUNK_OVERLAP
    batch_size = batch_size or vectorstore_config.DEFAULT_BATCH_SIZE
    max_workers = max_workers or vectorstore_config.DEFAULT_MAX_WORKERS

    extracted_folder = extracted_text_folder or ocr_config.GCS_EXTRACTED_TEXT_FOLDER
    folder_path = f"{session_id}/{extracted_folder}"
    local_temp_dir = get_local_temp_path(session_id, ocr_config.LOCAL_TMP_DIR)

    # Initialize result
    batch_result = BatchIngestionResult(session_id=session_id, total_documents=0)

    # Initialize processors
    file_processor = FileProcessor(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    vector_store = VectorStoreManager(collection_name=collection_name, batch_size=batch_size)

    try:
        # Step 1: List files
        logger.info("Listing JSON files in: %s", folder_path)
        gcs_uris = list_files_in_gcs_folder(folder_path, file_extension=".json")
        batch_result.total_documents = len(gcs_uris)

        if not gcs_uris:
            logger.warning("No JSON files found in '%s'", folder_path)
            return batch_result

        logger.info("Found %d files, processing with %d workers", len(gcs_uris), max_workers)

        # Step 2: Process files in parallel
        all_documents: list[Document] = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    _process_single_file,
                    uri,
                    session_id,
                    str(local_temp_dir),
                    file_processor,
                    additional_metadata,
                ): uri
                for uri in gcs_uris
            }

            for future in as_completed(futures):
                documents, result = future.result()
                batch_result.results.append(result)

                if result.success:
                    all_documents.extend(documents)
                    batch_result.successful_documents += 1
                else:
                    batch_result.failed_documents += 1
                    batch_result.errors.append(f"{result.gcs_source_uri}: {result.error}")

        # Step 3: Batch insert into vector store
        if all_documents:
            logger.info("Inserting %d chunks into vector store", len(all_documents))
            vector_ids = vector_store.batch_insert(all_documents)
            batch_result.total_chunks_stored = len(vector_ids)

        logger.info(
            "Ingestion complete: %d/%d docs successful, %d chunks stored",
            batch_result.successful_documents,
            batch_result.total_documents,
            batch_result.total_chunks_stored,
        )

    except Exception as e:
        logger.error("Ingestion failed for session '%s': %s", session_id, e)
        batch_result.errors.append(str(e))

    finally:
        # Cleanup temp files
        cleanup_local_data(session_id)

    return batch_result


def ingest_document(
    gcs_uri: str,
    session_id: str,
    collection_name: str | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    additional_metadata: dict | None = None,
) -> IngestionResult:
    """
    Ingest a single document from GCS.

    Args:
        gcs_uri: GCS URI to the extracted JSON file.
        session_id: Session ID for tracking and isolation.
        collection_name: PGVector collection name.
        chunk_size: Maximum characters per chunk. Defaults to config value.
        chunk_overlap: Characters to overlap between chunks. Defaults to config value.
        additional_metadata: Extra metadata to store with chunks.

    Returns:
        IngestionResult with details of the operation.
    """
    # Apply config defaults
    chunk_size = chunk_size or vectorstore_config.DEFAULT_CHUNK_SIZE
    chunk_overlap = chunk_overlap or vectorstore_config.DEFAULT_CHUNK_OVERLAP

    local_temp_dir = get_local_temp_path(session_id or "single_doc", ocr_config.LOCAL_TMP_DIR)
    file_processor = FileProcessor(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    vector_store = VectorStoreManager(collection_name=collection_name)

    try:
        # Process file
        documents, result = _process_single_file(
            gcs_uri, session_id, str(local_temp_dir), file_processor, additional_metadata
        )

        # Insert into vector store
        if documents:
            vector_ids = vector_store.insert(documents)
            result.vector_ids = vector_ids
            result.chunks_stored = len(vector_ids)

        return result

    finally:
        if session_id:
            cleanup_local_data(session_id)


# Backwards compatibility aliases
ingest_session_documents = ingest_session
ingest_document_from_gcs = ingest_document


if __name__ == "__main__":
    session_id = "73548964358912"
    print(f"Ingesting session: {session_id}")
    result = ingest_session(session_id)
    print(f"Result: {result.to_dict()}")
