"""
OCR Orchestrator for processing PDF documents.

This module orchestrates the complete OCR processing pipeline:
1. Reads GCS folder URI and lists all PDF files
2. For each PDF: Downloads, splits into chunks, and uploads to GCS
3. Processes all sub-files in parallel using concurrent execution
4. Aggregates results and saves to GCS

Flow:
    Input GCS Folder URI → List PDFs → For each PDF: Split → Upload chunks →
    Process all chunks in parallel → Save extracted text to GCS

GCS Layout:
    gs://<bucket>/<GCS_WORKING_FOLDER>/<session_id>/tmp/           - Split PDF chunks
    gs://<bucket>/<GCS_WORKING_FOLDER>/<session_id>/extracted_text/ - OCR results as JSON
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.core.config import ocr_config
from src.core.gcs_client import (
    download_folder_files,
    download_from_gcs,
    is_gcs_uri,
)
from src.core.local_directory_handler import (
    cleanup_local_data,
    get_local_temp_path,
)
from src.core.logger import get_logger
from src.ocr.data_models.orchestrator_models import (
    OcrOrchestrationResult,
    SubFileProcessingResult,
)
from src.ocr.ocr_model_client import OcrModelType
from src.ocr.pdf_handler import split_pdf_by_pages, split_pdf_by_size
from src.ocr.sub_file_handler import get_sub_file_handler

logger = get_logger(__name__)


class OcrOrchestrator:
    """
    Orchestrator for the OCR processing pipeline.

    This class coordinates the entire OCR workflow:
    1. Reads source PDF from GCS or local path
    2. Splits PDF into manageable chunks
    3. Processes each chunk with OCR
    4. Aggregates and saves results
    """

    def __init__(
        self,
        session_id: str,
        model_type: OcrModelType = "mistral",
        size_limit_mb: float = 5.0,
        pages_per_chunk: int | None = None,
        max_workers: int = 5,
    ):
        """
        Initialize the OCR Orchestrator.

        Args:
            session_id: Unique identifier for this processing session.
                        Used to organize files in GCS (e.g., session ID).
            model_type: Type of OCR model to use ("mistral" or "llm").
            size_limit_mb: Target size limit per chunk when splitting by size.
                          Used if pages_per_chunk is not specified.
            pages_per_chunk: If specified, split by fixed page count instead of size.
            max_workers: Maximum number of parallel workers for processing sub-files.
        """
        # Validate session_id
        key = str(session_id).strip()
        if not key:
            raise ValueError("session_id must not be empty")
        if "/" in key or "\\" in key or ".." in key:
            raise ValueError("session_id must not contain slashes or '..'")

        self._session_id = key
        self._model_type = model_type
        self._size_limit_mb = size_limit_mb
        self._pages_per_chunk = pages_per_chunk
        self._max_workers = max_workers

        # Initialize sub-file handler
        self._sub_file_handler = get_sub_file_handler(
            key=self._session_id,
            model_type=self._model_type,
        )

        logger.info(
            "OcrOrchestrator initialized: key=%s, model=%s, size_limit=%.1fMB, pages_per_chunk=%s, max_workers=%d",
            self._session_id,
            self._model_type,
            self._size_limit_mb,
            self._pages_per_chunk,
            self._max_workers,
        )

    @property
    def session_id(self) -> str:
        """Get the unique key for this orchestrator."""
        return self._session_id

    @property
    def tmp_folder_path(self) -> str:
        """Get the relative GCS path to the tmp folder for split files."""
        return f"{self._session_id}/{ocr_config.GCS_TEMP_FOLDER}"

    @property
    def extracted_text_folder_path(self) -> str:
        """Get the relative GCS path to the extracted text folder."""
        return f"{self._session_id}/{ocr_config.GCS_EXTRACTED_TEXT_FOLDER}"

    def _split_pdf(self, pdf_path: str) -> list[str]:
        """
        Split the PDF into chunks and upload to GCS.

        The splitting strategy depends on the OCR model type:
        - Mistral: Split by fixed page count (pages_per_chunk)
        - LLM: Split by target file size (size_limit_mb)

        Args:
            pdf_path: Path to the local PDF file

        Returns:
            List of GCS URIs for the split chunks
        """
        logger.info("Splitting PDF: %s", pdf_path)

        if self._model_type == "mistral":
            # Mistral model: split by fixed page count
            pages_per_chunk = self._pages_per_chunk or ocr_config.PAGES_PER_CHUNK
            logger.info(
                "Mistral model: Splitting by page count: %d pages per chunk", pages_per_chunk
            )
            chunk_uris = split_pdf_by_pages(
                pdf_path=pdf_path,
                pages_per_chunk=pages_per_chunk,
                unique_key=self._session_id,
            )
        else:
            # LLM model: split by target size
            logger.info("LLM model: Splitting by size: %.1f MB limit", self._size_limit_mb)
            chunk_uris = split_pdf_by_size(
                pdf_path=pdf_path,
                unique_key=self._session_id,
                size_limit_mb=self._size_limit_mb,
            )

        logger.info("PDF split into %d chunks", len(chunk_uris))
        return chunk_uris

    def _process_sub_file(
        self,
        sub_file_uri: str,
        timeout: float | None = None,
    ) -> SubFileProcessingResult:
        """
        Process a single sub-file with OCR.

        Args:
            sub_file_uri: GCS URI of the sub-file to process
            timeout: Request timeout in seconds

        Returns:
            SubFileProcessingResult with processing outcome including fallback info
        """
        logger.info("Processing sub-file: %s", sub_file_uri)

        try:
            # Run the sub-file handler pipeline
            result = self._sub_file_handler.run(
                pdf_path=sub_file_uri,
                timeout=timeout,
            )

            sub_file_result = result.get("result")

            if result["success"]:
                page_count = len(sub_file_result.pages) if sub_file_result else 0
                return SubFileProcessingResult(
                    gcs_uri=sub_file_uri,
                    success=True,
                    extracted_text_uri=result["gcs_uri"],
                    page_count=page_count,
                    model_used=sub_file_result.model_used if sub_file_result else None,
                    fallback_used=sub_file_result.fallback_used if sub_file_result else False,
                    primary_error=sub_file_result.primary_error if sub_file_result else None,
                )
            else:
                return SubFileProcessingResult(
                    gcs_uri=sub_file_uri,
                    success=False,
                    error=result["error"],
                    model_used=sub_file_result.model_used if sub_file_result else None,
                    fallback_used=sub_file_result.fallback_used if sub_file_result else False,
                    primary_error=sub_file_result.primary_error if sub_file_result else None,
                )

        except Exception as e:
            logger.error("Failed to process sub-file %s: %s", sub_file_uri, e)
            return SubFileProcessingResult(
                gcs_uri=sub_file_uri,
                success=False,
                error=str(e),
            )

    def _process_sub_files_parallel(
        self,
        sub_files: list[str],
        timeout: float | None = None,
    ) -> list[SubFileProcessingResult]:
        """
        Process multiple sub-files in parallel using ThreadPoolExecutor.

        Args:
            sub_files: List of GCS URIs for sub-files to process
            timeout: Request timeout in seconds for each OCR operation

        Returns:
            List of SubFileProcessingResult for each sub-file
        """
        logger.info(
            "Processing %d sub-files in parallel with %d workers",
            len(sub_files),
            self._max_workers,
        )

        results: list[SubFileProcessingResult] = []

        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            # Submit all tasks
            future_to_uri = {
                executor.submit(self._process_sub_file, uri, timeout): uri for uri in sub_files
            }

            # Collect results as they complete
            for future in as_completed(future_to_uri):
                uri = future_to_uri[future]
                try:
                    result = future.result()
                    results.append(result)

                    if result.success:
                        logger.info(
                            "Sub-file processed successfully: %s (%d pages)",
                            uri,
                            result.page_count,
                        )
                    else:
                        logger.error(
                            "Sub-file processing failed: %s - %s",
                            uri,
                            result.error,
                        )
                except Exception as e:
                    logger.error("Unexpected error processing %s: %s", uri, e)
                    results.append(
                        SubFileProcessingResult(
                            gcs_uri=uri,
                            success=False,
                            error=str(e),
                        )
                    )

        # Sort results by GCS URI to maintain consistent order
        results.sort(key=lambda r: r.gcs_uri)

        return results

    def run(
        self,
        source_uri: str,
        timeout: float | None = None,
    ) -> OcrOrchestrationResult:
        """
        Run the complete OCR orchestration pipeline.

        This method supports three modes:
        1. GCS folder: Downloads all PDFs in parallel, splits, and processes
        2. Single GCS file: Downloads, splits, and processes
        3. Local file: Splits and processes directly

        Pipeline:
        1. Download source file(s) using gcs_client functions
        2. Split each PDF into chunks and upload to GCS
        3. List all sub-files from GCS tmp folder
        4. Process all sub-files in parallel using ThreadPoolExecutor
        5. Aggregate results and cleanup

        Args:
            source_uri: GCS URI (file or folder) or local path to the source PDF
            timeout: Request timeout in seconds for each OCR operation

        Returns:
            OcrOrchestrationResult with complete processing outcome
        """
        logger.info("=" * 60)
        logger.info("Starting OCR orchestration for: %s", source_uri)
        logger.info("Session ID: %s", self._session_id)
        logger.info("Max parallel workers: %d", self._max_workers)
        logger.info("=" * 60)

        result = OcrOrchestrationResult(
            session_id=self._session_id,
            source_uri=source_uri,
            success=False,
        )

        # Use local directory handler for organized file storage
        local_download_dir = get_local_temp_path(self._session_id, "downloads")

        try:
            # Step 1: Download source file(s)
            logger.info("-" * 40)
            logger.info("Step 1: Downloading source file(s)")
            logger.info("Local download directory: %s", local_download_dir)

            all_pdf_paths: list[str] = []

            if is_gcs_uri(source_uri):
                # Check if it's a folder (ends with / or has no file extension)
                is_folder = source_uri.endswith("/") or not os.path.splitext(source_uri)[1]

                if is_folder:
                    # GCS folder - use parallel download from gcs_client
                    logger.info("Source is a GCS folder: %s", source_uri)
                    all_pdf_paths = download_folder_files(
                        folder_uri=source_uri,
                        local_dir=str(local_download_dir),
                        file_extension=".pdf",
                        max_workers=self._max_workers,
                    )
                    if not all_pdf_paths:
                        raise ValueError(f"No PDF files found in folder: {source_uri}")
                else:
                    # Single GCS file - use download_from_gcs
                    logger.info("Source is a single GCS file: %s", source_uri)
                    local_path = download_from_gcs(source_uri, str(local_download_dir))
                    all_pdf_paths = [local_path]
            else:
                # Local file - validate existence using local_directory_handler approach
                if not os.path.exists(source_uri):
                    raise FileNotFoundError(f"Local file not found: {source_uri}")
                logger.info("Source is a local file: %s", source_uri)
                all_pdf_paths = [source_uri]

            logger.info("Total PDFs to process: %d", len(all_pdf_paths))

            # Step 2: Split each PDF into chunks
            logger.info("-" * 40)
            logger.info("Step 2: Splitting PDF(s) into chunks")

            all_chunk_uris: list[str] = []
            for i, pdf_path in enumerate(all_pdf_paths, 1):
                logger.info("Splitting PDF %d/%d: %s", i, len(all_pdf_paths), pdf_path)
                chunk_uris = self._split_pdf(pdf_path)
                all_chunk_uris.extend(chunk_uris)
                logger.info(
                    "PDF %d/%d split into %d chunks", i, len(all_pdf_paths), len(chunk_uris)
                )

            if not all_chunk_uris:
                raise ValueError("PDF splitting produced no chunks")

            logger.info("Total chunks created: %d", len(all_chunk_uris))

            # Step 3: Use chunk URIs directly (sorted for consistent processing order)
            # Note: We use all_chunk_uris directly instead of re-listing from GCS to avoid
            # picking up stale files from previous runs with the same session_id
            logger.info("-" * 40)
            logger.info("Step 3: Preparing sub-files for processing")
            sub_files = sorted(all_chunk_uris)

            result.total_sub_files = len(sub_files)
            logger.info("Found %d sub-files to process", len(sub_files))

            # Step 4: Process all sub-files in parallel
            logger.info("-" * 40)
            logger.info(
                "Step 4: Processing %d sub-files in parallel with %d workers",
                len(sub_files),
                self._max_workers,
            )

            sub_file_results = self._process_sub_files_parallel(sub_files, timeout)

            # Aggregate results
            for sub_result in sub_file_results:
                result.sub_file_results.append(sub_result)

                if sub_result.success:
                    result.successful_sub_files += 1
                    result.total_pages += sub_result.page_count
                    if sub_result.extracted_text_uri:
                        result.extracted_text_uris.append(sub_result.extracted_text_uri)
                else:
                    result.failed_sub_files += 1

            # Step 5: Compute fallback statistics
            result.compute_fallback_stats()

            # Step 6: Determine overall success
            result.success = result.failed_sub_files == 0

            logger.info("=" * 60)
            logger.info("OCR orchestration completed")
            logger.info("Total sub-files: %d", result.total_sub_files)
            logger.info("Successful: %d", result.successful_sub_files)
            logger.info("Failed: %d", result.failed_sub_files)
            logger.info("Total pages: %d", result.total_pages)

            # Log fallback statistics if any fallbacks occurred
            if result.fallback_stats:
                stats = result.fallback_stats
                logger.info("-" * 40)
                logger.info("Fallback Statistics:")
                logger.info("  Primary model success: %d", stats.primary_success_count)
                logger.info("  Fallback success: %d", stats.fallback_success_count)
                logger.info("  Both failed: %d", stats.both_failed_count)
                logger.info("  Mistral used: %d", stats.mistral_used_count)
                logger.info("  LLM used: %d", stats.llm_used_count)
                logger.info("  Fallback rate: %.1f%%", stats.fallback_rate)

            logger.info("=" * 60)

            return result

        except Exception as e:
            logger.error("OCR orchestration failed: %s", e)
            result.error = str(e)
            return result

        finally:
            # Clean up local data directory using local_directory_handler
            cleanup_local_data(self._session_id)
            logger.debug("Cleaned up local data directory for session: %s", self._session_id)


def get_ocr_orchestrator(
    session_id: str,
    model_type: OcrModelType = "mistral",
    size_limit_mb: float = 5.0,
    pages_per_chunk: int | None = None,
    max_workers: int = 5,
) -> OcrOrchestrator:
    """
    Create an OcrOrchestrator instance.

    Args:
        session_id: Unique identifier for this processing session.
        model_type: Type of OCR model to use ("mistral" or "llm").
        size_limit_mb: Target size limit per chunk when splitting by size.
        pages_per_chunk: If specified, split by fixed page count instead of size.
        max_workers: Maximum number of parallel workers for processing sub-files.

    Returns:
        OcrOrchestrator instance
    """
    return OcrOrchestrator(
        session_id=session_id,
        model_type=model_type,
        size_limit_mb=size_limit_mb,
        pages_per_chunk=pages_per_chunk,
        max_workers=max_workers,
    )


if __name__ == "__main__":
    # CLI for testing
    logger.info("OCR Orchestrator CLI")

    # Example: Process a GCS folder with multiple PDFs
    source_uri = "gs://care_connect_ai_initiatives/test_full_adrs/BINGHAM,CALLIE_91202308017_FLC5_REDACTED.pdf"
    session_id = "73358912556778"
    model_type = "mistral"
    max_workers = 5

    # Create orchestrator
    orchestrator = OcrOrchestrator(
        session_id=session_id,
        model_type=model_type,
        max_workers=max_workers,
    )

    # Run the pipeline
    result = orchestrator.run(source_uri)

    print(f"\n{'=' * 60}")
    print("Orchestration Result")
    print(f"{'=' * 60}")
    print(f"Source: {result.source_uri}")
    print(f"Session ID: {result.session_id}")
    print(f"Overall Success: {result.success}")
    print(f"Total Sub-files: {result.total_sub_files}")
    print(f"Successful: {result.successful_sub_files}")
    print(f"Failed: {result.failed_sub_files}")
    print(f"Total Pages: {result.total_pages}")

    if result.extracted_text_uris:
        print("\nExtracted Text Files:")
        for uri in result.extracted_text_uris:
            print(f"  - {uri}")

    if result.error:
        print(f"\nError: {result.error}")

    if result.failed_sub_files > 0:
        print("\nFailed Sub-files:")
        for sub_result in result.sub_file_results:
            if not sub_result.success:
                print(f"  - {sub_result.gcs_uri}: {sub_result.error}")

    print(f"{'=' * 60}")
