"""
Sub-file handler for processing split PDF documents.

This module handles processing of PDF sub-files (chunks) that have been split
from a larger document. It extracts text using the OCR model client and creates
metadata with proper page number mapping based on the filename pattern.

Separation of Concerns:
- OcrModelClient: Handles OCR processing, returns normalized pages
- SubFileHandler: Handles page mapping to original document, GCS operations

Expected filename pattern:
    {document_name}_p{start_page}-{end_page}.pdf
    Example: COLLINS,ALEXANDER_5010280528530_FLC_REDACTED_part001_p1-18.pdf

The handler:
1. Parses the filename to extract document name and page range
2. Calls the OCR model client to extract text (normalized response)
3. Maps extracted page indices to actual page numbers in the original document
4. Creates comprehensive metadata for the processed sub-file
"""

from __future__ import annotations

import os
import re
from typing import Any

from src.core.config import ocr_config
from src.core.gcs_client import download_from_gcs, is_gcs_uri, upload_json_to_gcs
from src.core.local_directory_handler import get_local_data_path
from src.core.logger import get_logger
from src.ocr.data_models.normalized_response import NormalizedOcrResponse
from src.ocr.data_models.sub_file_models import (
    PageInfo,
    SubFileMetadata,
    SubFileResult,
    map_page_to_original,
)
from src.ocr.ocr_model_client import OcrModelClient, OcrModelType

logger = get_logger(__name__)


def parse_filename_page_range(filename: str) -> tuple[str, int, int]:
    """
    Parse the document name and page range from a sub-file filename.

    Expected pattern: {document_name}_p{start}-{end}.pdf
    Examples:
        - COLLINS,ALEXANDER_5010280528530_FLC_REDACTED_p1-18.pdf
        - test_document_p5-10.pdf
        - my_file_p1-1.pdf (single page)

    Args:
        filename: The PDF filename (with or without path)

    Returns:
        Tuple of (document_name, start_page, end_page)

    Raises:
        ValueError: If filename doesn't match expected pattern
    """
    # Extract just the filename if a path is provided
    basename = os.path.basename(filename)

    # Remove .pdf extension (case insensitive)
    if basename.lower().endswith(".pdf"):
        basename = basename[:-4]

    # Pattern to match _p{start}-{end} at the end
    # This captures the page range and everything before it as document name
    pattern = r"^(.+)_p(\d+)-(\d+)$"
    match = re.match(pattern, basename)

    if not match:
        raise ValueError(
            f"Filename '{filename}' does not match expected pattern: "
            f"{{document_name}}_p{{start}}-{{end}}.pdf"
        )

    document_name = match.group(1)
    start_page = int(match.group(2))
    end_page = int(match.group(3))

    # Validate page range
    if start_page < 1:
        raise ValueError(f"Start page must be >= 1, got {start_page}")
    if end_page < start_page:
        raise ValueError(f"End page ({end_page}) must be >= start page ({start_page})")

    logger.debug(
        "Parsed filename '%s': document='%s', pages=%d-%d",
        filename,
        document_name,
        start_page,
        end_page,
    )

    return document_name, start_page, end_page


class SubFileHandler:
    """
    Handler for processing PDF sub-files with OCR and page number mapping.

    This class takes split PDF files, processes them through OCR, and maps
    the extracted content to the correct page numbers in the original document.
    """

    def __init__(
        self,
        key: str,
        ocr_client: OcrModelClient | None = None,
        model_type: OcrModelType = "mistral",
    ):
        """
        Initialize the SubFileHandler.

        Args:
            key: Root folder key under which tmp/ and extracted_text/ folders will be saved in GCS
            ocr_client: Optional pre-configured OcrModelClient
            model_type: Type of OCR model to use ("mistral" or "llm").
                        Only used if ocr_client is not provided.
        """
        # Validate key
        key = str(key).strip()
        if not key:
            raise ValueError("key must not be empty")
        if "/" in key or "\\" in key or ".." in key:
            raise ValueError("key must not contain slashes or '..'")

        self._key = key

        if ocr_client is not None:
            self._ocr_client = ocr_client
        else:
            self._ocr_client = OcrModelClient(model_type=model_type)

        logger.info(
            "SubFileHandler initialized with key=%s, model_type=%s",
            self._key,
            self._ocr_client.model_type,
        )

    @property
    def model_type(self) -> OcrModelType:
        """Get the current OCR model type."""
        return self._ocr_client.model_type

    def process_sub_file(
        self,
        pdf_path: str,
        timeout: float | None = None,
        save_response: bool = False,
    ) -> SubFileResult:
        """
        Process a PDF sub-file and extract text with page number mapping.

        Args:
            pdf_path: Path to the PDF sub-file (local path or GCS URI like gs://bucket/path/file.pdf)
            timeout: Request timeout in seconds
            save_response: Whether to save the raw OCR response to a JSON file

        Returns:
            SubFileResult containing metadata, mapped pages, and combined text
        """
        logger.info("Processing sub-file: %s", pdf_path)

        # Track both local path and GCS URI for fallback support
        # - Mistral requires local files
        # - LLM (Gemini) requires GCS URIs
        local_pdf_path: str | None = None
        gcs_uri: str | None = None
        filename_for_parsing = pdf_path

        if is_gcs_uri(pdf_path):
            # Input is a GCS URI
            gcs_uri = pdf_path

            # Always download for Mistral (primary model) support
            # The local file will also be available if fallback to Mistral is needed
            try:
                local_dir = get_local_data_path(self._key)
                local_pdf_path = download_from_gcs(pdf_path, str(local_dir))
                filename_for_parsing = local_pdf_path
                logger.info("Downloaded GCS file to: %s", local_pdf_path)
            except Exception as e:
                logger.error("Failed to download from GCS: %s", e)
                metadata = SubFileMetadata(
                    document_name=os.path.basename(pdf_path),
                    base_page_number=1,
                    end_page_number=1,
                )
                return SubFileResult(
                    metadata=metadata, success=False, error=f"GCS download failed: {e}"
                )
        else:
            # Input is a local file path
            local_pdf_path = pdf_path
            # No GCS URI available - LLM-based processing paths that require GCS (for example, Gemini)
            # cannot be used for this request if they are configured as primary or fallback models.
            logger.warning(
                "Local file provided without GCS URI. Any configured LLM/GCS-based primary or fallback "
                "model will not be usable for this request: %s",
                pdf_path,
            )

        try:
            # Parse filename to get document name and page range
            try:
                document_name, base_page, end_page = parse_filename_page_range(filename_for_parsing)
            except ValueError as e:
                logger.error("Failed to parse filename: %s", e)
                metadata = SubFileMetadata(
                    document_name=os.path.basename(pdf_path),
                    base_page_number=1,
                    end_page_number=1,
                )
                return SubFileResult(metadata=metadata, success=False, error=str(e))

            expected_pages = end_page - base_page + 1
            logger.info(
                "Document: %s, Page range: %d-%d (expected %d pages)",
                document_name,
                base_page,
                end_page,
                expected_pages,
            )

            # Call OCR model with both local path and GCS URI
            # - OcrModelClient will use the appropriate path based on which model is being used
            # - Mistral uses local_pdf_path, LLM uses gcs_uri
            ocr_result = self._ocr_client.process_pdf(
                pdf_path=local_pdf_path,
                gcs_uri=gcs_uri,
                timeout=timeout,
                save_response=save_response,
            )

            # Initialize metadata
            metadata = SubFileMetadata(
                document_name=document_name,
                base_page_number=base_page,
                end_page_number=end_page,
            )

            if not ocr_result.success:
                logger.error("OCR processing failed: %s", ocr_result.error)
                return SubFileResult(
                    metadata=metadata,
                    success=False,
                    error=ocr_result.error,
                    model_used=ocr_result.model_used,
                    fallback_used=ocr_result.fallback_used,
                    primary_error=ocr_result.primary_error,
                )

            # Extract pages from normalized response and add metadata
            pages = self._extract_pages_from_response(ocr_result, base_page)

            # Log fallback information if applicable
            if ocr_result.fallback_used:
                logger.info(
                    "Successfully processed %d pages using fallback model (%s) after primary failed: %s",
                    len(pages),
                    ocr_result.model_used,
                    ocr_result.primary_error,
                )
            else:
                logger.info(
                    "Successfully processed %d pages from sub-file using %s model: %s",
                    len(pages),
                    ocr_result.model_used,
                    pdf_path,
                )

            return SubFileResult(
                metadata=metadata,
                pages=pages,
                success=True,
                model_used=ocr_result.model_used,
                fallback_used=ocr_result.fallback_used,
                primary_error=ocr_result.primary_error,
            )

        except Exception as e:
            logger.error("Unexpected error processing sub-file: %s", e)
            metadata = SubFileMetadata(
                document_name=os.path.basename(pdf_path),
                base_page_number=1,
                end_page_number=1,
            )
            return SubFileResult(metadata=metadata, success=False, error=str(e))

    def _extract_pages_from_response(
        self,
        ocr_result: NormalizedOcrResponse,
        base_page: int,
    ) -> list[PageInfo]:
        """
        Extract page information from NormalizedOcrResponse and add metadata.

        Takes the normalized pages from OcrModelClient and maps them to
        original document page numbers.

        Args:
            ocr_result: The NormalizedOcrResponse from OcrModelClient
            base_page: Starting page number in original document (1-based)

        Returns:
            List of PageInfo objects with mapped page numbers
        """
        pages: list[PageInfo] = []

        for normalized_page in ocr_result.pages:
            original_page_number = map_page_to_original(normalized_page.index, base_page)

            page_info = PageInfo(
                sub_file_index=normalized_page.index,
                original_page_number=original_page_number,
                extracted_text=normalized_page.extracted_text,
            )
            pages.append(page_info)

        logger.debug("Extracted %d pages from OCR response", len(pages))
        return pages

    def save_extracted_to_gcs(
        self,
        result: SubFileResult,
    ) -> str:
        """
        Save the extracted information from a SubFileResult to GCS as JSON.

        The file is stored in the extracted_text folder under the handler's key:
            gs://<bucket>/<GCS_WORKING_FOLDER>/<key>/extracted_text/<document_name>_p<start>-<end>.json

        Note: The base folder (GCS_WORKING_FOLDER) is automatically prepended by gcs_client.

        Args:
            result: The SubFileResult to save

        Returns:
            GCS URI of the saved JSON file

        Raises:
            ValueError: If config is missing
            Exception: If upload fails
        """
        # Get extracted text folder from config
        extracted_folder = ocr_config.GCS_EXTRACTED_TEXT_FOLDER

        # Build filename from metadata
        filename = f"{result.metadata.document_name}_p{result.metadata.base_page_number}-{result.metadata.end_page_number}.json"

        # Build relative GCS path using self._key (base folder is handled by gcs_client)
        gcs_path = f"{self._key}/{extracted_folder}/{filename}"

        # Build the data to save
        data = {
            "document_name": result.metadata.document_name,
            "base_page_number": result.metadata.base_page_number,
            "end_page_number": result.metadata.end_page_number,
            "pages": [
                {
                    "sub_file_index": page.sub_file_index,
                    "original_page_number": page.original_page_number,
                    "extracted_text": page.extracted_text,
                }
                for page in result.pages
            ],
            "success": result.success,
            "error": result.error,
        }

        logger.info("Saving extracted text to GCS path: %s", gcs_path)

        gcs_uri = upload_json_to_gcs(data, gcs_path)

        logger.info("Successfully saved extracted text to: %s", gcs_uri)

        return gcs_uri

    def run(
        self,
        pdf_path: str,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """
        Run the complete sub-file processing pipeline.

        This method orchestrates all steps:
        1. Process the PDF sub-file using OCR
        2. Save the extracted information to GCS

        Args:
            pdf_path: Path to the PDF sub-file (local path or GCS URI)
            timeout: Request timeout in seconds for OCR processing

        Returns:
            Dictionary containing:
                - success: Overall success status
                - result: The SubFileResult object
                - gcs_uri: GCS URI of the saved JSON (if successful)
                - error: Error message (if failed)
        """
        logger.info("Running sub-file handler pipeline for: %s", pdf_path)

        # Step 1: Process the PDF
        result = self.process_sub_file(
            pdf_path=pdf_path,
            timeout=timeout,
            save_response=False,
        )

        if not result.success:
            logger.error("Pipeline failed at OCR processing: %s", result.error)
            return {
                "success": False,
                "result": result,
                "gcs_uri": None,
                "error": result.error,
            }

        # Step 2: Save extracted information to GCS
        try:
            gcs_uri = self.save_extracted_to_gcs(result)
        except Exception as e:
            logger.error("Pipeline failed at GCS save: %s", e)
            return {
                "success": False,
                "result": result,
                "gcs_uri": None,
                "error": f"Failed to save to GCS: {e}",
            }

        logger.info("Pipeline completed successfully for: %s", pdf_path)

        return {
            "success": True,
            "result": result,
            "gcs_uri": gcs_uri,
            "error": None,
        }


def get_sub_file_handler(
    key: str,
    ocr_client: OcrModelClient | None = None,
    model_type: OcrModelType = "mistral",
) -> SubFileHandler:
    """
    Create a SubFileHandler instance.

    Args:
        key: Root folder key under which tmp/ and extracted_text/ folders will be saved in GCS
        ocr_client: Optional pre-configured OcrModelClient
        model_type: Type of OCR model to use ("mistral" or "llm").
                    Only used if ocr_client is not provided.

    Returns:
        SubFileHandler instance
    """
    return SubFileHandler(
        key=key,
        ocr_client=ocr_client,
        model_type=model_type,
    )


if __name__ == "__main__":
    # CLI for testing
    logger.info("SubFileHandler CLI")

    pdf_path = "test_p1-26.pdf"
    test_key = "654465468"

    handler = SubFileHandler(key=test_key, model_type="mistral")

    # Run the complete pipeline
    pipeline_result = handler.run(pdf_path)

    print(f"\n{'=' * 60}")
    print(f"Pipeline Success: {pipeline_result['success']}")

    if pipeline_result["success"]:
        result = pipeline_result["result"]
        print(f"Document: {result.metadata.document_name}")
        print(f"Page Range: {result.metadata.base_page_number}-{result.metadata.end_page_number}")
        print(f"GCS URI: {pipeline_result['gcs_uri']}")

        if result.pages:
            print(f"\nExtracted {len(result.pages)} pages:")
            for page in result.pages:
                text_preview = (
                    page.extracted_text[:100] + "..."
                    if len(page.extracted_text) > 100
                    else page.extracted_text
                )
                print(f"  - Page {page.original_page_number}: {len(page.extracted_text)} chars")
    else:
        print(f"Error: {pipeline_result['error']}")
    print(f"{'=' * 60}")
