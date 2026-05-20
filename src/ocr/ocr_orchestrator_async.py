"""Async OCR Orchestrator for processing PDF documents.

Replaces the sync ThreadPoolExecutor-based parallel processing with
asyncio.gather() + Semaphore for concurrent sub-file OCR.
"""

from __future__ import annotations

import asyncio

from src.core.gcs_client_async import list_files_in_gcs_folder
from src.core.logger import get_logger
from src.ocr.data_models.orchestrator_models import (
    OcrOrchestrationResult,
    SubFileProcessingResult,
)
from src.ocr.data_models.sub_file_models import SubFileMetadata, SubFileResult
from src.ocr.ocr_model_client import OcrModelType
from src.ocr.sub_file_handler_async import SubFileHandlerAsync

logger = get_logger(__name__)


class OcrOrchestratorAsync:
    """Async orchestrator for the OCR processing pipeline."""

    def __init__(
        self,
        session_id: str,
        model_type: OcrModelType = "mistral",
        max_workers: int = 5,
    ):
        self._session_id = session_id
        self._model_type = model_type
        self._max_workers = max_workers
        self._semaphore = asyncio.Semaphore(max_workers)

    @property
    def session_id(self) -> str:
        return self._session_id

    async def _process_single_sub_file(
        self, handler: SubFileHandlerAsync, sub_file_uri: str
    ) -> SubFileResult:
        async with self._semaphore:
            return await handler.process_sub_file(sub_file_uri)

    async def _process_sub_files_concurrent(self, sub_file_uris: list[str]) -> list[SubFileResult]:
        # Use `async with` so the handler's cached OCR client (and its
        # ThreadPoolExecutor for the LLM path) is released when we're done.
        async with SubFileHandlerAsync(
            session_id=self._session_id,
            model_type=self._model_type,
            max_concurrent_requests=self._max_workers,
        ) as handler:
            results = await asyncio.gather(
                *[self._process_single_sub_file(handler, uri) for uri in sub_file_uris],
                return_exceptions=True,
            )

        processed = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error("Sub-file failed: %s — %s", sub_file_uris[i], result)
                processed.append(
                    SubFileResult(
                        metadata=SubFileMetadata(
                            document_name=sub_file_uris[i].rsplit("/", 1)[-1],
                            base_page_number=0,
                            end_page_number=0,
                        ),
                        success=False,
                        error=str(result),
                    )
                )
            else:
                processed.append(result)
        return processed

    async def run(self, gcs_folder_uri: str) -> OcrOrchestrationResult:
        """Run the full async OCR pipeline."""
        logger.info(
            "Starting async OCR orchestration: session=%s, folder=%s, max_workers=%d",
            self._session_id,
            gcs_folder_uri,
            self._max_workers,
        )

        sub_file_uris = await list_files_in_gcs_folder(gcs_folder_uri, ".pdf")
        if not sub_file_uris:
            logger.warning("No PDF files found in %s", gcs_folder_uri)
            return OcrOrchestrationResult(
                session_id=self._session_id,
                source_uri=gcs_folder_uri,
                success=True,
                total_sub_files=0,
                successful_sub_files=0,
                failed_sub_files=0,
                total_pages=0,
            )

        logger.info("Found %d sub-files to process", len(sub_file_uris))
        sub_results = await self._process_sub_files_concurrent(sub_file_uris)

        successful = sum(1 for r in sub_results if r.success)
        failed = len(sub_results) - successful
        total_pages = sum(len(r.pages) for r in sub_results if r.success)

        logger.info(
            "Async OCR complete: session=%s, files=%d, success=%d, failed=%d, pages=%d",
            self._session_id,
            len(sub_results),
            successful,
            failed,
            total_pages,
        )

        return OcrOrchestrationResult(
            session_id=self._session_id,
            source_uri=gcs_folder_uri,
            success=failed == 0,
            total_sub_files=len(sub_results),
            successful_sub_files=successful,
            failed_sub_files=failed,
            total_pages=total_pages,
            sub_file_results=[
                SubFileProcessingResult(
                    gcs_uri=sub_file_uris[i] if i < len(sub_file_uris) else "unknown",
                    success=r.success,
                    page_count=len(r.pages) if r.success else 0,
                    error=r.error,
                    model_used=r.model_used,
                )
                for i, r in enumerate(sub_results)
            ],
        )
