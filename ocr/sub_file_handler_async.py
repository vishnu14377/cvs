"""Async sub-file handler for OCR processing.

Wraps the sync SubFileHandler operations with async support using
the async model clients and async GCS client.
"""

from __future__ import annotations

from typing import Any

from core.gcs_client import is_gcs_uri
from core.gcs_client_async import download_from_gcs as async_download_from_gcs
from core.gcs_client_async import upload_json_to_gcs as async_upload_json
from core.local_directory_handler import get_local_data_path
from core.logger import get_logger
from ocr.data_models.sub_file_models import PageInfo, SubFileMetadata, SubFileResult
from ocr.ocr_model_client import OcrModelType
from ocr.sub_file_handler import parse_filename_page_range

logger = get_logger(__name__)


class SubFileHandlerAsync:
    """Async handler for processing individual PDF sub-files via OCR."""

    def __init__(
        self,
        session_id: str,
        model_type: OcrModelType = "mistral",
        max_concurrent_requests: int = 5,
    ):
        self._session_id = session_id
        self._model_type = model_type
        self._max_concurrent = max_concurrent_requests
        # Lazily initialized and shared across all process_sub_file calls so
        # we don't leak a ThreadPoolExecutor (LLM client) per sub-file.
        self._client = None

    @property
    def model_type(self) -> OcrModelType:
        return self._model_type

    def _get_client(self):
        """Lazily construct and cache the OCR client for this handler."""
        if self._client is not None:
            return self._client
        if self._model_type == "mistral":
            from ocr.mistral_ocr_client_async import MistralOcrClientAsync

            self._client = MistralOcrClientAsync(
                max_concurrent_requests=self._max_concurrent,
            )
        else:
            from ocr.llm_ocr_client_async import LlmOcrClientAsync

            self._client = LlmOcrClientAsync(
                max_concurrent_requests=self._max_concurrent,
            )
        return self._client

    def close(self) -> None:
        """Release any resources held by the underlying client (e.g. LLM executor)."""
        if self._client is not None and hasattr(self._client, "close"):
            try:
                self._client.close()
            except Exception as e:
                logger.warning("Error closing OCR client: %s", e)
        self._client = None

    async def __aenter__(self) -> SubFileHandlerAsync:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        self.close()

    async def process_sub_file(self, sub_file_uri: str) -> SubFileResult:
        """Process a single sub-file through OCR asynchronously."""
        logger.info("Processing sub-file async: %s", sub_file_uri)

        filename = sub_file_uri.rsplit("/", 1)[-1]
        try:
            doc_name, start_page, end_page = parse_filename_page_range(filename)
        except Exception:
            # Match sync SubFileHandler: page numbering is 1-based in the original document.
            doc_name, start_page, end_page = filename, 1, 1

        metadata = SubFileMetadata(
            document_name=doc_name,
            base_page_number=start_page,
            end_page_number=end_page,
        )

        try:
            client = self._get_client()

            if self._model_type == "mistral":
                # Mistral requires a local file path — download from GCS first.
                # (The sync SubFileHandler has the same download step.)
                pdf_path = sub_file_uri
                if is_gcs_uri(sub_file_uri):
                    local_dir = get_local_data_path(self._session_id)
                    pdf_path = await async_download_from_gcs(sub_file_uri, str(local_dir))
                # save_response=False mirrors the sync SubFileHandler pipeline —
                # the orchestrator handles persistence separately, so per-call
                # JSON dumps to the local disk would be redundant and can fail
                # in read-only containers.
                result = await client.process_pdf(pdf_path, save_response=False)
            else:
                result = await client.process_document(sub_file_uri)

            if not result.get("success"):
                return SubFileResult(
                    metadata=metadata,
                    success=False,
                    error=result.get("error", "OCR processing failed"),
                )

            raw_pages = result.get("pages", [])
            pages = [
                PageInfo(
                    sub_file_index=p.get("index", i),
                    original_page_number=start_page + i,
                    extracted_text=p.get("extracted_text", ""),
                )
                for i, p in enumerate(raw_pages)
            ]

            return SubFileResult(
                metadata=metadata,
                pages=pages,
                success=True,
                model_used=self._model_type,
            )

        except Exception as e:
            logger.error("Sub-file processing failed for %s: %s", sub_file_uri, e)
            return SubFileResult(
                metadata=metadata,
                success=False,
                error=str(e),
            )

    async def save_extracted_to_gcs(self, pages: list[dict[str, Any]], gcs_path: str) -> str:
        return await async_upload_json({"pages": pages}, gcs_path)

    async def run(self, sub_file_uri: str, output_gcs_path: str | None = None) -> SubFileResult:
        """Full pipeline: process sub-file + optionally save to GCS."""
        result = await self.process_sub_file(sub_file_uri)
        if result.success and output_gcs_path and result.pages:
            page_dicts = [p.to_dict() for p in result.pages]
            await self.save_extracted_to_gcs(page_dicts, output_gcs_path)
        return result
