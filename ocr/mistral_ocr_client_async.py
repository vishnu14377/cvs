"""
Async Mistral OCR client for concurrent PDF processing.

Uses asyncio for non-blocking I/O and concurrent API calls.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
from typing import Any

import aiofiles
from google.api_core import gapic_v1
from google.api_core import retry as api_retry
from src.core.config import mistral_ocr_config
from src.core.logger import get_logger
from src.core.vertex_ai_client import get_vertex_ai_client

logger = get_logger(__name__)


class MistralOcrClientAsync:
    """
    Async client for Mistral OCR model on Vertex AI.

    Enables concurrent processing of multiple PDFs with:
    - Non-blocking file I/O
    - Concurrent API calls
    - Configurable concurrency limits
    """

    def __init__(
        self,
        model_id: str | None = None,
        project_id: str | None = None,
        region: str | None = None,
        max_concurrent_requests: int = 5,
    ):
        """
        Initialize the async Mistral OCR client.

        Args:
            model_id: Mistral model ID
            project_id: GCP project ID
            region: GCP region
            max_concurrent_requests: Maximum number of concurrent API calls
        """
        self.model_id = model_id or mistral_ocr_config.MISTRAL_MODEL_ID
        self.model_publisher = mistral_ocr_config.MISTRAL_PUBLISHER
        self._vertex_client = get_vertex_ai_client(project_id=project_id, region=region)

        self._max_concurrent = max_concurrent_requests
        self._semaphore = asyncio.Semaphore(max_concurrent_requests)

        logger.info(
            "MistralOcrClientAsync initialized: model_id=%s, max_concurrent=%d",
            self.model_id,
            max_concurrent_requests,
        )

    async def process_pdf(
        self,
        pdf_path: str,
        timeout: float | None = None,
        save_response: bool = True,
        retry: api_retry.Retry | gapic_v1.method._MethodDefault | None = None,
    ) -> dict[str, Any]:
        """
        Process a PDF file using Mistral OCR asynchronously.

        Args:
            pdf_path: Path to the PDF file
            timeout: Request timeout in seconds
            save_response: Whether to save response to JSON
            retry: Retry configuration

        Returns:
            Result dictionary with success status and response data
        """
        # Use semaphore to limit concurrent requests
        async with self._semaphore:
            return await self._process_pdf_internal(
                pdf_path=pdf_path,
                timeout=timeout,
                save_response=save_response,
                retry=retry,
            )

    async def _process_pdf_internal(
        self,
        pdf_path: str,
        timeout: float | None,
        save_response: bool,
        retry: api_retry.Retry | gapic_v1.method._MethodDefault | None,
    ) -> dict[str, Any]:
        """Internal method for processing a single PDF."""
        result: dict[str, Any] = {
            "success": False,
            "pdf_path": pdf_path,
            "response": None,
            "output_file": None,
            "error": None,
        }

        logger.info("Processing PDF async: %s", pdf_path)

        # Validate file exists
        if not os.path.exists(pdf_path):
            error_msg = f"File not found: {pdf_path}"
            logger.error(error_msg)
            result["error"] = error_msg
            return result

        # Read and encode PDF asynchronously
        try:
            async with aiofiles.open(pdf_path, "rb") as f:
                pdf_bytes = await f.read()

            file_size_mb = len(pdf_bytes) / (1024 * 1024)
            logger.debug("PDF read successfully: size=%.2f MB", file_size_mb)

            # Run CPU-bound base64 encoding in thread pool
            base64_encoded = await asyncio.to_thread(base64.b64encode, pdf_bytes)
            base64_str = base64_encoded.decode("utf-8")
            logger.debug("PDF base64 encoded: length=%d chars", len(base64_str))

        except Exception as e:
            error_msg = f"Failed to read PDF: {e}"
            logger.error(error_msg, exc_info=True)
            result["error"] = error_msg
            return result

        # Build payload
        document_url = f"data:application/pdf;base64,{base64_str}"
        payload = {
            "model": self.model_id,
            "document": {
                "type": "document_url",
                "document_url": document_url,
            },
        }

        # Call Mistral OCR via Vertex AI
        timeout = timeout or mistral_ocr_config.MODEL_TIMEOUT_SECONDS
        logger.info("Calling Mistral OCR API async with timeout=%s", timeout)

        try:
            # Run the synchronous API call in a thread pool
            response = await asyncio.to_thread(
                self._vertex_client.generate,
                payload=payload,
                model_id=self.model_id,
                publisher=self.model_publisher,
                timeout=timeout,
                retry=retry,
            )

            result["success"] = True
            result["response"] = response
            logger.info("Mistral OCR succeeded for: %s", pdf_path)

            # Save response asynchronously
            if save_response:
                output_file = f"{os.path.splitext(pdf_path)[0]}_mistral_response.json"
                async with aiofiles.open(output_file, "w", encoding="utf-8") as f:
                    await f.write(json.dumps(response, indent=2, ensure_ascii=False))
                result["output_file"] = output_file
                logger.info("Response saved to: %s", output_file)

        except Exception as e:
            error_msg = f"Prediction failed: {e}"
            logger.error(error_msg, exc_info=True)
            result["error"] = error_msg

        return result

    async def process_multiple_pdfs(
        self,
        pdf_paths: list[str],
        timeout: float | None = None,
        save_response: bool = True,
        retry: api_retry.Retry | gapic_v1.method._MethodDefault | None = None,
    ) -> list[dict[str, Any]]:
        """
        Process multiple PDFs concurrently.

        Args:
            pdf_paths: List of PDF file paths
            timeout: Request timeout per PDF
            save_response: Whether to save responses
            retry: Retry configuration

        Returns:
            List of result dictionaries for each PDF
        """
        logger.info("Processing %d PDFs concurrently", len(pdf_paths))

        tasks = [
            self.process_pdf(
                pdf_path=path,
                timeout=timeout,
                save_response=save_response,
                retry=retry,
            )
            for path in pdf_paths
        ]

        # Gather with exception handling
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert exceptions to error results
        processed_results: list[dict[str, Any]] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error("PDF processing failed: %s - %s", pdf_paths[i], result)
                processed_results.append(
                    {
                        "success": False,
                        "pdf_path": pdf_paths[i],
                        "response": None,
                        "output_file": None,
                        "error": str(result),
                    }
                )
            else:
                processed_results.append(result)  # type: ignore[arg-type]

        success_count = sum(1 for r in processed_results if r["success"])
        logger.info("Batch processing complete: %d/%d succeeded", success_count, len(pdf_paths))

        return processed_results


# CLI usage
if __name__ == "__main__":
    import sys

    async def main():
        logger.info("Starting Mistral OCR async client CLI")

        ocr_client = MistralOcrClientAsync(max_concurrent_requests=3)

        if len(sys.argv) > 2:
            # Process multiple files
            pdf_paths = sys.argv[1:]
            logger.info("Processing %d files", len(pdf_paths))
            results = await ocr_client.process_multiple_pdfs(pdf_paths)
        else:
            # Process single file
            test_pdf_path = "test1.pdf" if len(sys.argv) < 2 else sys.argv[1]
            logger.info("Processing single file: %s", test_pdf_path)
            results = [await ocr_client.process_pdf(test_pdf_path)]

        # Output results
        for result in results:
            output = json.dumps(result, indent=2, ensure_ascii=False)
            logger.info("Result:\n%s", output)

    # Run async main
    asyncio.run(main())
