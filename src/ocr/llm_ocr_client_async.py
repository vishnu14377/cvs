"""
Async LLM OCR client using LangChain for processing PDF documents stored in GCS.

This is the async version of llm_ocr_client.py, providing:
- Concurrent processing of multiple PDFs
- Non-blocking async operations
- Same LangChain + structured output approach as sync version
- Semaphore-based rate limiting for concurrent requests

Uses LangChain with Vertex AI's Gemini models for document understanding.
PDF files are referenced by GCS URI and sent directly to the model.

Usage:
    from src.ocr.llm_ocr_client_async import LlmOcrClientAsync

    client = LlmOcrClientAsync(max_concurrent_requests=5)

    # Single document
    result = await client.process_document("gs://bucket/path/document.pdf")

    # Multiple documents concurrently
    pdf_uris = ["gs://bucket/doc1.pdf", "gs://bucket/doc2.pdf"]
    results = await client.process_multiple_documents(pdf_uris)
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError
from src.core.gcs_client import is_gcs_uri
from src.core.langchain_client import LangChainClient
from src.core.logger import get_logger
from src.ocr.data_models.llm_response import DocumentExtraction
from src.ocr.llm_ocr_client import PDF_MIME_TYPE, SYSTEM_PROMPT, is_pdf_file

logger = get_logger(__name__)

DEFAULT_MAX_CONCURRENT = 5


# =============================================================================
# Async LLM OCR LangChain Client
# =============================================================================


class LlmOcrClientAsync:
    """Async client for calling Gemini model via LangChain for OCR tasks.

    Mirrors LLMOcrClient with async/await, concurrent batch processing,
    and semaphore-based rate limiting.
    """

    def __init__(
        self,
        temperature: float = 0.1,
        max_output_tokens: int | None = None,
        max_concurrent_requests: int = DEFAULT_MAX_CONCURRENT,
    ):
        self._temperature = temperature
        self._max_output_tokens = max_output_tokens
        self._langchain = LangChainClient()
        self._structured_llm = self._langchain.with_structured_output(DocumentExtraction)
        self._system_prompt = SYSTEM_PROMPT
        self._max_concurrent = max_concurrent_requests
        self._semaphore = asyncio.Semaphore(max_concurrent_requests)
        self._executor: ThreadPoolExecutor | None = ThreadPoolExecutor(
            max_workers=max_concurrent_requests
        )

        logger.info(
            "LlmOcrClientAsync initialized: model=%s, max_concurrent=%d",
            self._langchain.model_id,
            max_concurrent_requests,
        )

    def close(self) -> None:
        """Explicitly release executor-backed resources."""
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    def shutdown(self) -> None:
        """Alias for close()."""
        self.close()

    async def process_document(
        self,
        file_path: str,
        prompt: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        timeout: float | None = None,
        save_response: bool = False,
    ) -> dict[str, Any]:
        """
        Process a document (PDF) stored in GCS using Gemini for OCR via LangChain.

        Async version of the sync client's process_document() method.

        Args:
            file_path: GCS URI (gs://bucket/path/document.pdf)
            prompt: Custom prompt for extraction
            temperature: Override temperature for this request
            max_output_tokens: Override max tokens for this request
            timeout: Override timeout for this request (seconds)
            save_response: Whether to save response (for API compatibility)

        Returns:
            Result dictionary with keys:
                - success: bool indicating if processing succeeded
                - pages: list of page dicts with 'index' and 'extracted_text'
                - error: error message (if failed, None on success)
        """
        async with self._semaphore:
            return await self._process_document_internal(
                file_path=file_path,
                prompt=prompt,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                timeout=timeout,
            )

    async def _process_document_internal(
        self,
        file_path: str,
        prompt: str | None,
        temperature: float | None,
        max_output_tokens: int | None,
        timeout: float | None,
    ) -> dict[str, Any]:
        """Internal document processing with concurrency control."""
        result: dict[str, Any] = {
            "success": False,
            "pages": [],
            "error": None,
        }

        # Validate GCS URI
        if not is_gcs_uri(file_path):
            error_msg = f"Invalid GCS URI: {file_path}. Must start with 'gs://'"
            logger.error(error_msg)
            result["error"] = error_msg
            return result

        # Validate PDF file type
        if not is_pdf_file(file_path):
            error_msg = f"Invalid file type: {file_path}. Only PDF files are supported."
            logger.error(error_msg)
            result["error"] = error_msg
            return result

        logger.info("Processing GCS PDF document async: %s", file_path)

        try:
            # Build messages (same as sync version)
            messages = self._build_messages(file_path, prompt)

            # Get structured LLM (same as sync version)
            structured_llm = self._get_structured_llm(temperature, max_output_tokens, timeout)

            logger.info("Calling Gemini API via LangChain (async) with structured output")

            # Invoke LLM in thread pool (LangChain is sync, we run in executor)
            loop = asyncio.get_running_loop()
            extraction: DocumentExtraction = await loop.run_in_executor(
                self._executor,
                lambda: structured_llm.invoke(messages),
            )

            # Convert Pydantic model to dict format (same as sync)
            pages = [
                {"index": page.index, "extracted_text": page.extracted_text}
                for page in extraction.pages
            ]

            result["success"] = True
            result["pages"] = pages

            logger.info(
                "LangChain OCR succeeded for: %s - extracted %d pages", file_path, len(pages)
            )

        except ValidationError as e:
            error_msg = f"Response validation failed: {e}"
            logger.error(error_msg)
            result["error"] = error_msg
        except Exception as e:
            error_msg = f"Gemini prediction failed: {e}"
            logger.error(error_msg, exc_info=True)
            result["error"] = error_msg

        return result

    def _build_messages(
        self,
        gcs_uri: str,
        prompt: str | None = None,
    ) -> list:
        """
        Build the message list for LangChain with GCS PDF file reference.

        IDENTICAL to sync version.

        Args:
            gcs_uri: GCS URI of the PDF document
            prompt: Custom prompt (optional)

        Returns:
            List of messages for LangChain
        """
        user_prompt = prompt or "Extract all text content from this PDF document page by page."

        # System message
        system_message = SystemMessage(content=self._system_prompt)

        # Human message with GCS PDF file reference
        human_message = HumanMessage(
            content=[
                {
                    "type": "media",
                    "file_uri": gcs_uri,
                    "mime_type": PDF_MIME_TYPE,
                },
                {
                    "type": "text",
                    "text": user_prompt,
                },
            ]
        )

        return [system_message, human_message]

    def _get_structured_llm(
        self,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        timeout: float | None = None,
    ):
        """
        Get structured LLM with optional parameter overrides.

        IDENTICAL to sync version.

        Args:
            temperature: Override temperature
            max_output_tokens: Override max tokens
            timeout: Override timeout (seconds)

        Returns:
            Structured LLM that returns DocumentExtraction
        """
        # If no overrides needed, use the pre-configured structured LLM
        if temperature is None and max_output_tokens is None and timeout is None:
            return self._structured_llm

        # Apply overrides via .bind() and create new structured output
        bind_kwargs = {}
        if temperature is not None:
            bind_kwargs["temperature"] = temperature
        if max_output_tokens is not None:
            bind_kwargs["max_output_tokens"] = max_output_tokens
        if timeout is not None:
            bind_kwargs["timeout"] = timeout

        # Bind new parameters and wrap with structured output
        configured_llm = self._langchain.client.bind(**bind_kwargs)
        return configured_llm.with_structured_output(DocumentExtraction)  # type: ignore[union-attr]

    async def process_multiple_documents(
        self,
        file_paths: list[str],
        prompt: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        timeout: float | None = None,
    ) -> list[dict[str, Any]]:
        """
        Process multiple PDF documents concurrently.

        NEW METHOD - not in sync version (that's the whole point of async!).

        Args:
            file_paths: List of GCS URIs
            prompt: Custom prompt for all documents
            temperature: Override temperature
            max_output_tokens: Override max tokens
            timeout: Override timeout (seconds)

        Returns:
            List of result dictionaries (one per document)
        """
        logger.info(
            "Processing %d documents concurrently with max_concurrent=%d",
            len(file_paths),
            self._max_concurrent,
        )

        tasks = [
            self.process_document(
                file_path=path,
                prompt=prompt,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                timeout=timeout,
            )
            for path in file_paths
        ]

        # Gather with exception handling
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert exceptions to error results
        formatted_results: list[dict[str, Any]] = []
        for _i, result in enumerate(results):
            if isinstance(result, Exception):
                formatted_results.append(
                    {
                        "success": False,
                        "pages": [],
                        "error": f"Exception during processing: {str(result)}",
                    }
                )
            else:
                formatted_results.append(result)  # type: ignore[arg-type]

        success_count = sum(1 for r in formatted_results if r["success"])
        logger.info("Batch complete: %d/%d successful", success_count, len(formatted_results))

        return formatted_results


# =============================================================================
# Singleton Access (optional - async doesn't need singleton as much)
# =============================================================================

_llm_ocr_client_async: LlmOcrClientAsync | None = None


def get_llm_ocr_client_async(
    temperature: float = 0.1,
    max_output_tokens: int | None = None,
    max_concurrent_requests: int = DEFAULT_MAX_CONCURRENT,
) -> LlmOcrClientAsync:
    """
    Get or create the singleton LlmOcrClientAsync instance.

    Args:
        temperature: Sampling temperature
        max_output_tokens: Maximum tokens in response
        max_concurrent_requests: Max concurrent requests

    Returns:
        LlmOcrClientAsync instance
    """
    global _llm_ocr_client_async

    if _llm_ocr_client_async is None:
        _llm_ocr_client_async = LlmOcrClientAsync(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            max_concurrent_requests=max_concurrent_requests,
        )

    return _llm_ocr_client_async


def reset_llm_ocr_client_async() -> None:
    """Reset the singleton client for testing."""
    global _llm_ocr_client_async
    client = _llm_ocr_client_async
    _llm_ocr_client_async = None

    if client is not None:
        client.close()

    logger.debug("LlmOcrClientAsync singleton reset")


# Alias for backward compatibility with tests
LlmHandlerAsync = LlmOcrClientAsync
