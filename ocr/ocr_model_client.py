"""
Unified OCR model client that abstracts different OCR backends.

This module provides a single interface for OCR processing, supporting both:
- Mistral OCR: Returns pages with 'markdown' key
- LLM OCR (Gemini): Returns pages with 'extracted_text' key

The OcrModelClient normalizes both response formats to a standard structure
with 'index' and 'extracted_text' per page, which is then consumed by SubFileHandler.

Fallback Behavior:
- By default, uses Mistral as primary model with automatic fallback to LLM (Gemini)
- If Mistral fails (timeout, API error, invalid response), automatically retries with LLM
- Fallback can be disabled via enable_fallback parameter
- Tracks which model was used and whether fallback occurred in the response

Separation of Concerns:
- Mistral/LLM clients: Return raw JSON responses in their native formats
- OcrModelClient: Normalizes responses to unified format (NormalizedOcrResponse)
- SubFileHandler: Adds metadata like original page numbers, document name

Usage:
    from src.ocr.ocr_model_client import OcrModelClient

    # Use Mistral with automatic fallback to LLM (default)
    client = OcrModelClient(model_type="mistral", enable_fallback=True)
    result = client.process_pdf("document.pdf")

    # Check if fallback was used
    if result.fallback_used:
        print(f"Primary model failed: {result.primary_error}")
        print(f"Used fallback model: {result.model_used}")

    # Access normalized pages
    for page in result.pages:
        print(f"Page {page.index}: {page.extracted_text[:100]}")
"""

from __future__ import annotations

from typing import Literal

from src.core.logger import get_logger
from src.ocr.data_models.normalized_response import (
    NormalizedOcrResponse,
    NormalizedPage,
)
from src.ocr.llm_ocr_client import LLMOcrClient
from src.ocr.mistral_ocr_client import MistralOcrClient

logger = get_logger(__name__)

# Type alias for OCR model type
OcrModelType = Literal["mistral", "llm"]


class OcrModelClient:
    """
    Unified OCR client that normalizes responses from Mistral and LLM backends.

    This client is the interface between raw OCR model responses and the
    SubFileHandler. It reads JSON responses from both models and creates
    a normalized version (NormalizedOcrResponse).

    The client is responsible for:
    - Delegating to the appropriate backend (Mistral or LLM)
    - Implementing fallback logic (Mistral -> LLM if primary fails)
    - Understanding each model's response structure
    - Normalizing to standard format: {index, extracted_text}
    - Tracking which model was used and whether fallback occurred

    Returns:
        NormalizedOcrResponse with:
        - success: bool
        - pages: List[NormalizedPage] with 'index' and 'extracted_text'
        - error: Optional error message
        - model_used: Which model produced the response ("mistral" or "llm")
        - fallback_used: Whether fallback from primary to secondary model occurred
        - primary_error: Error from primary model if fallback was triggered
    """

    def __init__(
        self,
        model_type: OcrModelType = "mistral",
        client: MistralOcrClient | LLMOcrClient | None = None,
        enable_fallback: bool = True,
    ):
        """
        Initialize the OCR model client.

        Args:
            model_type: Type of OCR model ("mistral" or "llm").
                        Ignored if client is provided.
            client: Optional pre-configured OCR client instance.
                    If provided, model_type is inferred from the client type.
            enable_fallback: Whether to enable automatic fallback to secondary model
                            if primary model fails. Default is True.
                            - If model_type is "mistral", fallback is to LLM
                            - If model_type is "llm", fallback is to Mistral
        """
        if client is not None:
            self._primary_client = client
            self._model_type: OcrModelType = (
                "llm" if isinstance(client, LLMOcrClient) else "mistral"
            )
        else:
            self._model_type = model_type
            if model_type == "llm":
                self._primary_client = LLMOcrClient()
            else:
                self._primary_client = MistralOcrClient()

        self._enable_fallback = enable_fallback
        self._fallback_client: MistralOcrClient | LLMOcrClient | None = None

        logger.info(
            "OcrModelClient initialized with model_type=%s, enable_fallback=%s",
            self._model_type,
            self._enable_fallback,
        )

    @property
    def model_type(self) -> OcrModelType:
        """Get the primary model type."""
        return self._model_type

    @property
    def enable_fallback(self) -> bool:
        """Check if fallback is enabled."""
        return self._enable_fallback

    def _get_fallback_client(self) -> MistralOcrClient | LLMOcrClient:
        """
        Get or create the fallback client (lazy initialization).

        Returns the opposite model type from the primary.
        """
        if self._fallback_client is None:
            if self._model_type == "mistral":
                self._fallback_client = LLMOcrClient()
                logger.info("Initialized LLM fallback client")
            else:
                self._fallback_client = MistralOcrClient()
                logger.info("Initialized Mistral fallback client")
        return self._fallback_client

    def _get_fallback_model_type(self) -> OcrModelType:
        """Get the fallback model type."""
        return "llm" if self._model_type == "mistral" else "mistral"

    def process_pdf(
        self,
        pdf_path: str,
        gcs_uri: str | None = None,
        timeout: float | None = None,
        save_response: bool = False,
    ) -> NormalizedOcrResponse:
        """
        Process a PDF file using the configured OCR backend with fallback support.

        Args:
            pdf_path: Path to the local PDF file (required for Mistral)
            gcs_uri: GCS URI of the PDF file (required for LLM/Gemini).
                     If not provided and LLM is used, pdf_path is used as-is.
            timeout: Request timeout in seconds
            save_response: Whether to save the raw response

        Returns:
            NormalizedOcrResponse with normalized pages and fallback tracking info

        Note:
            - Mistral model uses pdf_path (local file)
            - LLM model uses gcs_uri (GCS file) if provided, otherwise pdf_path
            - When fallback occurs, the appropriate path is used for each model
        """
        logger.info(
            "Processing PDF with %s model: local=%s, gcs=%s", self._model_type, pdf_path, gcs_uri
        )

        # Try primary model first
        primary_result = self._process_with_model(
            client=self._primary_client,
            model_type=self._model_type,
            pdf_path=pdf_path,
            gcs_uri=gcs_uri,
            timeout=timeout,
            save_response=save_response,
        )

        # If primary succeeded or fallback is disabled, return the result
        if primary_result.success or not self._enable_fallback:
            return primary_result

        # Primary failed and fallback is enabled - try fallback model
        primary_error = primary_result.error or "Unknown error"
        fallback_model_type = self._get_fallback_model_type()

        logger.warning(
            "Primary model (%s) failed: %s. Attempting fallback to %s model...",
            self._model_type,
            primary_error,
            fallback_model_type,
        )

        fallback_client = self._get_fallback_client()
        fallback_result = self._process_with_model(
            client=fallback_client,
            model_type=fallback_model_type,
            pdf_path=pdf_path,
            gcs_uri=gcs_uri,
            timeout=timeout,
            save_response=save_response,
            fallback_used=True,
            primary_error=primary_error,
        )

        if fallback_result.success:
            logger.info(
                "Fallback to %s model succeeded after primary (%s) failed",
                fallback_model_type,
                self._model_type,
            )
        else:
            logger.error(
                "Both primary (%s) and fallback (%s) models failed. "
                "Primary error: %s, Fallback error: %s",
                self._model_type,
                fallback_model_type,
                primary_error,
                fallback_result.error,
            )

        return fallback_result

    def _process_with_model(
        self,
        client: MistralOcrClient | LLMOcrClient,
        model_type: OcrModelType,
        pdf_path: str,
        gcs_uri: str | None,
        timeout: float | None,
        save_response: bool,
        fallback_used: bool = False,
        primary_error: str | None = None,
    ) -> NormalizedOcrResponse:
        """
        Process PDF with a specific model client.

        Args:
            client: The OCR client to use
            model_type: Type of the model ("mistral" or "llm")
            pdf_path: Local path to the PDF file (used by Mistral)
            gcs_uri: GCS URI of the PDF file (used by LLM)
            timeout: Request timeout in seconds
            save_response: Whether to save the raw response
            fallback_used: Whether this is a fallback attempt
            primary_error: Error from primary model (if fallback)

        Returns:
            NormalizedOcrResponse with processing result
        """
        try:
            if model_type == "llm":
                # LLM uses GCS URI if available, otherwise falls back to pdf_path
                llm_path = gcs_uri if gcs_uri else pdf_path
                return self._process_with_llm(
                    client=client,
                    pdf_path=llm_path,
                    timeout=timeout,
                    save_response=save_response,
                    fallback_used=fallback_used,
                    primary_error=primary_error,
                )
            else:
                # Mistral uses local pdf_path
                return self._process_with_mistral(
                    client=client,
                    pdf_path=pdf_path,
                    timeout=timeout,
                    save_response=save_response,
                    fallback_used=fallback_used,
                    primary_error=primary_error,
                )
        except Exception as e:
            logger.error("Exception during %s OCR processing: %s", model_type, e)
            return NormalizedOcrResponse.from_error(
                error=f"{model_type} processing exception: {e}",
                model_used=model_type,
                fallback_used=fallback_used,
                primary_error=primary_error,
            )

    def _process_with_mistral(
        self,
        client: MistralOcrClient | LLMOcrClient,
        pdf_path: str,
        timeout: float | None,
        save_response: bool,
        fallback_used: bool = False,
        primary_error: str | None = None,
    ) -> NormalizedOcrResponse:
        """
        Process PDF using Mistral OCR and normalize the response.

        Mistral returns: {"pages": [{"index": 0, "markdown": "..."}]}
        We normalize to: {"pages": [{"index": 0, "extracted_text": "..."}]}
        """
        result = client.process_pdf(  # type: ignore[union-attr]
            pdf_path=pdf_path,
            timeout=timeout,
            save_response=save_response,
        )

        if not result.get("success"):
            return NormalizedOcrResponse.from_error(
                error=result.get("error", "Unknown error"),
                model_used="mistral",
                fallback_used=fallback_used,
                primary_error=primary_error,
            )

        # Extract pages from Mistral response structure
        response = result.get("response", {})
        raw_pages = response.get("pages", [])

        # Normalize: convert 'markdown' -> 'extracted_text'
        normalized_pages = []
        for i, page in enumerate(raw_pages):
            if isinstance(page, dict):
                normalized_pages.append(
                    NormalizedPage(
                        index=page.get("index", i),
                        extracted_text=page.get("markdown", ""),
                    )
                )

        logger.debug("Normalized %d pages from Mistral response", len(normalized_pages))

        return NormalizedOcrResponse(
            success=True,
            pages=normalized_pages,
            error=None,
            model_used="mistral",
            fallback_used=fallback_used,
            primary_error=primary_error,
        )

    def _process_with_llm(
        self,
        client: MistralOcrClient | LLMOcrClient,
        pdf_path: str,
        timeout: float | None,
        save_response: bool,
        fallback_used: bool = False,
        primary_error: str | None = None,
    ) -> NormalizedOcrResponse:
        """
        Process PDF using LLM (Gemini) OCR and normalize the response.

        LLM returns: {"pages": [{"index": 0, "extracted_text": "..."}]}
        Already in normalized format, just validate and wrap.
        """
        result = client.process_document(  # type: ignore[union-attr]
            file_path=pdf_path,
            timeout=timeout,
            save_response=save_response,
        )

        if not result.get("success"):
            return NormalizedOcrResponse.from_error(
                error=result.get("error", "Unknown error"),
                model_used="llm",
                fallback_used=fallback_used,
                primary_error=primary_error,
            )

        # Extract pages from LLM response
        raw_pages = result.get("pages", [])

        # Normalize using from_dict to handle any format variations
        normalized_pages = []
        for i, page in enumerate(raw_pages):
            if isinstance(page, dict):
                normalized_pages.append(NormalizedPage.from_dict(page, default_index=i))
            elif isinstance(page, str):
                normalized_pages.append(NormalizedPage(index=i, extracted_text=page))

        logger.debug("Normalized %d pages from LLM response", len(normalized_pages))

        return NormalizedOcrResponse(
            success=True,
            pages=normalized_pages,
            error=None,
            model_used="llm",
            fallback_used=fallback_used,
            primary_error=primary_error,
        )


def get_ocr_model_client(
    model_type: OcrModelType = "mistral",
    client: MistralOcrClient | LLMOcrClient | None = None,
    enable_fallback: bool = True,
) -> OcrModelClient:
    """
    Create an OcrModelClient instance.

    Args:
        model_type: Type of OCR model ("mistral" or "llm")
        client: Optional pre-configured OCR client instance
        enable_fallback: Whether to enable automatic fallback to secondary model

    Returns:
        OcrModelClient instance with fallback support
    """
    return OcrModelClient(
        model_type=model_type,
        client=client,
        enable_fallback=enable_fallback,
    )
