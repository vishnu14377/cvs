"""
LLM OCR client using LangChain for processing PDF documents stored in GCS.

Uses LangChain with Vertex AI's Gemini models for document understanding
and OCR tasks. PDF files are referenced by GCS URI and sent directly to the model.

LangChain provides built-in:
- Structured output parsing with `.with_structured_output()` - returns Pydantic objects directly
- Automatic retry logic with exponential backoff
- Error handling for various LLM failures
- Output validation via Pydantic

Usage:
    from src.ocr.llm_ocr_client import LLMOcrClient

    client = LLMOcrClient()
    result = client.process_document("gs://bucket/path/document.pdf")

    if result["success"]:
        for page in result["pages"]:
            print(f"Page {page['index']}: {page['extracted_text'][:100]}...")
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError
from src.core.gcs_client import is_gcs_uri
from src.core.langchain_client import LangChainClient
from src.core.logger import get_logger
from src.ocr.data_models.llm_response import DocumentExtraction

logger = get_logger(__name__)

# PDF MIME type constant
PDF_MIME_TYPE = "application/pdf"


# =============================================================================
# System Prompt (simplified - no format instructions needed with structured output)
# =============================================================================

SYSTEM_PROMPT = """You are an expert document analysis assistant specialized in extracting text from PDF documents with high accuracy.

Your task is to extract ALL visible text from the document, page by page, preserving the structure and formatting.

Instructions:
1. Process each page of the document separately.
2. Extract all visible text accurately, including headers, footers, and page numbers.
3. Preserve tables by formatting them with aligned columns using spaces.
4. Maintain lists with proper indentation and bullet points.
5. Keep the logical reading order of the document.
6. For forms, extract both field labels and their values."""


# =============================================================================
# Helper Functions
# =============================================================================


def is_pdf_file(file_path: str) -> bool:
    """Check if a file path is a PDF based on extension."""
    return file_path.lower().endswith(".pdf")


# =============================================================================
# LLM OCR LangChain Client
# =============================================================================


class LLMOcrClient:
    """
    Client for calling Gemini model via LangChain for OCR tasks.

    Uses LangChain's ChatGoogleGenerativeAI with Vertex AI backend for:
    - Built-in retry logic with exponential backoff
    - Structured output parsing with `.with_structured_output()` - no manual parsing!
    - Automatic error handling and validation
    - GCS URI support (files sent directly without download)
    """

    def __init__(
        self,
        temperature: float = 0.1,
        max_output_tokens: int | None = None,
    ):
        """
        Initialize the LangChain LLM OCR client.

        Args:
            temperature: Sampling temperature (lower = more deterministic)
            max_output_tokens: Maximum tokens in response
        """
        self._temperature = temperature
        self._max_output_tokens = max_output_tokens

        # Get the singleton LangChain client
        self._langchain = LangChainClient()

        # Create structured output LLM - returns DocumentExtraction directly!
        # LangChain handles all JSON parsing and Pydantic validation automatically
        self._structured_llm = self._langchain.with_structured_output(DocumentExtraction)

        self._system_prompt = SYSTEM_PROMPT

        logger.info("LLMOcrClient initialized with model: %s", self._langchain.model_id)

    def process_document(
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

        Uses `.with_structured_output()` to get parsed Pydantic objects directly
        from the LLM - no manual JSON parsing needed!

        Args:
            file_path: GCS URI (gs://bucket/path/document.pdf)
            prompt: Custom prompt for extraction (defaults to standard OCR prompt)
            temperature: Override temperature for this request
            max_output_tokens: Override max tokens for this request
            timeout: Override timeout for this request (seconds)
            save_response: Whether to save the response (not used, for API compatibility)

        Returns:
            Result dictionary with keys:
                - success: bool indicating if processing succeeded
                - pages: list of page dictionaries with 'index' and 'extracted_text'
                - error: error message (if failed, None on success)
        """
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

        logger.info("Processing GCS PDF document: %s", file_path)

        try:
            # Build messages
            messages = self._build_messages(file_path, prompt)

            # Get structured LLM (optionally with config overrides)
            structured_llm = self._get_structured_llm(temperature, max_output_tokens, timeout)

            logger.info("Calling Gemini API via LangChain with structured output")

            # Invoke the LLM - returns DocumentExtraction directly!
            # LangChain handles all parsing and validation automatically
            extraction: DocumentExtraction = structured_llm.invoke(messages)

            # Convert Pydantic model to dict format
            pages = [
                {"index": page.page_number, "extracted_text": page.page_text, "page_summary": page.page_summary, "page_insight": page.page_insight}
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
        # LangChain's ChatGoogleGenerativeAI supports file_uri for GCS files
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

        Returns an LLM configured with `.with_structured_output()` that
        returns DocumentExtraction objects directly.

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


# =============================================================================
# Singleton Access
# =============================================================================

# Singleton instance
_llm_ocr_client: LLMOcrClient | None = None


def get_llm_ocr_client(
    temperature: float = 0.1,
    max_output_tokens: int | None = None,
) -> LLMOcrClient:
    """
    Get or create the singleton LLMOcrClient instance.

    Args:
        temperature: Sampling temperature (only used on first call)
        max_output_tokens: Maximum tokens in response (only used on first call)

    Returns:
        LLMOcrClient instance
    """
    global _llm_ocr_client

    if _llm_ocr_client is None:
        _llm_ocr_client = LLMOcrClient(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )

    return _llm_ocr_client


def reset_llm_ocr_client() -> None:
    """Reset the singleton client for testing."""
    global _llm_ocr_client
    _llm_ocr_client = None
    logger.debug("LLMOcrClient singleton reset")


if __name__ == "__main__":
    logger.info("Starting LLM OCR client CLI")

    gcs_uri = "gs://care_connect_ai_initiatives/care_connect_ai_initiatives/adr_ai_agent/test3/tmp/test1_p1-10.pdf"

    client = get_llm_ocr_client()

    logger.info("Processing GCS file: %s", gcs_uri)
    result = client.process_document(gcs_uri)

    print(result)
