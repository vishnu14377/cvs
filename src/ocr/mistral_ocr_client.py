"""
Mistral OCR client for processing PDF documents.

Uses the VertexAIClient singleton to call Mistral OCR model via raw_predict.

Retry Behavior:
    By default, transient errors are automatically retried with exponential backoff.
    Retry attempts are logged with the operation name and exception details.
    You can customize or disable retries by passing the `retry` parameter.
"""

from __future__ import annotations

import base64
import json
import os
from typing import Any

from google.api_core import gapic_v1
from google.api_core import retry as api_retry
from src.core.config import mistral_ocr_config
from src.core.logger import get_logger
from src.core.vertex_ai_client import get_vertex_ai_client

# Module logger
logger = get_logger(__name__)


class MistralOcrClient:
    """
    Client for calling Mistral OCR model on Vertex AI.

    Processes PDF files by encoding them as base64 and sending to the
    Mistral OCR model via the Vertex AI raw_predict endpoint.

    Retry Behavior:
        By default, transient errors (503 Unavailable, 429 Rate Limited, etc.)
        are automatically retried with exponential backoff. Retry attempts
        are logged with the operation name and exception details.
    """

    def __init__(
        self,
        model_id: str | None = None,
        project_id: str | None = None,
        region: str | None = None,
    ):
        """
        Initialize the Mistral OCR client.

        Args:
            model_id: Mistral model ID (defaults to mistral_ocr_config.MISTRAL_MODEL_ID)
            project_id: GCP project ID (defaults to config)
            region: GCP region (defaults to config)
        """
        self.model_id = model_id or mistral_ocr_config.MISTRAL_MODEL_ID
        self.model_publisher = mistral_ocr_config.MISTRAL_PUBLISHER
        self._vertex_client = get_vertex_ai_client(project_id=project_id, region=region)

        logger.info("MistralOcrClient initialized with model_id=%s", self.model_id)

    def process_pdf(
        self,
        pdf_path: str,
        timeout: float | None = None,
        save_response: bool = True,
        retry: api_retry.Retry | gapic_v1.method._MethodDefault | None = None,
    ) -> dict[str, Any]:
        """
        Process a PDF file using Mistral OCR.

        Args:
            pdf_path: Path to the PDF file to process
            timeout: Request timeout in seconds (defaults to mistral_ocr_config.MODEL_TIMEOUT_SECONDS)
            save_response: Whether to save the response to a JSON file
            retry: Retry configuration for transient errors. Options:
                - None (default): Use DEFAULT_MISTRAL_RETRY with logging
                - gapic_v1.method.DEFAULT: SDK default retry (no custom logging)
                - api_retry.Retry(...): Custom retry configuration
                - create_retry_with_logging(...): Custom retry with logging

                Example custom retry:
                    from src.core import create_retry_with_logging
                    custom_retry = create_retry_with_logging(
                        operation_name="Custom OCR",
                        initial=0.5,
                        maximum=30.0,
                        timeout=300.0,
                    )
                    result = client.process_pdf("doc.pdf", retry=custom_retry)

        Returns:
            Result dictionary with keys:
                - success: bool indicating if processing succeeded
                - pdf_path: path to the processed PDF
                - response: parsed response from Mistral OCR (if successful)
                - output_file: path to saved response file (if save_response=True)
                - error: error message (if failed)
        """
        # Use default retry with logging if not specified

        result: dict[str, Any] = {
            "success": False,
            "pdf_path": pdf_path,
            "response": None,
            "output_file": None,
            "error": None,
        }

        logger.info("Processing PDF: %s", pdf_path)

        # Validate file exists
        if not os.path.exists(pdf_path):
            error_msg = f"File not found: {pdf_path}"
            logger.error(error_msg)
            result["error"] = error_msg
            return result

        # Read and encode PDF
        try:
            with open(pdf_path, "rb") as f:
                pdf_bytes = f.read()

            file_size_mb = len(pdf_bytes) / (1024 * 1024)
            logger.debug("PDF read successfully: size=%.2f MB", file_size_mb)

            base64_encoded = base64.b64encode(pdf_bytes).decode("utf-8")
            logger.debug("PDF base64 encoded: length=%d chars", len(base64_encoded))

        except Exception as e:
            error_msg = f"Failed to read PDF: {e}"
            logger.error(error_msg, exc_info=True)
            result["error"] = error_msg
            return result

        # Build payload for Mistral OCR
        document_url = f"data:application/pdf;base64,{base64_encoded}"
        payload = {
            "model": self.model_id,
            "document": {
                "type": "document_url",
                "document_url": document_url,
            },
        }

        # Call Mistral OCR via Vertex AI
        timeout = timeout or mistral_ocr_config.MODEL_TIMEOUT_SECONDS
        logger.info("Calling Mistral OCR API with timeout=%s", timeout)

        try:
            response = self._vertex_client.generate(
                payload=payload,
                model_id=self.model_id,
                publisher=self.model_publisher,
                timeout=timeout,
                retry=retry,
            )

            result["success"] = True
            result["response"] = response
            logger.info("Mistral OCR succeeded for: %s", pdf_path)

            # Save response to file
            if save_response:
                output_file = f"{os.path.splitext(pdf_path)[0]}_mistral_response.json"
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(response, f, indent=2, ensure_ascii=False)
                result["output_file"] = output_file
                logger.info("Response saved to: %s", output_file)

        except Exception as e:
            error_msg = f"Prediction failed: {e}"
            logger.error(error_msg, exc_info=True)
            result["error"] = error_msg

        return result


if __name__ == "__main__":
    import sys

    # Configure logging for CLI usage
    logger.info("Starting Mistral OCR client CLI")

    # Example usage
    ocr_client = MistralOcrClient()
    test_pdf_path = "test1.pdf" if len(sys.argv) < 2 else sys.argv[1]

    logger.info("Processing file: %s", test_pdf_path)
    result = ocr_client.process_pdf(test_pdf_path)

    # Output result as JSON
    output = json.dumps(result, indent=2, ensure_ascii=False)
    logger.info("Result:\n%s", output)
