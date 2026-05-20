"""
Vertex AI client singleton for making requests to Vertex AI.

This module provides a singleton wrapper around the google-cloud-aiplatform SDK's
PredictionServiceClient. It handles Application Default Credentials (ADC)
authentication and provides convenience methods for calling Vertex AI endpoints.

Usage:
    from src.core.vertex_ai_client import get_vertex_ai_client

    client = get_vertex_ai_client()
    response = client.raw_predict_mistral(payload)

Retry Behavior:
    The client uses google.api_core.retry.Retry for automatic retries on transient errors.

    By default, retries are enabled for:
    - UNAVAILABLE (503): Service temporarily unavailable
    - DEADLINE_EXCEEDED (504): Request took too long
    - RESOURCE_EXHAUSTED (429): Rate limiting / quota exceeded
    - INTERNAL (500): Internal server error

    Retry uses exponential backoff:
    - Initial delay: 0.1 seconds
    - Maximum delay: 60 seconds
    - Multiplier: 2x after each attempt
    - Default timeout: Configured via OCR_TIMEOUT_SECONDS

    Retry attempts are logged via the configured logger.

Retry Utilities:
    This module also exports retry helper functions:
    - create_retry_callback(operation_name): Create a logging callback for retries
    - create_retry_with_logging(...): Create a Retry object with logging enabled
    - DEFAULT_MISTRAL_RETRY: Pre-configured retry for Mistral OCR operations
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import Any

import google.api.httpbody_pb2 as httpbody_pb2
from google.api_core import gapic_v1
from google.api_core import retry as api_retry
from google.api_core.client_options import ClientOptions
from google.cloud import aiplatform_v1

from .config import vertex_config
from .logger import get_logger
from .stubs import stub_raw_predict_response

# Module logger
logger = get_logger(__name__)


# =============================================================================
# Retry Utilities
# =============================================================================


def create_retry_callback(
    operation_name: str = "Vertex AI model call",
    retry_logger: logging.Logger | None = None,
) -> Callable[[Exception], None]:
    """
    Create a callback function for logging retry attempts.

    Use this with google.api_core.retry.Retry's on_error parameter to log
    each retry attempt with details about the exception.

    Args:
        operation_name: Name of the operation being retried (for log messages)
        retry_logger: Logger instance to use; if None, uses module logger

    Returns:
        Callback function that accepts an exception and logs it

    Example:
        from google.api_core import retry as api_retry
        from src.core.vertex_ai_client import create_retry_callback

        retry = api_retry.Retry(
            initial=0.1,
            maximum=60.0,
            multiplier=2.0,
            timeout=300.0,
            on_error=create_retry_callback("Mistral OCR"),
        )
    """
    _logger = retry_logger or logger
    attempt_count = {"value": 0}  # Use dict to allow mutation in closure

    def log_retry(exception: Exception) -> None:
        attempt_count["value"] += 1
        _logger.warning(
            "[%s] Retry attempt %d - %s: %s",
            operation_name,
            attempt_count["value"],
            type(exception).__name__,
            str(exception),
        )

    return log_retry


def create_retry_with_logging(
    operation_name: str = "Operation",
    initial: float = 0.1,
    maximum: float = 60.0,
    multiplier: float = 2.0,
    timeout: float | None = None,
    retry_logger: logging.Logger | None = None,
) -> api_retry.Retry:
    """
    Create a google.api_core.retry.Retry object with logging enabled.

    This is a convenience function that creates a Retry with the on_error
    callback already configured for logging.

    Args:
        operation_name: Name of the operation (for log messages)
        initial: Initial delay between retries (seconds)
        maximum: Maximum delay between retries (seconds)
        multiplier: Factor to increase delay after each retry
        timeout: Total timeout for all retry attempts (seconds);
                 defaults to ocr_config.OCR_TIMEOUT_SECONDS
        retry_logger: Logger instance to use; if None, uses module logger

    Returns:
        Configured Retry object with logging

    Example:
        from src.core.vertex_ai_client import create_retry_with_logging

        retry = create_retry_with_logging(
            operation_name="Mistral OCR",
            initial=0.5,
            maximum=30.0,
            timeout=300.0,
        )
        response = client.raw_predict_mistral(payload, retry=retry)
    """
    _timeout = timeout if timeout is not None else float(vertex_config.TIMEOUT_SECONDS)

    return api_retry.Retry(
        initial=initial,
        maximum=maximum,
        multiplier=multiplier,
        timeout=_timeout,
        on_error=create_retry_callback(operation_name, retry_logger),
    )


# Default retry with logging.
DEFAULT_RETRY = create_retry_with_logging(
    operation_name="Model calling using vertex AI client",
    initial=vertex_config.INITIAL_DELAY,
    maximum=vertex_config.MAX_DELAY,
    timeout=float(vertex_config.TIMEOUT_SECONDS),
)


# =============================================================================
# Vertex AI Client
# =============================================================================


class VertexAIClient:
    """
    Singleton client for Vertex AI PredictionService.

    Wraps PredictionServiceClient and provides helper methods for
    calling Model Garden models (e.g., Mistral OCR).

    Retry Behavior:
        By default, transient errors (503 Unavailable, 429 Rate Limited, etc.)
        are automatically retried with exponential backoff. Retry attempts
        are logged with the operation name and exception details.
    """

    _instance: VertexAIClient | None = None

    def __init__(
        self,
        project_id: str | None = None,
        region: str | None = None,
    ):
        """
        Initialize the Vertex AI client.

        Args:
            project_id: GCP project ID (defaults to vertex_config.GCP_PROJECT)
            region: GCP region (defaults to vertex_config.GCP_REGION)
        """
        self.project_id = project_id or vertex_config.GCP_PROJECT
        self.region = region or vertex_config.GCP_REGION

        if vertex_config.VERTEX_AI_MODE == "stub":
            logger.warning(
                "VERTEX_AI_MODE=stub — VertexAIClient will not construct "
                "PredictionServiceClient (no Vertex AI calls will be made)"
            )
            self._prediction_client = None
            logger.info(
                "VertexAIClient initialized successfully (stub mode — no PredictionServiceClient)"
            )
            return

        if not self.project_id:
            logger.error("project_id not provided and GCP_PROJECT_ID env var not set")
            raise ValueError("project_id must be provided or set via GCP_PROJECT_ID env var")

        logger.info(
            "Initializing VertexAIClient for project=%s, region=%s",
            self.project_id,
            self.region,
        )

        api_endpoint = f"{self.region}-aiplatform.googleapis.com"
        client_options = ClientOptions(api_endpoint=api_endpoint)
        self._prediction_client = aiplatform_v1.PredictionServiceClient(
            client_options=client_options
        )
        logger.debug("PredictionServiceClient initialized with endpoint=%s", api_endpoint)

    @classmethod
    def get_instance(
        cls,
        project_id: str | None = None,
        region: str | None = None,
    ) -> VertexAIClient:
        """
        Get or create the singleton instance.

        Args:
            project_id: GCP project ID
            region: GCP region

        Returns:
            VertexAIClient singleton instance
        """
        if cls._instance is None:
            logger.debug("Creating new VertexAIClient singleton instance")
            cls._instance = cls(project_id=project_id, region=region)
        else:
            logger.debug("Returning existing VertexAIClient singleton instance")
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (useful for testing)."""
        logger.debug("Resetting VertexAIClient singleton instance")
        cls._instance = None

    @property
    def prediction_client(self) -> aiplatform_v1.PredictionServiceClient | None:
        """Return the underlying PredictionServiceClient (None in stub mode)."""
        return self._prediction_client

    def get_model_endpoint(self, model_id: str | None = None, publisher: str | None = None) -> str:
        """
        Build the endpoint resource string for a Mistral Model Garden model.

        For Model Garden publisher models, the endpoint format is:
        projects/{project}/locations/{location}/publishers/{publisher}/models/{model}

        Args:
            model_id: Name of the model (e.g., "mistral-7b-instruct-v0.1.Q4_0.gguf")
            publisher: Name of the publisher (e.g., "google")

        Returns:
            Fully qualified endpoint resource string
        """
        model_id = model_id
        publisher = publisher

        endpoint = (
            f"projects/{self.project_id}/locations/{self.region}/"
            f"publishers/{publisher}/models/{model_id}"
        )

        logger.debug("Built Mistral endpoint: %s", endpoint)
        return endpoint

    def generate(
        self,
        payload: dict[str, Any],
        model_id: str | None = None,
        publisher: str | None = None,
        timeout: float | None = None,
        retry: api_retry.Retry | gapic_v1.method._MethodDefault | None = None,
    ) -> dict[str, Any]:
        """
        Call Model model via raw_predict.

        Uses PredictionServiceClient.raw_predict() with an HttpBody containing
        the JSON payload.

        Args:
            payload: Dictionary payload to send (will be JSON-encoded)
            model_id: Name of the model
            timeout: Request timeout in seconds (defaults to vertex_config.TIMEOUT_SECONDS)
            retry: Retry configuration. Options:
                - None (default): Use DEFAULT_MISTRAL_RETRY with logging enabled
                - gapic_v1.method.DEFAULT: SDK default retry (no custom logging)
                - api_retry.Retry(...): Custom retry configuration
                - create_retry_with_logging(...): Custom retry with logging

        Returns:
            Parsed JSON response as a dictionary

        Raises:
            google.api_core.exceptions.GoogleAPICallError: If the request fails after retries
            json.JSONDecodeError: If response is not valid JSON
        """
        if vertex_config.VERTEX_AI_MODE == "stub":
            logger.info("generate (stub mode): model=%s — returning stub payload", model_id)
            return stub_raw_predict_response(payload)

        # Use default retry with logging if not specified
        if retry is None:
            retry = DEFAULT_RETRY

        model_id = model_id
        publisher = publisher
        endpoint = self.get_model_endpoint(model_id, publisher)
        timeout = timeout or vertex_config.TIMEOUT_SECONDS

        logger.info(
            "Calling generate: model=%s, timeout=%s",
            model_id,
            timeout,
        )
        logger.debug("Endpoint: %s", endpoint)

        # Ensure model is in payload
        payload_copy = dict(payload)
        if "model" not in payload_copy:
            payload_copy["model"] = model_id

        # Build HttpBody with JSON payload
        http_body = httpbody_pb2.HttpBody(
            content_type="application/json",
            data=json.dumps(payload_copy).encode("utf-8"),
        )

        logger.debug("Payload size: %d bytes", len(http_body.data))

        if self._prediction_client is None:
            raise RuntimeError(
                "VertexAIClient._prediction_client is None in real mode — "
                "stub/real branching logic is broken"
            )

        try:
            # Call raw_predict with retry configuration
            response: httpbody_pb2.HttpBody = self._prediction_client.raw_predict(
                endpoint=endpoint,
                http_body=http_body,
                timeout=timeout,
                retry=retry,
            )

            # Parse response
            response_data = response.data.decode("utf-8")
            result = json.loads(response_data)

            logger.info("generate succeeded, response size: %d bytes", len(response_data))
            logger.debug("Response content_type: %s", response.content_type)

            return result

        except Exception as e:
            logger.error(
                "generate failed: %s: %s",
                type(e).__name__,
                str(e),
            )
            raise


def get_vertex_ai_client(
    project_id: str | None = None,
    region: str | None = None,
) -> VertexAIClient:
    """
    Get or create the VertexAIClient singleton.

    Args:
        project_id: GCP project ID (defaults to vertex_config.GCP_PROJECT)
        region: GCP region (defaults to vertex_config.GCP_REGION)

    Returns:
        VertexAIClient singleton instance
    """
    return VertexAIClient.get_instance(project_id=project_id, region=region)
