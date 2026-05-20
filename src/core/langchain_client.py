"""
LangChain client singleton using ChatGoogleGenerativeAI with Vertex AI backend.

Provides a thread-safe singleton instance of ChatGoogleGenerativeAI for consistent
LLM access across the application. Uses Application Default Credentials (ADC) for
authentication with Vertex AI backend.

Usage:
    from src.core.langchain_client import LangChainClient

    # Get singleton instance
    client = LangChainClient()

    # Invoke with a simple prompt
    response = client.invoke("Hello, world!")
    print(response.content)

    # Or use structured output
    structured_llm = client.with_structured_output(MyPydanticModel)
    result = structured_llm.invoke(messages)

Authentication:
    Uses Application Default Credentials (ADC). If authentication fails,
    run 'gcloud auth application-default login' to reauthenticate.
"""

from __future__ import annotations

import threading
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.outputs import LLMResult
from langchain_google_genai import ChatGoogleGenerativeAI
from src.core.config import llm_config
from src.core.logger import get_logger
from src.core.stubs import StubChatModel

logger = get_logger(__name__)


class LLMLoggingCallback(BaseCallbackHandler):
    """
    Callback handler to log LLM requests, retries, and errors.

    This provides visibility into:
    - When LLM requests start and complete
    - When errors occur (which may trigger retries)
    - Retry attempts
    """

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: Any = None,
        parent_run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        """Log when LLM starts processing."""
        model_name = serialized.get("kwargs", {}).get("model", "unknown")
        logger.debug("LLM request started: model=%s", model_name)

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: Any = None,
        parent_run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        """Log when LLM completes successfully."""
        logger.debug("LLM request completed successfully")

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: Any = None,
        parent_run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        """Log when an LLM error occurs (before retry)."""
        logger.warning(
            "LLM error occurred (may retry): %s - %s",
            type(error).__name__,
            str(error),
        )

    def on_retry(
        self,
        retry_state: Any,
        *,
        run_id: Any = None,
        parent_run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        """Log retry attempts."""
        logger.info("LLM retry attempt: %s", retry_state)


class LangChainClient:
    """
    Thread-safe singleton LangChain client using ChatGoogleGenerativeAI.

    Uses Vertex AI backend with Application Default Credentials (ADC).
    The singleton pattern ensures only one client instance exists across
    the entire application, avoiding redundant initialization.

    Example:
        >>> client = LangChainClient()
        >>> response = client.invoke("What is 2 + 2?")
        >>> print(response.content)
        "4"

        >>> # Use structured output
        >>> structured = client.with_structured_output(MyModel)
        >>> result = structured.invoke(messages)
    """

    _instance: LangChainClient | None = None
    _lock: threading.Lock = threading.Lock()
    _initialized: bool = False

    def __new__(cls) -> LangChainClient:
        """Create or return the singleton instance."""
        if cls._instance is None:
            with cls._lock:
                # Double-checked locking for thread safety
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize the LangChain client (only runs once)."""
        # Skip if already initialized (singleton pattern)
        if LangChainClient._initialized:
            return

        with LangChainClient._lock:
            if LangChainClient._initialized:
                return

            self._model_id = llm_config.GEMINI_MODEL_ID
            self._project_id = llm_config.GCP_PROJECT
            self._location = llm_config.GCP_REGION
            self._temperature = 0.1
            self._max_retries = 3

            # Create callback handler for logging retries and errors
            self._callbacks = [LLMLoggingCallback()]

            logger.info(
                "Initializing LangChainClient: model=%s, project=%s, location=%s",
                self._model_id,
                self._project_id,
                self._location,
            )

            self._client: ChatGoogleGenerativeAI | StubChatModel
            if llm_config.VERTEX_AI_MODE == "stub":
                logger.warning(
                    "VERTEX_AI_MODE=stub — LangChainClient using StubChatModel "
                    "(no Vertex AI calls will be made)"
                )
                self._client = StubChatModel()
            else:
                self._client = ChatGoogleGenerativeAI(
                    model=self._model_id,
                    project=self._project_id,
                    location=self._location,
                    vertexai=True,
                    temperature=self._temperature,
                    max_retries=self._max_retries,
                    timeout=llm_config.LLM_TIMEOUT_SECONDS,
                    callbacks=self._callbacks,
                )

            LangChainClient._initialized = True
            mode_desc = (
                "stub (StubChatModel)"
                if llm_config.VERTEX_AI_MODE == "stub"
                else "Vertex AI backend with ADC"
            )
            logger.info("LangChainClient initialized successfully (%s)", mode_desc)

    @property
    def client(self) -> ChatGoogleGenerativeAI | StubChatModel:
        """Get the underlying chat model client."""
        return self._client

    @property
    def model_id(self) -> str:
        """Get the model ID."""
        return self._model_id

    @property
    def project_id(self) -> str | None:
        """Get the GCP project ID."""
        return self._project_id

    @property
    def location(self) -> str:
        """Get the GCP location/region."""
        return self._location

    def invoke(
        self,
        messages: str | list[BaseMessage],
    ) -> BaseMessage:
        """
        Invoke the Gemini model with messages.

        Args:
            messages: A string prompt or list of BaseMessage objects.

        Returns:
            The model's response as a BaseMessage.
        """
        if isinstance(messages, str):
            messages = [HumanMessage(content=messages)]

        return self._client.invoke(messages)

    def with_structured_output(self, schema):
        """
        Create a runnable that returns structured output.

        Uses LangChain's built-in structured output parsing with Pydantic models.

        Args:
            schema: Pydantic model class for the expected output structure.

        Returns:
            A runnable that returns instances of the schema directly.

        Example:
            >>> structured_llm = client.with_structured_output(DocumentExtraction)
            >>> result: DocumentExtraction = structured_llm.invoke(messages)
        """
        return self._client.with_structured_output(schema)

    @classmethod
    def reset(cls) -> None:
        """
        Reset the singleton instance.

        Useful for testing or when you need to reinitialize with different parameters.
        After reset, the next instantiation will create a fresh client.
        """
        with cls._lock:
            cls._instance = None
            cls._initialized = False
            logger.debug("LangChainClient singleton reset")


# Convenience function for backward compatibility
def get_langchain_client() -> LangChainClient:
    """
    Get the singleton LangChainClient instance.

    This is a convenience function for backward compatibility.
    Prefer using LangChainClient() directly.

    Returns:
        The singleton LangChainClient instance.
    """
    return LangChainClient()
