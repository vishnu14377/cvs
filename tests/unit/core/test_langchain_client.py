"""
Unit tests for the LangChain client singleton.

Tests cover:
- Singleton initialization and thread safety
- Client configuration (model ID, project, region)
- Invoke and structured output methods
- LLM callback handler for logging
- Singleton reset functionality

Run with: pytest tests/unit/test_langchain_client.py -v
"""

import threading
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the LangChainClient singleton before and after each test."""
    from src.core.langchain_client import LangChainClient

    LangChainClient.reset()
    yield
    LangChainClient.reset()


@pytest.fixture
def mock_chat_model():
    """Create a mock ChatGoogleGenerativeAI instance."""
    mock = MagicMock()
    mock.invoke.return_value = AIMessage(content="Test response")
    return mock


@pytest.fixture
def mock_llm_config():
    """Mock the llm_config module."""
    with patch("src.core.langchain_client.llm_config") as mock_config:
        mock_config.GEMINI_MODEL_ID = "gemini-1.5-flash"
        mock_config.GCP_PROJECT = "test-project"
        mock_config.GCP_REGION = "us-central1"
        mock_config.LLM_TIMEOUT_SECONDS = 60
        yield mock_config


# =============================================================================
# Test: Singleton Behavior
# =============================================================================


class TestLangChainClientSingleton:
    """Tests for singleton pattern of LangChainClient."""

    def test_same_instance_returned(self, mock_llm_config):
        """Multiple instantiations should return the same instance."""
        with patch("src.core.langchain_client.ChatGoogleGenerativeAI") as mock_chat:
            from src.core.langchain_client import LangChainClient

            client1 = LangChainClient()
            client2 = LangChainClient()

            assert client1 is client2
            # Constructor should only be called once
            mock_chat.assert_called_once()

    def test_thread_safety(self, mock_llm_config):
        """Singleton should be thread-safe."""
        with patch("src.core.langchain_client.ChatGoogleGenerativeAI") as mock_chat:
            from src.core.langchain_client import LangChainClient

            instances = []

            def create_client():
                instances.append(LangChainClient())

            threads = [threading.Thread(target=create_client) for _ in range(10)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            # All instances should be the same
            assert all(inst is instances[0] for inst in instances)
            # ChatGoogleGenerativeAI should only be instantiated once
            mock_chat.assert_called_once()

    def test_reset_allows_new_instance(self, mock_llm_config):
        """After reset, a new instance should be created."""
        with patch("src.core.langchain_client.ChatGoogleGenerativeAI") as mock_chat:
            from src.core.langchain_client import LangChainClient

            client1 = LangChainClient()
            LangChainClient.reset()
            client2 = LangChainClient()

            # These should be different instances
            assert client1 is not client2
            # Constructor called twice (before and after reset)
            assert mock_chat.call_count == 2


# =============================================================================
# Test: Client Initialization
# =============================================================================


class TestLangChainClientInitialization:
    """Tests for LangChainClient initialization."""

    def test_init_uses_config_values(self, mock_llm_config):
        """Client should use config values for initialization."""
        with patch("src.core.langchain_client.ChatGoogleGenerativeAI") as mock_chat:
            from src.core.langchain_client import LangChainClient

            client = LangChainClient()

            # Check that config values are stored
            assert client.model_id == "gemini-1.5-flash"
            assert client.project_id == "test-project"
            assert client.location == "us-central1"

            # Check ChatGoogleGenerativeAI was called with correct args
            mock_chat.assert_called_once()
            call_kwargs = mock_chat.call_args[1]
            assert call_kwargs["model"] == "gemini-1.5-flash"
            assert call_kwargs["project"] == "test-project"
            assert call_kwargs["location"] == "us-central1"
            assert call_kwargs["vertexai"] is True
            assert call_kwargs["temperature"] == 0.1
            assert call_kwargs["max_retries"] == 3

    def test_client_property_returns_underlying_client(self, mock_llm_config):
        """The client property should return the ChatGoogleGenerativeAI instance."""
        with patch("src.core.langchain_client.ChatGoogleGenerativeAI") as mock_chat:
            mock_instance = MagicMock()
            mock_chat.return_value = mock_instance

            from src.core.langchain_client import LangChainClient

            client = LangChainClient()

            assert client.client is mock_instance


# =============================================================================
# Test: Invoke Method
# =============================================================================


class TestLangChainClientInvoke:
    """Tests for LangChainClient.invoke method."""

    def test_invoke_with_string_prompt(self, mock_llm_config, mock_chat_model):
        """Invoke with string should wrap in HumanMessage."""
        with patch(
            "src.core.langchain_client.ChatGoogleGenerativeAI", return_value=mock_chat_model
        ):
            from src.core.langchain_client import LangChainClient

            client = LangChainClient()
            response = client.invoke("Hello, world!")

            # Check invoke was called with HumanMessage
            mock_chat_model.invoke.assert_called_once()
            call_args = mock_chat_model.invoke.call_args[0][0]
            assert len(call_args) == 1
            assert isinstance(call_args[0], HumanMessage)
            assert call_args[0].content == "Hello, world!"

            assert response.content == "Test response"

    def test_invoke_with_message_list(self, mock_llm_config, mock_chat_model):
        """Invoke with message list should pass messages directly."""
        with patch(
            "src.core.langchain_client.ChatGoogleGenerativeAI", return_value=mock_chat_model
        ):
            from src.core.langchain_client import LangChainClient

            client = LangChainClient()
            messages = [HumanMessage(content="Test message")]
            response = client.invoke(messages)

            # Check invoke was called with original messages
            mock_chat_model.invoke.assert_called_once_with(messages)
            assert response.content == "Test response"


# =============================================================================
# Test: Structured Output
# =============================================================================


class TestLangChainClientStructuredOutput:
    """Tests for LangChainClient.with_structured_output method."""

    def test_with_structured_output_returns_runnable(self, mock_llm_config, mock_chat_model):
        """with_structured_output should return a structured runnable."""
        mock_structured_llm = MagicMock()
        mock_chat_model.with_structured_output.return_value = mock_structured_llm

        with patch(
            "src.core.langchain_client.ChatGoogleGenerativeAI", return_value=mock_chat_model
        ):
            from pydantic import BaseModel
            from src.core.langchain_client import LangChainClient

            class TestSchema(BaseModel):
                field: str

            client = LangChainClient()
            structured = client.with_structured_output(TestSchema)

            mock_chat_model.with_structured_output.assert_called_once_with(TestSchema)
            assert structured is mock_structured_llm


# =============================================================================
# Test: LLM Logging Callback
# =============================================================================


class TestLLMLoggingCallback:
    """Tests for LLMLoggingCallback handler."""

    def test_on_llm_start_logs_model_name(self):
        """on_llm_start should log the model name."""
        from src.core.langchain_client import LLMLoggingCallback

        callback = LLMLoggingCallback()
        serialized = {"kwargs": {"model": "gemini-1.5-flash"}}

        with patch("src.core.langchain_client.logger") as mock_logger:
            callback.on_llm_start(serialized, ["prompt"])
            mock_logger.debug.assert_called()

    def test_on_llm_end_logs_success(self):
        """on_llm_end should log successful completion."""
        from src.core.langchain_client import LLMLoggingCallback

        callback = LLMLoggingCallback()
        mock_response = MagicMock()

        with patch("src.core.langchain_client.logger") as mock_logger:
            callback.on_llm_end(mock_response)
            mock_logger.debug.assert_called()

    def test_on_llm_error_logs_warning(self):
        """on_llm_error should log a warning."""
        from src.core.langchain_client import LLMLoggingCallback

        callback = LLMLoggingCallback()
        error = ValueError("Test error")

        with patch("src.core.langchain_client.logger") as mock_logger:
            callback.on_llm_error(error)
            mock_logger.warning.assert_called()

    def test_on_retry_logs_info(self):
        """on_retry should log retry attempt."""
        from src.core.langchain_client import LLMLoggingCallback

        callback = LLMLoggingCallback()
        retry_state = MagicMock()

        with patch("src.core.langchain_client.logger") as mock_logger:
            callback.on_retry(retry_state)
            mock_logger.info.assert_called()


# =============================================================================
# Test: Convenience Function
# =============================================================================


class TestGetLangchainClient:
    """Tests for get_langchain_client function."""

    def test_get_langchain_client_returns_singleton(self, mock_llm_config):
        """get_langchain_client should return the singleton instance."""
        with patch("src.core.langchain_client.ChatGoogleGenerativeAI"):
            from src.core.langchain_client import LangChainClient, get_langchain_client

            client = get_langchain_client()
            direct_client = LangChainClient()

            assert client is direct_client


# =============================================================================
# Test: Stub Mode
# =============================================================================


class TestLangChainClientStubMode:
    """Verify VERTEX_AI_MODE=stub bypasses ChatGoogleGenerativeAI."""

    def setup_method(self) -> None:
        from src.core.langchain_client import LangChainClient

        LangChainClient.reset()

    def teardown_method(self) -> None:
        from src.core.langchain_client import LangChainClient

        LangChainClient.reset()

    def test_stub_mode_uses_stub_chat_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from src.core.stubs.chat_stub import StubChatModel

        monkeypatch.setattr("src.core.langchain_client.llm_config.VERTEX_AI_MODE", "stub")
        from src.core.langchain_client import LangChainClient

        LangChainClient.reset()
        client = LangChainClient()
        assert isinstance(client.client, StubChatModel)

    def test_real_mode_uses_chat_google_generative_ai(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("src.core.langchain_client.llm_config.VERTEX_AI_MODE", "real")
        with patch("src.core.langchain_client.ChatGoogleGenerativeAI") as mock_chat:
            mock_chat.return_value = MagicMock()
            from src.core.langchain_client import LangChainClient

            LangChainClient.reset()
            _ = LangChainClient()
            mock_chat.assert_called_once()
            call_kwargs = mock_chat.call_args.kwargs
            assert "timeout" in call_kwargs
            assert "callbacks" in call_kwargs
