"""
Unit tests for the Vertex AI client.

Tests cover:
- Client initialization and singleton behavior
- Retry callback and logging
- Retry configuration
- generate method
- Error handling

Run with: pytest tests/unit/test_vertex_ai_client.py -v
"""

import json
from unittest.mock import MagicMock, patch

import google.api.httpbody_pb2 as httpbody_pb2
import pytest
from google.api_core import exceptions as google_exceptions
from src.core.vertex_ai_client import (
    DEFAULT_RETRY,
    VertexAIClient,
    create_retry_callback,
    create_retry_with_logging,
    get_vertex_ai_client,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_prediction_client():
    """Create a mock PredictionServiceClient."""
    return MagicMock()


@pytest.fixture
def mock_vertex_client(mock_prediction_client):
    """Create a VertexAIClient with mocked prediction client."""
    # Reset singleton before test
    VertexAIClient.reset_instance()

    with patch("src.core.vertex_ai_client.aiplatform_v1.PredictionServiceClient") as mock_cls:
        mock_cls.return_value = mock_prediction_client

        client = get_vertex_ai_client(project_id="test-project", region="us-central1")
        yield client, mock_prediction_client

    # Cleanup after test
    VertexAIClient.reset_instance()


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the singleton before and after each test."""
    VertexAIClient.reset_instance()
    yield
    VertexAIClient.reset_instance()


# =============================================================================
# Test: Client Initialization
# =============================================================================


class TestVertexAIClientInitialization:
    """Tests for VertexAIClient initialization."""

    def test_init_with_project_and_region(self):
        """Client should initialize with provided project and region."""
        with patch("src.core.vertex_ai_client.aiplatform_v1.PredictionServiceClient"):
            client = VertexAIClient(project_id="my-project", region="us-west1")

            assert client.project_id == "my-project"
            assert client.region == "us-west1"

    def test_init_without_project_raises_error(self):
        """Client should raise ValueError if no project_id is provided."""
        with patch("src.core.vertex_ai_client.vertex_config") as mock_config:
            mock_config.GCP_PROJECT = None

            with pytest.raises(ValueError, match="project_id must be provided"):
                VertexAIClient(project_id=None)

    def test_init_uses_config_defaults(self):
        """Client should use config values when not provided."""
        with (
            patch("src.core.vertex_ai_client.aiplatform_v1.PredictionServiceClient"),
            patch("src.core.vertex_ai_client.vertex_config") as mock_config,
        ):
            mock_config.GCP_PROJECT = "config-project"
            mock_config.GCP_REGION = "config-region"

            client = VertexAIClient()

            assert client.project_id == "config-project"
            assert client.region == "config-region"

    def test_prediction_client_property(self, mock_vertex_client):
        """prediction_client property should return the GAPIC client."""
        client, mock_prediction_client = mock_vertex_client

        assert client.prediction_client == mock_prediction_client


# =============================================================================
# Test: Singleton Behavior
# =============================================================================


class TestVertexAIClientSingleton:
    """Tests for singleton behavior."""

    def test_get_instance_returns_singleton(self):
        """get_instance should return the same instance."""
        with patch("src.core.vertex_ai_client.aiplatform_v1.PredictionServiceClient"):
            client1 = get_vertex_ai_client(project_id="test-project")
            client2 = get_vertex_ai_client()

            assert client1 is client2

    def test_reset_instance_clears_singleton(self):
        """reset_instance should clear the singleton."""
        with patch("src.core.vertex_ai_client.aiplatform_v1.PredictionServiceClient"):
            client1 = get_vertex_ai_client(project_id="test-project")

            VertexAIClient.reset_instance()

            client2 = get_vertex_ai_client(project_id="test-project")

            assert client1 is not client2


# =============================================================================
# Test: Retry Callback
# =============================================================================


class TestRetryCallback:
    """Tests for the retry callback function."""

    def test_callback_increments_counter(self):
        """Callback should increment attempt counter."""
        callback = create_retry_callback("Test Operation")

        exc1 = google_exceptions.ServiceUnavailable("Error 1")
        exc2 = google_exceptions.ServiceUnavailable("Error 2")
        exc3 = google_exceptions.ServiceUnavailable("Error 3")

        # Call three times - should not raise
        callback(exc1)
        callback(exc2)
        callback(exc3)

    def test_callback_logs_warning(self, caplog):
        """Callback should log warning with exception details."""
        import logging

        caplog.set_level(logging.WARNING)

        callback = create_retry_callback("My Operation")
        exc = google_exceptions.DeadlineExceeded("Request timed out")

        callback(exc)

        assert "My Operation" in caplog.text
        assert "Retry attempt 1" in caplog.text
        assert "DeadlineExceeded" in caplog.text
        assert "Request timed out" in caplog.text

    def test_callback_with_custom_logger(self, caplog):
        """Callback should use custom logger when provided."""
        import logging

        custom_logger = logging.getLogger("custom.test.logger")
        caplog.set_level(logging.WARNING)

        callback = create_retry_callback("Custom Op", retry_logger=custom_logger)
        callback(google_exceptions.ServiceUnavailable("Test"))

        assert "Custom Op" in caplog.text


# =============================================================================
# Test: Retry Configuration
# =============================================================================


class TestRetryWithLogging:
    """Tests for create_retry_with_logging function."""

    def test_creates_retry_with_defaults(self):
        """Should create retry with default values."""
        retry = create_retry_with_logging(operation_name="Test")

        assert retry is not None
        assert retry._initial == 0.1
        assert retry._maximum == 60.0
        assert retry._multiplier == 2.0

    def test_creates_retry_with_custom_values(self):
        """Should create retry with custom values."""
        retry = create_retry_with_logging(
            operation_name="Custom",
            initial=0.5,
            maximum=30.0,
            multiplier=1.5,
            timeout=120.0,
        )

        assert retry._initial == 0.5
        assert retry._maximum == 30.0
        assert retry._multiplier == 1.5
        assert retry._timeout == 120.0

    def test_default_retry_exists(self):
        """DEFAULT_RETRY should be configured."""
        assert DEFAULT_RETRY is not None
        assert DEFAULT_RETRY._initial == 1.0


# =============================================================================
# Test: get_model_endpoint
# =============================================================================


class TestGetModelEndpoint:
    """Tests for endpoint building."""

    def test_builds_correct_endpoint(self, mock_vertex_client):
        """Should build correct endpoint string."""
        client, _ = mock_vertex_client

        endpoint = client.get_model_endpoint(model_id="mistral-ocr-2505", publisher="mistralai")

        assert endpoint == (
            "projects/test-project/locations/us-central1/"
            "publishers/mistralai/models/mistral-ocr-2505"
        )

    def test_builds_endpoint_with_different_publisher(self, mock_vertex_client):
        """Should build endpoint with different publisher."""
        client, _ = mock_vertex_client

        endpoint = client.get_model_endpoint(model_id="gemini-pro", publisher="google")

        assert endpoint == (
            "projects/test-project/locations/us-central1/publishers/google/models/gemini-pro"
        )


# =============================================================================
# Test: generate
# =============================================================================


class TestGenerate:
    """Tests for generate method."""

    def test_successful_prediction(self, mock_vertex_client):
        """Successful prediction should return parsed response."""
        client, mock_prediction_client = mock_vertex_client

        # Mock successful response
        mock_response = httpbody_pb2.HttpBody(
            content_type="application/json",
            data=json.dumps({"pages": [{"text": "Hello"}]}).encode("utf-8"),
        )
        mock_prediction_client.raw_predict.return_value = mock_response

        payload = {"document": {"type": "test"}}
        result = client.generate(payload, model_id="test-model", publisher="test-publisher")

        assert result == {"pages": [{"text": "Hello"}]}
        mock_prediction_client.raw_predict.assert_called_once()

    def test_adds_model_to_payload(self, mock_vertex_client):
        """Should add model to payload if not present."""
        client, mock_prediction_client = mock_vertex_client

        mock_response = httpbody_pb2.HttpBody(
            content_type="application/json",
            data=b'{"result": "ok"}',
        )
        mock_prediction_client.raw_predict.return_value = mock_response

        payload = {"document": {"type": "test"}}
        client.generate(payload, model_id="my-model", publisher="test-publisher")

        # Check the http_body that was sent
        call_args = mock_prediction_client.raw_predict.call_args
        http_body = call_args.kwargs.get("http_body") or call_args[1].get("http_body")
        sent_payload = json.loads(http_body.data.decode("utf-8"))

        assert sent_payload["model"] == "my-model"

    def test_uses_provided_timeout(self, mock_vertex_client):
        """Should use provided timeout."""
        client, mock_prediction_client = mock_vertex_client

        mock_response = httpbody_pb2.HttpBody(
            content_type="application/json",
            data=b"{}",
        )
        mock_prediction_client.raw_predict.return_value = mock_response

        client.generate({}, model_id="test-model", publisher="test-publisher", timeout=123)

        call_args = mock_prediction_client.raw_predict.call_args
        assert call_args.kwargs.get("timeout") == 123

    def test_uses_default_retry(self, mock_vertex_client):
        """Should use DEFAULT_RETRY when retry=None."""
        client, mock_prediction_client = mock_vertex_client

        mock_response = httpbody_pb2.HttpBody(
            content_type="application/json",
            data=b"{}",
        )
        mock_prediction_client.raw_predict.return_value = mock_response

        client.generate({}, model_id="test-model", publisher="test-publisher", retry=None)

        call_args = mock_prediction_client.raw_predict.call_args
        assert call_args.kwargs.get("retry") == DEFAULT_RETRY

    def test_propagates_google_api_error(self, mock_vertex_client):
        """Should propagate Google API errors."""
        client, mock_prediction_client = mock_vertex_client

        mock_prediction_client.raw_predict.side_effect = google_exceptions.InvalidArgument(
            "Bad request"
        )

        with pytest.raises(google_exceptions.InvalidArgument):
            client.generate({}, model_id="test-model", publisher="test-publisher")

    def test_propagates_service_unavailable(self, mock_vertex_client):
        """Should propagate ServiceUnavailable after retries exhausted."""
        client, mock_prediction_client = mock_vertex_client

        mock_prediction_client.raw_predict.side_effect = google_exceptions.ServiceUnavailable(
            "Service unavailable"
        )

        # Use a retry with very short timeout to speed up test
        short_retry = create_retry_with_logging(
            operation_name="Test",
            initial=0.01,
            maximum=0.02,
            timeout=0.1,
        )

        with pytest.raises(google_exceptions.ServiceUnavailable):
            client.generate(
                {}, model_id="test-model", publisher="test-publisher", retry=short_retry
            )

    def test_handles_invalid_json_response(self, mock_vertex_client):
        """Should raise JSONDecodeError for invalid JSON response."""
        client, mock_prediction_client = mock_vertex_client

        mock_response = httpbody_pb2.HttpBody(
            content_type="application/json",
            data=b"not valid json",
        )
        mock_prediction_client.raw_predict.return_value = mock_response

        with pytest.raises(json.JSONDecodeError):
            client.generate({}, model_id="test-model", publisher="test-publisher")


# =============================================================================
# Test: Error Handling
# =============================================================================


class TestErrorHandling:
    """Tests for error handling."""

    def test_logs_error_on_failure(self, mock_vertex_client, caplog):
        """Should log error when prediction fails."""
        import logging

        caplog.set_level(logging.ERROR)

        client, mock_prediction_client = mock_vertex_client
        mock_prediction_client.raw_predict.side_effect = google_exceptions.InternalServerError(
            "Internal error"
        )

        with pytest.raises(google_exceptions.InternalServerError):
            client.generate({}, model_id="test-model", publisher="test-publisher", retry=None)

        assert "generate failed" in caplog.text
        assert "InternalServerError" in caplog.text


# =============================================================================
# Test: Integration with Retry
# =============================================================================


class TestRetryIntegration:
    """Tests for retry behavior integration."""

    def test_no_retry_on_invalid_argument(self, mock_vertex_client):
        """Should not retry on InvalidArgument (400)."""
        client, mock_prediction_client = mock_vertex_client

        mock_prediction_client.raw_predict.side_effect = google_exceptions.InvalidArgument(
            "Bad request"
        )

        retry = create_retry_with_logging(
            operation_name="No Retry Test",
            initial=0.01,
            maximum=0.02,
            timeout=5.0,
        )

        with pytest.raises(google_exceptions.InvalidArgument):
            client.generate({}, model_id="test-model", publisher="test-publisher", retry=retry)

        # Should only be called once (no retry)
        assert mock_prediction_client.raw_predict.call_count == 1


# =============================================================================
# Test: Stub Mode Gate
# =============================================================================


class TestVertexAIClientStubMode:
    def setup_method(self) -> None:
        from src.core.vertex_ai_client import VertexAIClient

        VertexAIClient.reset_instance()

    def teardown_method(self) -> None:
        from src.core.vertex_ai_client import VertexAIClient

        VertexAIClient.reset_instance()

    def test_generate_in_stub_mode_returns_stub_payload(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("src.core.vertex_ai_client.vertex_config.VERTEX_AI_MODE", "stub")
        monkeypatch.setattr("src.core.vertex_ai_client.vertex_config.GCP_PROJECT", "stub-project")
        with patch("src.core.vertex_ai_client.aiplatform_v1.PredictionServiceClient") as mock_cls:
            from src.core.vertex_ai_client import VertexAIClient

            VertexAIClient.reset_instance()
            client = VertexAIClient()
            # Stub mode must NOT construct the PredictionServiceClient.
            mock_cls.assert_not_called()
            response = client.generate(
                payload={"document": {"document_url": "gs://b/f.pdf"}},
                model_id="mistral-ocr-2505",
                publisher="mistralai",
            )
            assert "pages" in response
            assert response["pages"][0]["index"] == 0

    def test_init_in_real_mode_still_constructs_prediction_client(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("src.core.vertex_ai_client.vertex_config.VERTEX_AI_MODE", "real")
        monkeypatch.setattr("src.core.vertex_ai_client.vertex_config.GCP_PROJECT", "real-project")
        with patch("src.core.vertex_ai_client.aiplatform_v1.PredictionServiceClient") as mock_cls:
            from src.core.vertex_ai_client import VertexAIClient

            VertexAIClient.reset_instance()
            _ = VertexAIClient()
            mock_cls.assert_called_once()

    def test_stub_mode_init_without_project_id(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Stub mode must not require GCP_PROJECT_ID — CI runners have no credentials."""
        monkeypatch.setattr("src.core.vertex_ai_client.vertex_config.VERTEX_AI_MODE", "stub")
        monkeypatch.setattr("src.core.vertex_ai_client.vertex_config.GCP_PROJECT", None)
        from src.core.vertex_ai_client import VertexAIClient

        VertexAIClient.reset_instance()
        # Must not raise ValueError about missing project_id.
        client = VertexAIClient()
        assert client.project_id is None
        assert client._prediction_client is None

    def test_real_mode_without_project_id_still_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Real-mode guard on missing GCP_PROJECT_ID is preserved."""
        monkeypatch.setattr("src.core.vertex_ai_client.vertex_config.VERTEX_AI_MODE", "real")
        monkeypatch.setattr("src.core.vertex_ai_client.vertex_config.GCP_PROJECT", None)
        from src.core.vertex_ai_client import VertexAIClient

        VertexAIClient.reset_instance()
        with pytest.raises(ValueError, match="project_id"):
            VertexAIClient()
