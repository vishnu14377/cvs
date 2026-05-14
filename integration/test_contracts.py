"""Contract tests — validate API responses match Pydantic schemas.

Each test hits the real API and parses the response through the actual Pydantic
model.  If Pydantic raises ValidationError the contract is broken — a field was
renamed, its type changed, or it was dropped without updating the schema.

Prerequisites:
    docker-compose up --build

Run:
    .venv/bin/python -m pytest tests/integration/test_contracts.py -v --timeout=180
"""

from __future__ import annotations

import pytest

from tests.integration.conftest import AUTH_TOKEN, SAMPLE_ADR_URI, skip_if_no_api

# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


@skip_if_no_api
@pytest.mark.integration
class TestHealthContracts:
    def test_health_schema(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        assert isinstance(data["status"], str)

    def test_health_ready_schema(self, client):
        resp = client.get("/health/ready")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        assert "dependencies" in data
        assert isinstance(data["dependencies"], dict)


# ---------------------------------------------------------------------------
# Sessions, Query, History, Feedback  (ordered — later tests reuse session_id)
# ---------------------------------------------------------------------------


@skip_if_no_api
@pytest.mark.integration
class TestSessionContracts:
    """Full session lifecycle.  Tests run in definition order to share state."""

    session_id: str = ""
    message_id: str = ""

    def test_create_session_schema(self, client, auth_headers):
        from src.api.models.sessions import SessionResponse

        resp = client.post(
            "/api/v1/sessions",
            json={"gcs_uris": [SAMPLE_ADR_URI]},
            headers=auth_headers,
        )
        assert resp.status_code == 201, f"Expected 201, got {resp.status_code}: {resp.text}"
        model = SessionResponse(**resp.json())
        assert model.session_id, "session_id must be non-empty"
        assert model.status in ("ready", "error"), f"Unexpected status value: {model.status!r}"
        TestSessionContracts.session_id = model.session_id

    def test_get_session_schema(self, client, auth_headers):
        from src.api.models.sessions import SessionResponse

        sid = TestSessionContracts.session_id
        if not sid:
            pytest.skip("No session created — test_create_session_schema must pass first")
        resp = client.get(f"/api/v1/sessions/{sid}", headers=auth_headers)
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
        model = SessionResponse(**resp.json())
        assert model.session_id == sid

    def test_list_sessions_schema(self, client, auth_headers):
        from src.api.models.sessions import SessionListResponse

        resp = client.get("/api/v1/sessions", headers=auth_headers)
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
        model = SessionListResponse(**resp.json())
        assert isinstance(model.sessions, list)

    def test_query_schema(self, client, auth_headers):
        from src.api.models.query import QueryResponse

        sid = TestSessionContracts.session_id
        if not sid:
            pytest.skip("No session created — test_create_session_schema must pass first")
        resp = client.post(
            f"/api/v1/sessions/{sid}/query",
            json={"message": "What is the patient's name?"},
            headers=auth_headers,
        )
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
        model = QueryResponse(**resp.json())
        assert model.session_id == sid
        assert model.message.role == "assistant"
        assert model.message.content, "message.content must be non-empty"
        assert model.message.content_html, "message.content_html must be non-empty"
        assert model.message.content_base64, "message.content_base64 must be non-empty"
        assert model.message_id.startswith("msg_"), (
            f"message_id should start with 'msg_', got: {model.message_id!r}"
        )
        TestSessionContracts.message_id = model.message_id

    def test_history_schema(self, client, auth_headers):
        from src.api.models.query import HistoryResponse

        sid = TestSessionContracts.session_id
        if not sid:
            pytest.skip("No session created — test_create_session_schema must pass first")
        resp = client.get(f"/api/v1/sessions/{sid}/history", headers=auth_headers)
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
        model = HistoryResponse(**resp.json())
        assert model.session_id == sid
        assert len(model.messages) >= 2, (
            f"Expected at least 2 messages in history, got {len(model.messages)}"
        )

    def test_feedback_submit_schema(self, client, auth_headers):
        from src.api.models.feedback import FeedbackResponse

        sid = TestSessionContracts.session_id
        mid = TestSessionContracts.message_id
        if not sid or not mid:
            pytest.skip("No session/message — earlier tests must pass first")
        resp = client.post(
            f"/api/v1/sessions/{sid}/feedback",
            json={"message_id": mid, "rating": "positive", "comment": "Contract test"},
            headers=auth_headers,
        )
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
        model = FeedbackResponse(**resp.json())
        assert model.feedback_id.startswith("fb_"), (
            f"feedback_id should start with 'fb_', got: {model.feedback_id!r}"
        )
        assert model.status == "stored"

    def test_feedback_get_schema(self, client, auth_headers):
        from src.api.models.feedback import FeedbackListResponse

        sid = TestSessionContracts.session_id
        if not sid:
            pytest.skip("No session — test_create_session_schema must pass first")
        resp = client.get(f"/api/v1/sessions/{sid}/feedback", headers=auth_headers)
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
        model = FeedbackListResponse(**resp.json())
        assert model.session_id == sid
        assert len(model.feedback) >= 1, (
            "Expected at least 1 feedback entry after test_feedback_submit_schema"
        )

    def test_upload_session_schema(self, client, auth_headers):
        import os

        from src.api.models.sessions import SessionResponse

        pdf_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "data",
            "sample_adrs",
            "BINGHAM,CALLIE_91202308017_FLC5_REDACTED.pdf",
        )
        if not os.path.exists(pdf_path):
            pytest.skip("Sample PDF not found")
        with open(pdf_path, "rb") as f:
            resp = client.post(
                "/api/v1/sessions/upload",
                files={"files": ("test_upload.pdf", f, "application/pdf")},
                data={"ocr_engine": "mistral"},
                headers={"Authorization": f"Bearer {AUTH_TOKEN}"},
                timeout=120,
            )
        assert resp.status_code == 201, f"Expected 201, got {resp.status_code}: {resp.text}"
        model = SessionResponse(**resp.json())
        assert model.session_id
        assert model.status in ("ready", "error")
        # Clean up
        if model.session_id:
            client.delete(f"/api/v1/sessions/{model.session_id}", headers=auth_headers)

    def test_delete_session_schema(self, client, auth_headers):
        from src.api.models.sessions import SessionDeleteResponse

        sid = TestSessionContracts.session_id
        if not sid:
            pytest.skip("No session — test_create_session_schema must pass first")
        resp = client.delete(f"/api/v1/sessions/{sid}", headers=auth_headers)
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
        model = SessionDeleteResponse(**resp.json())
        assert model.session_id == sid
        assert model.status == "deleted"


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------


@skip_if_no_api
@pytest.mark.integration
class TestPolicyContracts:
    def test_list_policies_schema(self, client, auth_headers):
        from src.api.models.policies import PolicyListResponse

        resp = client.get("/api/v1/policies", headers=auth_headers)
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
        model = PolicyListResponse(**resp.json())
        assert isinstance(model.policies, list)


# ---------------------------------------------------------------------------
# Error response contracts
# ---------------------------------------------------------------------------


@skip_if_no_api
@pytest.mark.integration
class TestErrorContracts:
    def test_401_schema(self, client):
        """Unauthenticated requests must return 401 with a 'detail' field."""
        resp = client.post("/api/v1/sessions", json={"gcs_uris": ["gs://x/y.pdf"]})
        assert resp.status_code == 401, f"Expected 401, got {resp.status_code}: {resp.text}"
        data = resp.json()
        assert "detail" in data, f"401 response must include 'detail': {data}"

    def test_404_schema(self, client, auth_headers):
        """Non-existent session must return 404 with a 'detail' field."""
        resp = client.get("/api/v1/sessions/nonexistent", headers=auth_headers)
        assert resp.status_code == 404, f"Expected 404, got {resp.status_code}: {resp.text}"
        data = resp.json()
        assert "detail" in data, f"404 response must include 'detail': {data}"

    def test_422_schema(self, client, auth_headers):
        """Empty gcs_uris array must return 422 with a 'detail' field."""
        resp = client.post(
            "/api/v1/sessions",
            json={"gcs_uris": []},
            headers=auth_headers,
        )
        assert resp.status_code == 422, f"Expected 422, got {resp.status_code}: {resp.text}"
        data = resp.json()
        assert "detail" in data, f"422 response must include 'detail': {data}"
