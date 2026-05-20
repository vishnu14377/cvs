"""Tests for session endpoints."""

import time
from unittest.mock import MagicMock, patch

import pytest


class TestCreateSession:
    """Tests for POST /api/v1/sessions."""

    @pytest.mark.asyncio
    async def test_create_session_success(self, client, auth_headers):
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.total_pages_processed = 5
        mock_manager = MagicMock()
        mock_manager.session_id = "sess_test_123"

        with patch("src.api.routes.sessions.initialize_session") as mock_init:
            mock_init.return_value = ("sess_test_123", mock_result, mock_manager)
            response = await client.post(
                "/api/v1/sessions",
                json={"gcs_uris": ["gs://bucket/doc.pdf"]},
                headers=auth_headers,
            )

        assert response.status_code == 201
        data = response.json()
        assert data["session_id"] == "sess_test_123"
        assert data["status"] == "ready"

    @pytest.mark.asyncio
    async def test_create_session_requires_auth(self, client):
        response = await client.post(
            "/api/v1/sessions",
            json={"gcs_uris": ["gs://bucket/doc.pdf"]},
        )
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_create_session_validates_empty_uris(self, client, auth_headers):
        response = await client.post(
            "/api/v1/sessions",
            json={"gcs_uris": []},
            headers=auth_headers,
        )
        assert response.status_code == 422


class TestGetSession:
    """Tests for GET /api/v1/sessions/{sessionId}."""

    @pytest.mark.asyncio
    async def test_get_session_found(self, client, auth_headers):
        from src.api.dependencies import get_session_registry

        registry = get_session_registry()
        mock_manager = MagicMock()
        mock_manager.session_id = "sess_existing"
        mock_manager.result = MagicMock()
        mock_manager.result.total_pages_processed = 3
        registry["sess_existing"] = (mock_manager, time.time())

        try:
            response = await client.get(
                "/api/v1/sessions/sess_existing",
                headers=auth_headers,
            )
            assert response.status_code == 200
            data = response.json()
            assert data["session_id"] == "sess_existing"
        finally:
            registry.pop("sess_existing", None)

    @pytest.mark.asyncio
    async def test_get_session_not_found(self, client, auth_headers):
        response = await client.get(
            "/api/v1/sessions/nonexistent",
            headers=auth_headers,
        )
        assert response.status_code == 404


class TestDeleteSession:
    """Tests for DELETE /api/v1/sessions/{sessionId}."""

    @pytest.mark.asyncio
    async def test_delete_session_success(self, client, auth_headers):
        from src.api.dependencies import get_session_registry

        registry = get_session_registry()
        registry["sess_to_delete"] = (MagicMock(), time.time())

        with patch("src.api.routes.sessions.delete_session") as mock_del:
            mock_del.return_value = MagicMock(
                session_id="sess_to_delete",
                vectors_deleted=10,
                success=True,
                errors=[],
            )
            response = await client.delete(
                "/api/v1/sessions/sess_to_delete",
                headers=auth_headers,
            )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "deleted"
        assert "sess_to_delete" not in registry

    @pytest.mark.asyncio
    async def test_delete_session_not_found(self, client, auth_headers):
        response = await client.delete(
            "/api/v1/sessions/nonexistent",
            headers=auth_headers,
        )
        assert response.status_code == 404


class TestListSessions:
    """Tests for GET /api/v1/sessions."""

    @pytest.mark.asyncio
    async def test_list_sessions_empty(self, client, auth_headers):
        response = await client.get("/api/v1/sessions", headers=auth_headers)
        assert response.status_code == 200
        assert response.json()["sessions"] == []

    @pytest.mark.asyncio
    async def test_list_sessions_with_data(self, client, auth_headers):
        from src.api.dependencies import get_session_registry

        registry = get_session_registry()
        mock_manager = MagicMock()
        mock_manager.result = MagicMock()
        mock_manager.result.total_pages_processed = 5
        registry["sess_list_1"] = (mock_manager, time.time())
        registry["sess_list_2"] = (MagicMock(result=None), time.time())

        try:
            response = await client.get("/api/v1/sessions", headers=auth_headers)
            assert response.status_code == 200
            data = response.json()
            assert len(data["sessions"]) >= 2
            ids = [s["session_id"] for s in data["sessions"]]
            assert "sess_list_1" in ids
            assert "sess_list_2" in ids
        finally:
            registry.pop("sess_list_1", None)
            registry.pop("sess_list_2", None)


class TestCreateSessionUpload:
    """Tests for POST /api/v1/sessions/upload."""

    @pytest.mark.asyncio
    async def test_upload_success(self, client, auth_headers):
        import io

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.total_pages_processed = 3
        mock_manager = MagicMock()

        with patch("src.api.routes.sessions.upload_to_gcs") as mock_upload:
            mock_upload.return_value = "gs://bucket/uploads/abc/test.pdf"
            with patch("src.api.routes.sessions.initialize_session") as mock_init:
                mock_init.return_value = ("sess_upload_1", mock_result, mock_manager)
                response = await client.post(
                    "/api/v1/sessions/upload",
                    files={
                        "files": (
                            "test.pdf",
                            io.BytesIO(b"%PDF-1.4 fake content"),
                            "application/pdf",
                        )
                    },
                    data={"ocr_engine": "mistral"},
                    headers={"Authorization": "Bearer test-token-secret"},
                )

        assert response.status_code == 201
        data = response.json()
        assert data["session_id"] == "sess_upload_1"
        assert data["status"] == "ready"

    @pytest.mark.asyncio
    async def test_upload_rejects_non_pdf(self, client, auth_headers):
        import io

        response = await client.post(
            "/api/v1/sessions/upload",
            files={"files": ("test.txt", io.BytesIO(b"not a pdf"), "text/plain")},
            data={"ocr_engine": "mistral"},
            headers={"Authorization": "Bearer test-token-secret"},
        )
        assert response.status_code == 400
        assert "PDF" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_upload_requires_auth(self, client):
        import io

        response = await client.post(
            "/api/v1/sessions/upload",
            files={"files": ("test.pdf", io.BytesIO(b"%PDF-1.4"), "application/pdf")},
        )
        assert response.status_code == 401
