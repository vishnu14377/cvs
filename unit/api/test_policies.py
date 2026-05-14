"""Tests for policy endpoints."""

from unittest.mock import MagicMock, patch

import pytest
from src.policy_vector_database.models import PolicyDocument


class TestPolicyEndpoints:
    """Tests for /api/v1/policies."""

    @pytest.mark.asyncio
    async def test_list_policies_empty(self, client, auth_headers):
        with patch("src.api.routes.policies.get_policy_repository") as mock_repo_fn:
            mock_repo_fn.return_value.list_all.return_value = []
            response = await client.get("/api/v1/policies", headers=auth_headers)
        assert response.status_code == 200
        assert response.json()["policies"] == []

    @pytest.mark.asyncio
    async def test_list_policies_with_data(self, client, auth_headers):
        doc = PolicyDocument(
            policy_id="pol_1", policy_name="CPB-123", gcs_uri="gs://b/1.pdf", page_count=10
        )
        with patch("src.api.routes.policies.get_policy_repository") as mock_repo_fn:
            mock_repo_fn.return_value.list_all.return_value = [doc]
            response = await client.get("/api/v1/policies", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert len(data["policies"]) == 1
        assert data["policies"][0]["policy_name"] == "CPB-123"

    @pytest.mark.asyncio
    async def test_create_policy_success(self, client, auth_headers):
        with patch("src.api.routes.policies.PolicyProcessor") as MockProc:
            mock_result = MagicMock()
            mock_result.policy_id = "pol_new"
            mock_result.policy_name = "New Policy"
            mock_result.page_count = 20
            mock_result.success = True
            MockProc.return_value.process.return_value = mock_result

            with patch("src.api.routes.policies.get_policy_repository"):
                response = await client.post(
                    "/api/v1/policies",
                    json={"gcs_uri": "gs://bucket/policy.pdf", "policy_name": "New Policy"},
                    headers=auth_headers,
                )

        assert response.status_code == 201
        data = response.json()
        assert data["policy_name"] == "New Policy"
        assert data["status"] == "processed"

    @pytest.mark.asyncio
    async def test_delete_policy_success(self, client, auth_headers):
        with patch("src.api.routes.policies.get_policy_repository") as mock_repo_fn:
            mock_repo_fn.return_value.get.return_value = PolicyDocument(
                policy_id="pol_del", policy_name="Del", gcs_uri="gs://b/d.pdf"
            )
            mock_repo_fn.return_value.delete.return_value = True

            with patch("src.api.routes.policies.delete_session_documents") as mock_del:
                mock_del.return_value = 50
                response = await client.delete("/api/v1/policies/pol_del", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "deleted"

    @pytest.mark.asyncio
    async def test_delete_policy_not_found(self, client, auth_headers):
        with patch("src.api.routes.policies.get_policy_repository") as mock_repo_fn:
            mock_repo_fn.return_value.get.return_value = None
            response = await client.delete("/api/v1/policies/nonexistent", headers=auth_headers)
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_update_policy_success(self, client, auth_headers):
        with patch("src.api.routes.policies.get_policy_repository") as mock_repo_fn:
            mock_repo_fn.return_value.get.return_value = PolicyDocument(
                policy_id="pol_upd", policy_name="Old Name", gcs_uri="gs://b/old.pdf"
            )

            with patch("src.api.routes.policies.delete_session_documents") as mock_del:
                mock_del.return_value = 30

                with patch("src.api.routes.policies.PolicyProcessor") as MockProc:
                    mock_result = MagicMock()
                    mock_result.policy_id = "pol_upd"
                    mock_result.policy_name = "Updated Policy"
                    mock_result.page_count = 25
                    mock_result.success = True
                    MockProc.return_value.process.return_value = mock_result

                    response = await client.put(
                        "/api/v1/policies/pol_upd",
                        json={"gcs_uri": "gs://bucket/new.pdf", "policy_name": "Updated Policy"},
                        headers=auth_headers,
                    )

        assert response.status_code == 200
        data = response.json()
        assert data["policy_name"] == "Updated Policy"
        assert data["status"] == "processed"

    @pytest.mark.asyncio
    async def test_update_policy_not_found(self, client, auth_headers):
        with patch("src.api.routes.policies.get_policy_repository") as mock_repo_fn:
            mock_repo_fn.return_value.get.return_value = None
            response = await client.put(
                "/api/v1/policies/nonexistent",
                json={"gcs_uri": "gs://b/x.pdf", "policy_name": "X"},
                headers=auth_headers,
            )
        assert response.status_code == 404
