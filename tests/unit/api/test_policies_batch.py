"""Tests for POST /api/v1/policies/batch."""

from unittest.mock import MagicMock, patch

import pytest

BATCH_URL = "/api/v1/policies/batch"


class TestBatchCreatePolicies:
    """Tests for POST /api/v1/policies/batch."""

    @pytest.mark.asyncio
    async def test_batch_all_succeed(self, client, auth_headers):
        documents = [
            {"gcs_uri": f"gs://bucket/policy_{i}.pdf", "policy_name": f"Policy {i}"}
            for i in range(3)
        ]

        with patch("src.api.routes.policies.PolicyProcessor") as MockProc:
            mock_results = []
            for i in range(3):
                mr = MagicMock()
                mr.policy_id = f"pol_{i}"
                mr.policy_name = f"Policy {i}"
                mr.page_count = 10 + i
                mr.success = True
                mock_results.append(mr)
            MockProc.return_value.process.side_effect = mock_results

            with patch("src.api.routes.policies.get_policy_repository"):
                response = await client.post(
                    BATCH_URL,
                    json={"documents": documents},
                    headers=auth_headers,
                )

        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) == 3
        for i, result in enumerate(data["results"]):
            assert result["status"] == "success"
            assert result["policy_id"] == f"pol_{i}"
            assert result["title"] == f"Policy {i}"
            assert result["error"] is None
        assert data["summary"]["total"] == 3
        assert data["summary"]["succeeded"] == 3
        assert data["summary"]["failed"] == 0

    @pytest.mark.asyncio
    async def test_batch_partial_failure(self, client, auth_headers):
        documents = [
            {"gcs_uri": f"gs://bucket/policy_{i}.pdf", "policy_name": f"Policy {i}"}
            for i in range(3)
        ]

        with patch("src.api.routes.policies.PolicyProcessor") as MockProc:
            success_0 = MagicMock()
            success_0.policy_id = "pol_0"
            success_0.policy_name = "Policy 0"
            success_0.page_count = 10
            success_0.success = True

            success_2 = MagicMock()
            success_2.policy_id = "pol_2"
            success_2.policy_name = "Policy 2"
            success_2.page_count = 12
            success_2.success = True

            MockProc.return_value.process.side_effect = [
                success_0,
                RuntimeError("OCR engine exploded"),
                success_2,
            ]

            with patch("src.api.routes.policies.get_policy_repository"):
                response = await client.post(
                    BATCH_URL,
                    json={"documents": documents},
                    headers=auth_headers,
                )

        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) == 3
        assert data["results"][0]["status"] == "success"
        assert data["results"][1]["status"] == "failed"
        # Error messages must be sanitized — never leak raw exception text
        # (matches sibling create_policy endpoint and repo-wide policy)
        assert "OCR engine exploded" not in data["results"][1]["error"]
        assert "Policy processing failed" in data["results"][1]["error"]
        assert data["results"][2]["status"] == "success"
        assert data["summary"]["succeeded"] == 2
        assert data["summary"]["failed"] == 1

    @pytest.mark.asyncio
    async def test_batch_empty_list(self, client, auth_headers):
        response = await client.post(
            BATCH_URL,
            json={"documents": []},
            headers=auth_headers,
        )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_batch_exceeds_max(self, client, auth_headers):
        documents = [
            {"gcs_uri": f"gs://bucket/policy_{i}.pdf", "policy_name": f"Policy {i}"}
            for i in range(21)
        ]
        response = await client.post(
            BATCH_URL,
            json={"documents": documents},
            headers=auth_headers,
        )
        assert response.status_code == 422
