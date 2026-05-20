"""Tests for policy repository."""

import pytest
from src.policy_vector_database.models import PolicyDocument
from src.policy_vector_database.repository import PolicyRepository


class TestPolicyRepository:
    """Tests for in-memory policy repository."""

    @pytest.fixture
    def repo(self):
        repo = PolicyRepository()
        repo._policies.clear()
        return repo

    def test_add_policy(self, repo):
        doc = PolicyDocument(
            policy_id="pol_test1",
            policy_name="Test Policy",
            gcs_uri="gs://bucket/policy.pdf",
            page_count=10,
        )
        repo.add(doc)
        assert repo.get("pol_test1") is not None
        assert repo.get("pol_test1").policy_name == "Test Policy"

    def test_list_policies(self, repo):
        doc1 = PolicyDocument(policy_id="pol_1", policy_name="Policy 1", gcs_uri="gs://b/1.pdf")
        doc2 = PolicyDocument(policy_id="pol_2", policy_name="Policy 2", gcs_uri="gs://b/2.pdf")
        repo.add(doc1)
        repo.add(doc2)
        policies = repo.list_all()
        assert len(policies) == 2

    def test_get_missing_returns_none(self, repo):
        assert repo.get("nonexistent") is None

    def test_delete_policy(self, repo):
        doc = PolicyDocument(policy_id="pol_del", policy_name="Delete Me", gcs_uri="gs://b/d.pdf")
        repo.add(doc)
        assert repo.delete("pol_del") is True
        assert repo.get("pol_del") is None

    def test_delete_missing_returns_false(self, repo):
        assert repo.delete("nonexistent") is False
