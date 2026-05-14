"""Tests for policy list tool."""

from unittest.mock import patch

from src.policy_vector_database.models import PolicyDocument
from src.tools.policy_list import PolicyListTool


class TestPolicyListTool:
    """Tests for PolicyListTool."""

    def test_tool_name(self):
        tool = PolicyListTool()
        assert tool.name == "policy_list"

    @patch("src.tools.policy_list.get_policy_repository")
    def test_list_returns_policies(self, mock_get_repo):
        mock_repo = mock_get_repo.return_value
        mock_repo.list_all.return_value = [
            PolicyDocument(policy_id="pol_1", policy_name="Cardiology CPB", gcs_uri="gs://b/1.pdf"),
            PolicyDocument(policy_id="pol_2", policy_name="Oncology CPB", gcs_uri="gs://b/2.pdf"),
        ]

        tool = PolicyListTool()
        result = tool._run()
        assert "Cardiology CPB" in result
        assert "Oncology CPB" in result
        assert "2 policy" in result.lower()

    @patch("src.tools.policy_list.get_policy_repository")
    def test_list_empty(self, mock_get_repo):
        mock_repo = mock_get_repo.return_value
        mock_repo.list_all.return_value = []

        tool = PolicyListTool()
        result = tool._run()
        assert "no policy" in result.lower()
