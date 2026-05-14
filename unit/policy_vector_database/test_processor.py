"""Tests for policy document processor."""

from unittest.mock import MagicMock, patch

import pytest
from src.policy_vector_database.processor import PolicyProcessor


class TestPolicyProcessor:
    """Tests for PolicyProcessor."""

    @pytest.fixture
    def processor(self):
        return PolicyProcessor(
            collection_name="policy_documents",
        )

    def test_generate_policy_id(self, processor):
        pid = processor._generate_policy_id()
        assert pid.startswith("pol_")
        assert len(pid) > 4

    @patch("src.policy_vector_database.processor.AdrDocumentProcessor")
    def test_process_calls_underlying_processor(self, MockProcessor, processor):
        mock_instance = MockProcessor.return_value
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.total_pages_processed = 15
        mock_instance.process.return_value = mock_result

        result = processor.process(
            gcs_uri="gs://bucket/policy.pdf",
            policy_name="Test Policy",
        )

        assert result.policy_id.startswith("pol_")
        assert result.policy_name == "Test Policy"
        assert result.page_count == 15
        mock_instance.process.assert_called_once()
