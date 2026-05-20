"""
Shared fixtures for session_manager unit tests.

The session_manager package transitively imports ``langchain_postgres``
(via ``src.core.__init__``).  Since that optional dependency is not
always installed in the test environment, we inject a mock module into
``sys.modules`` at collection time so the import chain succeeds.
"""

import sys
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Inject mock for langchain_postgres BEFORE any src.* imports
# ---------------------------------------------------------------------------
if "langchain_postgres" not in sys.modules:
    _mock_lp = MagicMock()
    sys.modules["langchain_postgres"] = _mock_lp
    # PGVector is accessed as langchain_postgres.PGVector
    sys.modules["langchain_postgres.vectorstores"] = _mock_lp


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_processing_result():
    """Create a mock AdrProcessingResult for testing."""
    result = MagicMock()
    result.success = True
    result.ocr_success = True
    result.ocr_total_pages = 10
    result.ocr_total_sub_files = 2
    result.ocr_successful_sub_files = 2
    result.ocr_failed_sub_files = 0
    result.ocr_extracted_text_uris = ["gs://bucket/text1.txt"]
    result.ingestion_success = True
    result.ingestion_total_documents = 5
    result.ingestion_successful_documents = 5
    result.ingestion_failed_documents = 0
    result.ingestion_total_chunks = 20
    result.error = None
    return result


@pytest.fixture
def mock_retriever():
    """Create a mock retriever."""
    return MagicMock()


@pytest.fixture
def mock_agent_graph():
    """Create a mock compiled LangGraph agent."""
    return MagicMock(name="mock_compiled_graph")
