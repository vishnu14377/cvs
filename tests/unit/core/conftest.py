"""
Shared fixtures for core unit tests.

The ``core`` package transitively imports ``langchain_postgres``
(via ``src.core.__init__`` → ``pgvector_store``).  Since that optional
dependency is not always installed in the test environment, we inject
a mock module into ``sys.modules`` at collection time so the import
chain succeeds.
"""

import sys
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Inject mock for langchain_postgres BEFORE any core.* imports
# ---------------------------------------------------------------------------
if "langchain_postgres" not in sys.modules:
    _mock_lp = MagicMock()
    sys.modules["langchain_postgres"] = _mock_lp
    sys.modules["langchain_postgres.vectorstores"] = _mock_lp
