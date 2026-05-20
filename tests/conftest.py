"""Root conftest.

Automatically tags tests with markers based on their directory, so that
`pytest -m unit` / `-m integration` / `-m e2e` filter correctly without
requiring every file to carry an explicit pytestmark line.
"""

from pathlib import Path

import pytest


def pytest_collection_modifyitems(config, items):
    tests_root = Path(__file__).parent.resolve()
    for item in items:
        rel = Path(item.fspath).resolve().relative_to(tests_root)
        top = rel.parts[0] if rel.parts else ""
        if top == "unit":
            item.add_marker(pytest.mark.unit)
        elif top == "integration":
            item.add_marker(pytest.mark.integration)
        elif top == "e2e":
            item.add_marker(pytest.mark.e2e)
