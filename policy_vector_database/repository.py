"""Policy document CRUD operations.

Stores policy metadata in-memory for now. Can be backed by Postgres
metadata table in a future iteration.
"""

from __future__ import annotations

import threading

from src.core.logger import get_logger
from src.policy_vector_database.models import PolicyDocument

logger = get_logger(__name__)


class PolicyRepository:
    """Repository for policy document metadata."""

    def __init__(self):
        self._policies: dict[str, PolicyDocument] = {}
        self._lock = threading.Lock()

    def add(self, doc: PolicyDocument) -> None:
        """Add or update a policy document."""
        with self._lock:
            self._policies[doc.policy_id] = doc
        logger.info("Policy added: %s (%s)", doc.policy_id, doc.policy_name)

    def get(self, policy_id: str) -> PolicyDocument | None:
        """Get a policy document by ID."""
        return self._policies.get(policy_id)

    def list_all(self) -> list[PolicyDocument]:
        """List all policy documents."""
        return list(self._policies.values())

    def delete(self, policy_id: str) -> bool:
        """Delete a policy document. Returns True if found and deleted."""
        with self._lock:
            if policy_id in self._policies:
                del self._policies[policy_id]
                logger.info("Policy deleted: %s", policy_id)
                return True
            return False
