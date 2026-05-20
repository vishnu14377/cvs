"""Policy List Tool for LangChain.

Lists all available policy documents in the system.
"""

from __future__ import annotations

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel
from src.core.logger import get_logger

logger = get_logger(__name__)

_policy_repository = None


def set_policy_repository(repo) -> None:
    """Set the global policy repository instance."""
    global _policy_repository
    _policy_repository = repo


def get_policy_repository():
    """Get the global policy repository."""
    if _policy_repository is None:
        from src.policy_vector_database.repository import PolicyRepository

        set_policy_repository(PolicyRepository())
    return _policy_repository


class PolicyListInput(BaseModel):
    """Input schema for the policy list tool (no inputs required)."""

    pass


class PolicyListTool(BaseTool):
    """List all available clinical policy bulletins."""

    name: str = "policy_list"
    description: str = (
        "List all available clinical policy bulletins. Use this to see "
        "what policies are loaded in the system before searching."
    )
    args_schema: type[BaseModel] = PolicyListInput

    def _run(
        self,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        """List all policy documents."""
        logger.info("Listing policy documents")
        repo = get_policy_repository()
        policies = repo.list_all()

        if not policies:
            return "No policy documents are currently loaded in the system."

        parts = [f"Found {len(policies)} policy document(s):", ""]
        for p in policies:
            cat = f" [{p.category}]" if p.category else ""
            parts.append(f"- {p.policy_name}{cat} (ID: {p.policy_id})")

        return "\n".join(parts)
