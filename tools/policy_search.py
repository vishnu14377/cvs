"""Policy Search Tool for LangChain.

Searches the persistent policy document collection (Clinical Policy Bulletins).
Uses the same retriever infrastructure as ADR search but queries the policy collection.
"""

from __future__ import annotations

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.documents import Document
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from src.adr_vector_database.retriever import get_vector_store_singleton
from src.core.logger import get_logger

logger = get_logger(__name__)

_POLICY_COLLECTION = "policy_documents"


class PolicySearchInput(BaseModel):
    """Input schema for the policy search tool."""

    query: str = Field(description="Search query to find relevant clinical policy bulletins.")
    session_id: str = Field(
        default="",
        json_schema_extra={"hidden": True},
    )

    @classmethod
    def model_json_schema(cls, *args, **kwargs):
        schema = super().model_json_schema(*args, **kwargs)
        schema.get("properties", {}).pop("session_id", None)
        if "session_id" in schema.get("required", []):
            schema["required"] = [r for r in schema["required"] if r != "session_id"]
        return schema


class PolicySearchTool(BaseTool):
    """Search clinical policy bulletins in the persistent policy collection."""

    name: str = "policy_search"
    description: str = (
        "Search the clinical policy bulletins database. Use this tool to find "
        "policy guidelines, coverage criteria, and clinical protocols relevant "
        "to the patient's case."
    )
    args_schema: type[BaseModel] = PolicySearchInput
    collection_name: str = Field(default=_POLICY_COLLECTION)
    k: int = Field(default=4)

    def _run(
        self,
        query: str,
        session_id: str = "",
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        """Search policy documents."""
        logger.info("Policy search: query='%s...'", query[:50])

        try:
            vector_store = get_vector_store_singleton().get_vector_store(
                collection_name=self.collection_name,
            )
            retriever = vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": self.k},
            )
            documents: list[Document] = retriever.invoke(query)

            if not documents:
                return f"No relevant policy documents found for query '{query}'."

            parts = [f"Found {len(documents)} relevant policy document(s):", ""]
            for i, doc in enumerate(documents, 1):
                meta = doc.metadata or {}
                name = meta.get("policy_name", meta.get("source", "Unknown"))
                page = meta.get("page", "N/A")
                parts.append(f"--- Policy {i} ---")
                parts.append(f"Policy: {name}")
                if page != "N/A":
                    parts.append(f"Page: {page}")
                parts.append(f"Content:\n{doc.page_content}")
                parts.append("")

            return "\n".join(parts)

        except Exception as e:
            logger.error("Policy search failed: %s", str(e), exc_info=True)
            return f"Error searching policy documents: {str(e)}"
