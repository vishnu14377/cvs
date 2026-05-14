"""Policy Summary Tool for LangChain.

Retrieves all policy document chunks, groups by source and page,
and produces per-page summaries of clinical policy bulletins using Gemini.
"""

from __future__ import annotations

from collections import defaultdict

from langchain_core.callbacks import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from src.adr_vector_database.retriever import get_session_documents
from src.core.langchain_client import LangChainClient
from src.core.logger import get_logger
from src.tools.policy_list import get_policy_repository

logger = get_logger(__name__)

_POLICY_COLLECTION = "policy_documents"

_SUMMARIZATION_PROMPT = (
    "You are a clinical policy document summarizer. Given the following extracted "
    "text from page {page} of policy document '{source}', provide a concise summary "
    "highlighting coverage criteria, medical necessity requirements, exclusions, "
    "and key policy guidelines."
    "\n\n---\n\n{text}"
)


class PolicySummaryInput(BaseModel):
    """Input schema for the policy summary tool."""

    session_id: str = Field(
        default="",
        description="The session ID. Automatically injected.",
        json_schema_extra={"hidden": True},
    )

    @classmethod
    def model_json_schema(cls, *args, **kwargs):
        schema = super().model_json_schema(*args, **kwargs)
        schema.get("properties", {}).pop("session_id", None)
        if "session_id" in schema.get("required", []):
            schema["required"] = [r for r in schema["required"] if r != "session_id"]
        return schema


class PolicySummaryTool(BaseTool):
    """Summarize policy documents."""

    name: str = "policy_summary"
    description: str = (
        "Summarize policy documents. Returns per-page summaries of clinical "
        "policy bulletins. Use when the user asks for an overview of available policies."
    )
    args_schema: type[BaseModel] = PolicySummaryInput
    collection_name: str = Field(default=_POLICY_COLLECTION)

    def _run(
        self,
        session_id: str = "",
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        logger.info("Policy summary tool invoked: session='%s'", session_id)

        try:
            # Policies are ingested with session_id=<policy_id> (see
            # PolicyProcessor), so we can't filter by the chat session_id.
            # Fetch documents per policy_id from the policy repository.
            repo = get_policy_repository()
            policies = repo.list_all()
            documents = []
            for policy in policies:
                documents.extend(
                    get_session_documents(
                        session_id=policy.policy_id,
                        collection_name=self.collection_name,
                    )
                )

            if not documents:
                return "No policy documents found."

            groups: dict[tuple[str, int], list[str]] = defaultdict(list)
            for doc in documents:
                source = doc.metadata.get("source", "Unknown")
                page = doc.metadata.get("page", 0)
                groups[(source, page)].append(doc.page_content)

            llm = LangChainClient().client
            parts: list[str] = []

            for (source, page), chunks in sorted(groups.items(), key=lambda x: (x[0][0], x[0][1])):
                combined_text = "\n\n".join(chunks)
                prompt = _SUMMARIZATION_PROMPT.format(page=page, source=source, text=combined_text)
                response = llm.invoke([HumanMessage(content=prompt)])
                parts.append(f"--- {source} | Page {page} ---")
                parts.append(str(response.content))
                parts.append("")

            logger.info(
                "Policy summary completed: session='%s', pages_summarized=%d",
                session_id,
                len(groups),
            )
            return "\n".join(parts)

        except Exception as e:
            logger.error(
                "Policy summary failed: session='%s', error=%s", session_id, str(e), exc_info=True
            )
            return f"Error summarizing policy documents: {str(e)}"

    async def _arun(
        self,
        session_id: str = "",
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> str:
        import asyncio

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._run, session_id)
