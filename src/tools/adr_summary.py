"""ADR Summary Tool for LangChain.

Retrieves all ADR document chunks for a session, groups them by source and page,
and produces per-page clinical summaries using Gemini.
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

logger = get_logger(__name__)

_SUMMARIZATION_PROMPT = (
    "You are a clinical document summarizer. Given the following extracted text "
    "from page {page} of document '{source}', provide a concise clinical summary "
    "highlighting key diagnoses, procedures, findings, and dates."
    "\n\n---\n\n{text}"
)


class AdrSummaryInput(BaseModel):
    """Input schema for the ADR summary tool."""

    session_id: str = Field(
        default="",
        description="The session ID to summarize documents for. Automatically injected.",
        json_schema_extra={"hidden": True},
    )

    @classmethod
    def model_json_schema(cls, *args, **kwargs):
        schema = super().model_json_schema(*args, **kwargs)
        schema.get("properties", {}).pop("session_id", None)
        if "session_id" in schema.get("required", []):
            schema["required"] = [r for r in schema["required"] if r != "session_id"]
        return schema


class AdrSummaryTool(BaseTool):
    """Summarize all ADR documents for the current session."""

    name: str = "adr_summary"
    description: str = (
        "Summarize all ADR documents for the current session. Returns per-page "
        "clinical summaries. Use when the user asks for an overview or summary "
        "of the entire document."
    )
    args_schema: type[BaseModel] = AdrSummaryInput
    collection_name: str | None = Field(default=None)

    def _run(
        self,
        session_id: str = "",
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        logger.info("ADR summary tool invoked: session='%s'", session_id)

        try:
            documents = get_session_documents(
                session_id=session_id,
                collection_name=self.collection_name,
            )

            if not documents:
                return "No ADR documents found for this session."

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
                "ADR summary completed: session='%s', pages_summarized=%d",
                session_id,
                len(groups),
            )
            return "\n".join(parts)

        except Exception as e:
            logger.error(
                "ADR summary failed: session='%s', error=%s", session_id, str(e), exc_info=True
            )
            return f"Error summarizing ADR documents: {str(e)}"

    async def _arun(
        self,
        session_id: str = "",
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> str:
        import asyncio

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._run, session_id)
