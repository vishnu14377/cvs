"""
ADR Search Tool for LangChain.

Provides a LangChain-compatible tool for searching ADR documents in the vector database.
Supports both semantic search and hybrid search (BM25 + semantic).

Usage:
    from src.tools.adr_search import adr_search_tool, get_adr_search_tool

    # Use the pre-configured tool directly (semantic search)
    result = adr_search_tool.invoke({"query": "What is the diagnosis?", "session_id": "session-123"})

    # Use hybrid search (BM25 + semantic)
    hybrid_tool = get_adr_search_tool(use_hybrid=True, bm25_weight=0.5, semantic_weight=0.5)
    result = hybrid_tool.invoke({"query": "patient MRN 12345", "session_id": "session-123"})

Architecture:
    All retriever management (semantic and hybrid) is handled in retriever.py.
    This tool simply calls the retriever functions, keeping the tool layer thin and focused.
"""

from __future__ import annotations

from langchain_core.callbacks import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from langchain_core.documents import Document
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from src.adr_vector_database.retriever import (
    get_hybrid_retriever,
    get_session_retriever,
)
from src.core.config import vectorstore_config
from src.core.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# Tool Input Schema
# =============================================================================


class ADRSearchInput(BaseModel):
    """Input schema for the ADR search tool.

    ``session_id`` is **not** exposed to the LLM (excluded via
    ``json_schema_extra``).  The ``inject_session_id`` node in
    ``tool_node.py`` injects the value from the agent state before
    the tool is invoked.
    """

    query: str = Field(
        description="The search query to find relevant ADR documents. "
        "This should be a natural language question or keywords "
        "related to the patient's medical records."
    )
    session_id: str = Field(
        default="",
        description="The session ID to scope the search. "
        "Only documents belonging to this session will be searched. "
        "Automatically injected by the tool node.",
        json_schema_extra={"hidden": True},
    )

    @classmethod
    def model_json_schema(cls, *args, **kwargs):
        """Override to hide ``session_id`` from the schema shown to the LLM."""
        schema = super().model_json_schema(*args, **kwargs)
        schema.get("properties", {}).pop("session_id", None)
        if "session_id" in schema.get("required", []):
            schema["required"] = [r for r in schema["required"] if r != "session_id"]
        return schema


# =============================================================================
# ADR Search Tool
# =============================================================================


class ADRSearchTool(BaseTool):
    """
    LangChain tool for searching ADR documents in the vector database.

    Supports two search modes:
    - Semantic search: Vector similarity search (default)
    - Hybrid search: Combines BM25 (keyword) + semantic search

    All retriever management (caching, session isolation, BM25 indexing) is
    handled by the functions in retriever.py. This tool provides a clean
    interface for LangChain agents to invoke searches.

    The tool searches within a specific session's documents, ensuring
    data isolation between different sessions/patients.

    Attributes:
        name: Tool name for LangChain.
        description: Description shown to the LLM for tool selection.
        args_schema: Pydantic model for input validation.
        use_hybrid: Whether to use hybrid retrieval (BM25 + semantic).
        search_type: Search strategy for semantic retriever.
        k: Number of results to return.
        bm25_weight: Weight for BM25 in hybrid mode.
        semantic_weight: Weight for semantic in hybrid mode.
    """

    name: str = "adr_search"
    description: str = (
        "Search the ADR (Additional Document Request) vector database for relevant claim processing documents. "
        "Use this tool to find information such as patient details, encounter details, full legal charts (FLC), "
        "itemized bills (IB), discharge summaries, medical notes, and other supporting documentation required for claims. "
        "Provide a clear search query to retrieve relevant documents."
    )
    args_schema: type[BaseModel] = ADRSearchInput

    # Hybrid vs semantic configuration
    use_hybrid: bool = Field(
        default=False,
        description="If True, use hybrid retrieval (BM25 + semantic). "
        "Better for queries with specific terms like MRNs, codes, or names.",
    )
    bm25_weight: float = Field(
        default=0.5, description="Weight for BM25 (keyword) results in hybrid mode (0.0 to 1.0)."
    )
    semantic_weight: float = Field(
        default=0.5, description="Weight for semantic results in hybrid mode (0.0 to 1.0)."
    )

    # Search configuration (can be overridden at instantiation)
    search_type: str = Field(default=vectorstore_config.DEFAULT_SEARCH_TYPE)
    k: int = Field(default=vectorstore_config.DEFAULT_K)
    score_threshold: float | None = Field(default=None)
    fetch_k: int = Field(default=vectorstore_config.DEFAULT_FETCH_K)
    lambda_mult: float = Field(default=vectorstore_config.DEFAULT_LAMBDA_MULT)
    collection_name: str | None = Field(default=None)

    def _run(
        self,
        query: str,
        session_id: str,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        """
        Execute the ADR search synchronously.

        Args:
            query: The search query string.
            session_id: The session ID to scope the search.
            run_manager: Optional callback manager.

        Returns:
            Formatted string containing the search results.
        """
        search_mode = "hybrid" if self.use_hybrid else "semantic"
        logger.info(
            "ADR search tool invoked: session='%s', mode='%s', query='%s...'",
            session_id,
            search_mode,
            query[:50] if len(query) > 50 else query,
        )

        try:
            # Get retriever from retriever.py functions (handles caching internally)
            if self.use_hybrid:
                retriever = get_hybrid_retriever(
                    session_id=session_id,
                    k=self.k,
                    bm25_weight=self.bm25_weight,
                    semantic_weight=self.semantic_weight,
                    collection_name=self.collection_name,
                )
            else:
                retriever = get_session_retriever(
                    session_id=session_id,
                    search_type=self.search_type,
                    k=self.k,
                    score_threshold=self.score_threshold,
                    fetch_k=self.fetch_k,
                    lambda_mult=self.lambda_mult,
                    collection_name=self.collection_name,
                )

            # Execute the search
            documents: list[Document] = retriever.invoke(query)

            logger.info(
                "ADR search completed: session='%s', mode='%s', found %d documents",
                session_id,
                search_mode,
                len(documents),
            )

            # Format results for the LLM
            return self._format_results(documents, query, session_id)

        except Exception as e:
            logger.error(
                "ADR search failed: session='%s', error=%s",
                session_id,
                str(e),
                exc_info=True,
            )
            return f"Error searching ADR documents: {str(e)}"

    async def _arun(
        self,
        query: str,
        session_id: str,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> str:
        """
        Execute the ADR search asynchronously.

        The underlying PGVector store only has a sync engine, so we run the
        synchronous ``retriever.invoke`` in a thread-pool executor to avoid
        blocking the event loop while still being compatible with the async
        agent graph.

        Args:
            query: The search query string.
            session_id: The session ID to scope the search.
            run_manager: Optional async callback manager.

        Returns:
            Formatted string containing the search results.
        """
        import asyncio

        search_mode = "hybrid" if self.use_hybrid else "semantic"
        logger.info(
            "ADR async search tool invoked: session='%s', mode='%s', query='%s...'",
            session_id,
            search_mode,
            query[:50] if len(query) > 50 else query,
        )

        try:
            # Get retriever (factory calls are lightweight / synchronous)
            if self.use_hybrid:
                retriever = get_hybrid_retriever(
                    session_id=session_id,
                    k=self.k,
                    bm25_weight=self.bm25_weight,
                    semantic_weight=self.semantic_weight,
                    collection_name=self.collection_name,
                )
            else:
                retriever = get_session_retriever(
                    session_id=session_id,
                    search_type=self.search_type,
                    k=self.k,
                    score_threshold=self.score_threshold,
                    fetch_k=self.fetch_k,
                    lambda_mult=self.lambda_mult,
                    collection_name=self.collection_name,
                )

            # Run sync retriever in a thread-pool executor so we don't block
            # the event loop (the PGVector store has no async engine).
            loop = asyncio.get_running_loop()
            documents: list[Document] = await loop.run_in_executor(None, retriever.invoke, query)

            logger.info(
                "ADR async search completed: session='%s', mode='%s', found %d documents",
                session_id,
                search_mode,
                len(documents),
            )

            # Format results synchronously (pure string manipulation, no I/O)
            return self._format_results(documents, query, session_id)

        except Exception as e:
            logger.error(
                "ADR async search failed: session='%s', error=%s",
                session_id,
                str(e),
                exc_info=True,
            )
            return f"Error searching ADR documents: {str(e)}"

    def _format_results(
        self,
        documents: list[Document],
        query: str,
        session_id: str,
    ) -> str:
        """
        Format search results into a readable string for the LLM.

        Args:
            documents: List of retrieved documents.
            query: The original search query.
            session_id: The session ID.

        Returns:
            Formatted string with search results.
        """
        if not documents:
            return f"No relevant documents found for query '{query}' "

        # Build formatted output
        parts = [
            f"Found {len(documents)} relevant document(s):",
            "",
        ]

        for i, doc in enumerate(documents, 1):
            # Extract metadata
            metadata = doc.metadata or {}
            source = metadata.get("source", "Unknown source")
            page = metadata.get("page", "N/A")

            # Format each result
            parts.append(f"--- Document {i} ---")
            parts.append(f"Source: {source}")
            if page != "N/A":
                parts.append(f"Page: {page}")
            parts.append(f"Content:\n{doc.page_content}")
            parts.append("")

        return "\n".join(parts)


# =============================================================================
# Factory Function and Pre-configured Instances
# =============================================================================


def get_adr_search_tool(
    use_hybrid: bool = False,
    search_type: str | None = None,
    k: int | None = None,
    score_threshold: float | None = None,
    fetch_k: int | None = None,
    lambda_mult: float | None = None,
    bm25_weight: float = 0.5,
    semantic_weight: float = 0.5,
    collection_name: str | None = None,
) -> ADRSearchTool:
    """
    Create a configured ADRSearchTool instance.

    Args:
        use_hybrid: If True, use hybrid retrieval (BM25 + semantic).
                    Better for queries with specific terms like MRNs, codes, names.
        search_type: Search strategy for semantic - "similarity", "mmr", or "similarity_score_threshold".
        k: Number of documents to return.
        score_threshold: Minimum score for similarity_score_threshold search.
        fetch_k: Number of candidates for MMR reranking.
        lambda_mult: MMR diversity factor (0=diverse, 1=relevant).
        bm25_weight: Weight for BM25 results in hybrid mode (0.0 to 1.0).
        semantic_weight: Weight for semantic results in hybrid mode (0.0 to 1.0).
        collection_name: Optional PGVector collection name.

    Returns:
        Configured ADRSearchTool instance.

    Examples:
        >>> # Semantic search (default)
        >>> tool = get_adr_search_tool(search_type="mmr", k=5)
        >>> result = tool.invoke({"query": "diagnosis", "session_id": "123"})

        >>> # Hybrid search (BM25 + semantic)
        >>> hybrid_tool = get_adr_search_tool(use_hybrid=True)
        >>> result = hybrid_tool.invoke({"query": "MRN 12345", "session_id": "123"})

        >>> # Hybrid with custom weights (emphasize keywords)
        >>> tool = get_adr_search_tool(
        ...     use_hybrid=True,
        ...     bm25_weight=0.7,
        ...     semantic_weight=0.3,
        ... )
    """
    return ADRSearchTool(
        use_hybrid=use_hybrid,
        search_type=search_type or vectorstore_config.DEFAULT_SEARCH_TYPE,
        k=k or vectorstore_config.DEFAULT_K,
        score_threshold=score_threshold,
        fetch_k=fetch_k or vectorstore_config.DEFAULT_FETCH_K,
        lambda_mult=lambda_mult or vectorstore_config.DEFAULT_LAMBDA_MULT,
        bm25_weight=bm25_weight,
        semantic_weight=semantic_weight,
        collection_name=collection_name,
    )


# Pre-configured tool instances
adr_search_tool = get_adr_search_tool()
adr_hybrid_search_tool = get_adr_search_tool(use_hybrid=True)


__all__ = [
    "ADRSearchTool",
    "ADRSearchInput",
    "get_adr_search_tool",
    "adr_search_tool",
    "adr_hybrid_search_tool",
]
