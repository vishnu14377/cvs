"""
Tools package for the CareConnect ADR AI Agent.

Available Tools:
- adr_search_tool: Semantic search for ADR documents (default)
- adr_hybrid_search_tool: Hybrid search (BM25 + semantic) for keyword matching

Usage:
    from src.tools import adr_search_tool, adr_hybrid_search_tool, get_adr_search_tool

    # Semantic search (default)
    result = adr_search_tool.invoke({"query": "diagnosis", "session_id": "123"})

    # Hybrid search (better for MRNs, codes, names)
    result = adr_hybrid_search_tool.invoke({"query": "MRN 12345", "session_id": "123"})

    # Custom configured tool
    tool = get_adr_search_tool(use_hybrid=True, bm25_weight=0.7, semantic_weight=0.3)
"""

from src.tools.adr_search import (
    ADRSearchInput,
    ADRSearchTool,
    adr_hybrid_search_tool,
    adr_search_tool,
    get_adr_search_tool,
)

__all__ = [
    "ADRSearchTool",
    "ADRSearchInput",
    "get_adr_search_tool",
    "adr_search_tool",
    "adr_hybrid_search_tool",
]
