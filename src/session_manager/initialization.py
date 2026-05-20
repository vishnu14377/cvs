"""
Session Initialization.

Thin entry-point that wires together a ``SessionManager``, runs the
document-processing pipeline, and returns the session artefacts.

The heavy lifting lives in:

- ``core/session_id_generator.py``  – unique ID generation
- ``core/session_manager.py``       – ``SessionManager`` class (per-session lifecycle)
- ``core/agent_factory.py``         – singleton LangGraph agent (shared across sessions)

Usage:
    from src.session_manager.initialization import initialize_session
    from src.agents.graph import invoke_graph

    session_id, result, manager = initialize_session(
        gcs_uri="gs://bucket/path/to/document.pdf"
    )
    # Retriever is per-session; agent is the shared singleton
    graph = manager.agent
    result = await invoke_graph(graph, "What is the diagnosis?", session_id)
"""

from __future__ import annotations

from src.adr_document_processor import AdrProcessingResult, OcrModelType
from src.session_manager.core.session_manager import SessionManager


def initialize_session(
    gcs_uri: str,
    *,
    model_type: OcrModelType = "mistral",
    size_limit_mb: float = 5.0,
    pages_per_chunk: int | None = None,
    max_workers: int = 5,
    collection_name: str | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    batch_size: int | None = None,
    timeout: float | None = None,
    additional_metadata: dict | None = None,
    skip_ocr: bool = False,
    skip_ingestion: bool = False,
    # Retriever options
    use_hybrid: bool = False,
    bm25_weight: float = 0.5,
    semantic_weight: float = 0.5,
    search_type: str | None = None,
    retriever_k: int | None = None,
) -> tuple[str, AdrProcessingResult, SessionManager]:
    """
    Create a session, process ADR document(s), and return the manager.

    This is the main convenience entry-point.  It:

    1. Builds a ``SessionManager`` with the supplied configuration.
    2. Calls ``manager.initialize()`` to generate a session ID and run
       the OCR → ingestion pipeline.
    3. Returns ``(session_id, result, manager)`` so the caller can
       access ``manager.retriever`` (per-session) and ``manager.agent``
       (shared singleton) on demand.

    Note:
        Agent configuration is **application-level**, not per-session.
        Call ``configure_agent()`` from ``agent_factory`` once at startup
        to customise the system prompt, tools, checkpointer, etc.

    Args:
        gcs_uri: GCS URI to the source PDF document(s).
        model_type: OCR model type (``"mistral"`` or ``"llm"``).
        size_limit_mb: Target size limit per sub-file chunk.
        pages_per_chunk: Fixed page count per chunk (overrides size splitting).
        max_workers: Parallel workers for OCR / ingestion.
        collection_name: PGVector collection name override.
        chunk_size: Max characters per text chunk.
        chunk_overlap: Overlap characters between chunks.
        batch_size: Batch size for vector store inserts.
        timeout: Per-request OCR timeout in seconds.
        additional_metadata: Extra metadata attached to every vector chunk.
        skip_ocr: Skip the OCR stage.
        skip_ingestion: Skip the ingestion stage.
        use_hybrid: Use hybrid (BM25 + semantic) retrieval.
        bm25_weight: BM25 weight in hybrid mode.
        semantic_weight: Semantic weight in hybrid mode.
        search_type: Semantic search strategy.
        retriever_k: Number of documents returned by the retriever.

    Returns:
        ``(session_id, AdrProcessingResult, SessionManager)``

    Example:
        >>> from src.agents.graph import invoke_graph
        >>> session_id, result, manager = initialize_session(
        ...     gcs_uri="gs://bucket/path/to/document.pdf"
        ... )
        >>> print(f"Session: {session_id}, Success: {result.success}")
        >>> graph = manager.agent
        >>> response = await invoke_graph(graph, "What is the diagnosis?", session_id)
    """
    manager = SessionManager(
        gcs_uri=gcs_uri,
        model_type=model_type,
        size_limit_mb=size_limit_mb,
        pages_per_chunk=pages_per_chunk,
        max_workers=max_workers,
        collection_name=collection_name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        batch_size=batch_size,
        timeout=timeout,
        additional_metadata=additional_metadata,
        skip_ocr=skip_ocr,
        skip_ingestion=skip_ingestion,
        use_hybrid=use_hybrid,
        bm25_weight=bm25_weight,
        semantic_weight=semantic_weight,
        search_type=search_type,
        retriever_k=retriever_k,
    )

    session_id, processing_result = manager.initialize()
    return session_id, processing_result, manager


if __name__ == "__main__":
    test_gcs_uri = "gs://care_connect_ai_initiatives/test_full_adrs/"
    sid, res, mgr = initialize_session(gcs_uri=test_gcs_uri)
    print(f"Session ID: {sid}")
    print(f"Processing Success: {res.success}")

    # Retriever is per-session; agent is the shared singleton:
    # retriever = mgr.retriever
    # graph     = mgr.agent
    # response  = await invoke_graph(graph, "What is the diagnosis?", sid)
