"""
Session Manager.

Owns the per-session lifecycle of an ADR processing run:

1. Holds the session ID and document-processing configuration.
2. Runs document processing (OCR → ingestion) via ``AdrDocumentProcessor``.
3. Lazily creates and caches a **retriever** (semantic or hybrid) for the
   session — keyed by session ID in the retriever layer.
4. Ensures the **shared singleton agent** exists (see ``agent_factory``).
   The agent is session-agnostic; the ``session_id`` in ``AgentState``
   drives tool-call scoping and checkpointer isolation.

Usage:
    from src.session_manager.core.session_manager import SessionManager
    from src.agents.graph import invoke_graph

    manager = SessionManager(gcs_uri="gs://bucket/doc.pdf")
    session_id, result = manager.initialize()

    # Retriever is built per-session on first access
    docs = manager.retriever.invoke("diagnosis")

    # Agent is the shared singleton — invoke it directly with session_id
    graph = manager.agent
    result = await invoke_graph(graph, "What is the diagnosis?", session_id)
"""

from __future__ import annotations

from langchain_core.retrievers import BaseRetriever
from src.adr_document_processor import (
    AdrDocumentProcessor,
    AdrProcessingResult,
    OcrModelType,
)
from src.adr_vector_database.retriever import (
    get_hybrid_retriever,
    get_session_retriever,
)
from src.core.logger import get_logger
from src.session_manager.core.agent_factory import get_agent
from src.session_manager.core.session_id_generator import generate_session_id

logger = get_logger(__name__)


class SessionManager:
    """
    Per-session lifecycle manager.

    Responsibilities
    ----------------
    - Generate a unique session ID.
    - Run the ADR document-processing pipeline.
    - Lazily create and cache a retriever scoped to the session.
    - Expose the shared singleton agent for querying with this session's ID.
    """

    # ------------------------------------------------------------------ #
    #  Construction
    # ------------------------------------------------------------------ #

    def __init__(
        self,
        gcs_uri: str,
        # Document processing options
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
    ):
        """
        Initialise the Session Manager.

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
            search_type: Semantic search strategy (``"similarity"``, ``"mmr"``, …).
            retriever_k: Number of documents returned by the retriever.
        """
        if not gcs_uri or not gcs_uri.strip():
            raise ValueError("gcs_uri must not be empty")

        # Document processing config
        self._gcs_uri = gcs_uri.strip()
        self._model_type = model_type
        self._size_limit_mb = size_limit_mb
        self._pages_per_chunk = pages_per_chunk
        self._max_workers = max_workers
        self._collection_name = collection_name
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._batch_size = batch_size
        self._timeout = timeout
        self._additional_metadata = additional_metadata
        self._skip_ocr = skip_ocr
        self._skip_ingestion = skip_ingestion

        # Retriever config
        self._use_hybrid = use_hybrid
        self._bm25_weight = bm25_weight
        self._semantic_weight = semantic_weight
        self._search_type = search_type
        self._retriever_k = retriever_k

        # Lazily-initialised artefacts
        self._session_id: str | None = None
        self._result: AdrProcessingResult | None = None
        self._retriever: BaseRetriever | None = None

    # ------------------------------------------------------------------ #
    #  Read-only properties
    # ------------------------------------------------------------------ #

    @property
    def session_id(self) -> str | None:
        """Session ID (available after ``initialize()`` is called)."""
        return self._session_id

    @property
    def result(self) -> AdrProcessingResult | None:
        """Processing result (available after ``initialize()`` is called)."""
        return self._result

    @property
    def gcs_uri(self) -> str:
        """The GCS URI this session was created for."""
        return self._gcs_uri

    # ------------------------------------------------------------------ #
    #  Document processing
    # ------------------------------------------------------------------ #

    def initialize(self) -> tuple[str, AdrProcessingResult]:
        """
        Generate a session ID and run the document-processing pipeline.

        Returns:
            ``(session_id, AdrProcessingResult)``
        """
        self._session_id = generate_session_id()

        logger.info("=" * 60)
        logger.info("Initialising new ADR session")
        logger.info("Session ID : %s", self._session_id)
        logger.info("GCS URI    : %s", self._gcs_uri)
        logger.info("=" * 60)

        processor = AdrDocumentProcessor(
            session_id=self._session_id,
            model_type=self._model_type,
            size_limit_mb=self._size_limit_mb,
            pages_per_chunk=self._pages_per_chunk,
            max_workers=self._max_workers,
            collection_name=self._collection_name,
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
            batch_size=self._batch_size,
        )

        self._result = processor.process(
            gcs_uri=self._gcs_uri,
            timeout=self._timeout,
            additional_metadata=self._additional_metadata,
            skip_ocr=self._skip_ocr,
            skip_ingestion=self._skip_ingestion,
        )

        logger.info(
            "Session %s initialisation complete – success=%s",
            self._session_id,
            self._result.success,
        )
        return self._session_id, self._result

    # ------------------------------------------------------------------ #
    #  Retriever (lazy, per-session)
    # ------------------------------------------------------------------ #

    @property
    def retriever(self) -> BaseRetriever:
        """
        Retriever for this session (created on first access).

        Raises:
            RuntimeError: If ``initialize()`` has not been called yet.
        """
        self._ensure_initialised()
        assert self._session_id is not None
        if self._retriever is None:
            if self._use_hybrid:
                self._retriever = get_hybrid_retriever(
                    session_id=self._session_id,
                    k=self._retriever_k,
                    bm25_weight=self._bm25_weight,
                    semantic_weight=self._semantic_weight,
                    semantic_search_type=self._search_type,
                    collection_name=self._collection_name,
                )
            else:
                self._retriever = get_session_retriever(
                    session_id=self._session_id,
                    search_type=self._search_type,
                    k=self._retriever_k,
                    collection_name=self._collection_name,
                )
        return self._retriever

    def get_retriever(self, force_refresh: bool = False) -> BaseRetriever:
        """
        Get (or refresh) the session retriever.

        Args:
            force_refresh: Discard the cached retriever and rebuild it.
        """
        if force_refresh:
            self._retriever = None
        return self.retriever

    # ------------------------------------------------------------------ #
    #  Agent (singleton, shared across sessions)
    # ------------------------------------------------------------------ #

    @property
    def agent(self):
        """
        The shared singleton LangGraph agent.

        The agent itself is session-agnostic.  Session isolation is
        achieved through:

        - ``AgentState.session_id`` — the ``inject_session_id`` node
          injects this value into tool-call arguments so the ADR search
          tool queries the correct session's documents.
        - The checkpointer's ``thread_id`` (set to ``session_id``) —
          isolates conversation history per session.

        Raises:
            RuntimeError: If ``initialize()`` has not been called yet.
        """
        self._ensure_initialised()
        return get_agent()

    # ------------------------------------------------------------------ #
    #  Internal helpers
    # ------------------------------------------------------------------ #

    def _ensure_initialised(self) -> None:
        """Raise if ``initialize()`` has not been called."""
        if self._session_id is None:
            raise RuntimeError("Session not initialised – call initialize() first.")
