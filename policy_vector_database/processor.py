"""Policy document processor.

Reuses the existing ADR document processing pipeline (OCR + ingestion)
but with a separate persistent collection for policy documents. Unlike
ADR docs which are session-scoped, policy docs use a fixed collection
name and are not tied to any session.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from src.adr_document_processor import AdrDocumentProcessor, AdrProcessingResult, OcrModelType
from src.core.logger import get_logger

logger = get_logger(__name__)

_DEFAULT_POLICY_COLLECTION = "policy_documents"


@dataclass
class PolicyProcessingResult:
    """Result of processing a policy document."""

    policy_id: str
    policy_name: str
    page_count: int
    success: bool
    processing_result: AdrProcessingResult


class PolicyProcessor:
    """Processes policy documents through OCR and ingestion into a persistent collection."""

    def __init__(
        self,
        collection_name: str = _DEFAULT_POLICY_COLLECTION,
        model_type: OcrModelType = "mistral",
        max_workers: int = 5,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        batch_size: int | None = None,
    ):
        self._collection_name = collection_name
        self._model_type = model_type
        self._max_workers = max_workers
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._batch_size = batch_size

    def _generate_policy_id(self) -> str:
        """Generate a unique policy ID."""
        return f"pol_{uuid.uuid4().hex[:12]}"

    def process(
        self,
        gcs_uri: str,
        policy_name: str,
        policy_id: str | None = None,
        metadata: dict | None = None,
    ) -> PolicyProcessingResult:
        """Process a policy document: OCR + embed + store in persistent collection."""
        pid = policy_id or self._generate_policy_id()

        logger.info("Processing policy '%s' (%s) from %s", policy_name, pid, gcs_uri)

        processor = AdrDocumentProcessor(
            session_id=pid,
            model_type=self._model_type,
            max_workers=self._max_workers,
            collection_name=self._collection_name,
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
            batch_size=self._batch_size,
        )

        additional_metadata = {"policy_id": pid, "policy_name": policy_name}
        if metadata:
            additional_metadata.update(metadata)

        result = processor.process(
            gcs_uri=gcs_uri,
            additional_metadata=additional_metadata,
        )

        pages = getattr(result, "total_pages_processed", 0)
        logger.info(
            "Policy '%s' processing complete: success=%s, pages=%d", pid, result.success, pages
        )

        return PolicyProcessingResult(
            policy_id=pid,
            policy_name=policy_name,
            page_count=pages,
            success=result.success,
            processing_result=result,
        )
