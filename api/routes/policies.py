"""Policy document CRUD endpoints."""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends, HTTPException, status
from src.adr_document_processor import OcrModelType
from src.adr_vector_database.vector_store import delete_session_documents
from src.api.dependencies import verify_token
from src.api.models.policies import (
    BatchCreatePolicyRequest,
    BatchPolicyResponse,
    BatchPolicyResult,
    BatchSummary,
    CreatePolicyRequest,
    PolicyDeleteResponse,
    PolicyListResponse,
    PolicyResponse,
)
from src.core.logger import get_logger
from src.policy_vector_database.models import PolicyDocument
from src.policy_vector_database.processor import PolicyProcessor
from src.tools.policy_list import get_policy_repository

logger = get_logger(__name__)

_OCR_ENGINE_MAP: dict[str, OcrModelType] = {
    "mistral": "mistral",
    "mistral-ocr": "mistral",
    "gemini-vision": "llm",
    "document-ai": "llm",
}

router = APIRouter(
    prefix="/api/v1/policies",
    tags=["policies"],
    dependencies=[Depends(verify_token)],
)


@router.get("", response_model=PolicyListResponse)
async def list_policies():
    """List all available policy documents."""
    repo = get_policy_repository()
    policies = repo.list_all()
    return PolicyListResponse(
        policies=[
            PolicyResponse(
                policy_id=p.policy_id,
                policy_name=p.policy_name,
                status=p.status,
                page_count=p.page_count,
                category=p.category,
                created_at=p.created_at,
                metadata=p.metadata,
            )
            for p in policies
        ]
    )


@router.post("", status_code=status.HTTP_201_CREATED, response_model=PolicyResponse)
async def create_policy(body: CreatePolicyRequest):
    """Add a policy document to the persistent policy RAG."""
    model_type = _OCR_ENGINE_MAP.get(body.ocr_engine)
    if model_type is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown OCR engine: {body.ocr_engine}",
        )

    start = time.monotonic()
    processor = PolicyProcessor(model_type=model_type)
    try:
        result = processor.process(
            gcs_uri=body.gcs_uri,
            policy_name=body.policy_name,
            metadata=body.metadata,
        )
    except Exception as e:
        logger.error("Policy processing failed: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Policy processing failed. Please try again.",
        ) from None
    elapsed_ms = int((time.monotonic() - start) * 1000)

    repo = get_policy_repository()
    doc = PolicyDocument(
        policy_id=result.policy_id,
        policy_name=result.policy_name,
        gcs_uri=body.gcs_uri,
        page_count=result.page_count,
        ocr_engine=body.ocr_engine,
        category=(body.metadata or {}).get("category"),
        metadata=body.metadata or {},
        status="processed" if result.success else "error",
    )
    repo.add(doc)

    return PolicyResponse(
        policy_id=result.policy_id,
        policy_name=result.policy_name,
        status="processed" if result.success else "error",
        page_count=result.page_count,
        processing_time_ms=elapsed_ms,
        category=doc.category,
        created_at=doc.created_at,
        metadata=body.metadata,
    )


@router.post("/batch", response_model=BatchPolicyResponse)
async def batch_create_policies(body: BatchCreatePolicyRequest):
    """Process multiple policy documents with partial-success semantics."""
    repo = get_policy_repository()
    results: list[BatchPolicyResult] = []
    succeeded = 0

    for doc_req in body.documents:
        model_type = _OCR_ENGINE_MAP.get(doc_req.ocr_engine)
        if model_type is None:
            results.append(
                BatchPolicyResult(
                    title=doc_req.policy_name,
                    status="failed",
                    error=f"Unknown OCR engine: {doc_req.ocr_engine}",
                )
            )
            continue

        try:
            processor = PolicyProcessor(model_type=model_type)
            result = processor.process(
                gcs_uri=doc_req.gcs_uri,
                policy_name=doc_req.policy_name,
                metadata=doc_req.metadata,
            )
        except Exception as e:
            logger.error(
                "Batch policy processing failed for '%s': %s",
                doc_req.policy_name,
                e,
                exc_info=True,
            )
            results.append(
                BatchPolicyResult(
                    title=doc_req.policy_name,
                    status="failed",
                    error="Policy processing failed. Please try again.",
                )
            )
            continue

        doc = PolicyDocument(
            policy_id=result.policy_id,
            policy_name=result.policy_name,
            gcs_uri=doc_req.gcs_uri,
            page_count=result.page_count,
            ocr_engine=doc_req.ocr_engine,
            category=(doc_req.metadata or {}).get("category"),
            metadata=doc_req.metadata or {},
            status="processed" if result.success else "error",
        )
        repo.add(doc)

        results.append(
            BatchPolicyResult(
                policy_id=result.policy_id,
                title=doc_req.policy_name,
                status="success",
            )
        )
        succeeded += 1

    return BatchPolicyResponse(
        results=results,
        summary=BatchSummary(
            total=len(body.documents),
            succeeded=succeeded,
            failed=len(body.documents) - succeeded,
        ),
    )


@router.get("/{policy_id}", response_model=PolicyResponse)
async def get_policy(policy_id: str):
    """Get policy document metadata."""
    repo = get_policy_repository()
    doc = repo.get(policy_id)
    if doc is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Policy '{policy_id}' not found"
        )
    return PolicyResponse(
        policy_id=doc.policy_id,
        policy_name=doc.policy_name,
        status=doc.status,
        page_count=doc.page_count,
        category=doc.category,
        created_at=doc.created_at,
        metadata=doc.metadata,
    )


@router.put("/{policy_id}", response_model=PolicyResponse)
async def update_policy(policy_id: str, body: CreatePolicyRequest):
    """Re-process a policy document with new or updated content."""
    repo = get_policy_repository()
    doc = repo.get(policy_id)
    if doc is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Policy '{policy_id}' not found"
        )

    model_type = _OCR_ENGINE_MAP.get(body.ocr_engine)
    if model_type is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown OCR engine: {body.ocr_engine}",
        )

    # Delete old vectors
    try:
        delete_session_documents(session_id=policy_id, collection_name="policy_documents")
    except Exception as e:
        logger.error("Failed to delete old policy vectors: %s", e)

    start = time.monotonic()
    processor = PolicyProcessor(model_type=model_type)
    try:
        result = processor.process(
            gcs_uri=body.gcs_uri,
            policy_name=body.policy_name,
            policy_id=policy_id,
            metadata=body.metadata,
        )
    except Exception as e:
        logger.error("Policy update failed: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Policy processing failed. Please try again.",
        ) from None
    elapsed_ms = int((time.monotonic() - start) * 1000)

    updated_doc = PolicyDocument(
        policy_id=policy_id,
        policy_name=result.policy_name,
        gcs_uri=body.gcs_uri,
        page_count=result.page_count,
        ocr_engine=body.ocr_engine,
        category=(body.metadata or {}).get("category"),
        metadata=body.metadata or {},
        status="processed" if result.success else "error",
    )
    repo.add(updated_doc)

    return PolicyResponse(
        policy_id=policy_id,
        policy_name=result.policy_name,
        status="processed" if result.success else "error",
        page_count=result.page_count,
        processing_time_ms=elapsed_ms,
        category=updated_doc.category,
        created_at=updated_doc.created_at,
        metadata=body.metadata,
    )


@router.delete("/{policy_id}", response_model=PolicyDeleteResponse)
async def delete_policy(policy_id: str):
    """Remove a policy document and its vectors."""
    repo = get_policy_repository()
    doc = repo.get(policy_id)
    if doc is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Policy '{policy_id}' not found"
        )

    vectors_deleted = 0
    try:
        vectors_deleted = delete_session_documents(
            session_id=policy_id,
            collection_name="policy_documents",
        )
    except Exception as e:
        logger.error("Failed to delete policy vectors: %s", e)

    repo.delete(policy_id)

    return PolicyDeleteResponse(
        policy_id=policy_id,
        status="deleted",
        vectors_deleted=vectors_deleted,
    )
