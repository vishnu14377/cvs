"""Policy endpoint request/response models."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class CreatePolicyRequest(BaseModel):
    """Request body for POST /api/v1/policies."""

    gcs_uri: str = Field(..., description="GCS URI of the policy PDF")
    policy_name: str = Field(..., min_length=1, max_length=500, description="Human-readable name")
    ocr_engine: str = Field(default="mistral", description="OCR engine to use")
    metadata: dict[str, str] | None = Field(
        default=None, description="Optional metadata (category, effectiveDate, etc.)"
    )


class PolicyResponse(BaseModel):
    """Response for policy operations."""

    policy_id: str
    policy_name: str
    status: str
    page_count: int = 0
    processing_time_ms: int = 0
    category: str | None = None
    created_at: datetime | None = None
    metadata: dict[str, str] | None = None


class PolicyListResponse(BaseModel):
    """Response for GET /api/v1/policies."""

    policies: list[PolicyResponse] = Field(default_factory=list)


class PolicyDeleteResponse(BaseModel):
    """Response for DELETE /api/v1/policies/{policyId}."""

    policy_id: str
    status: str = "deleted"
    vectors_deleted: int = 0


class BatchCreatePolicyRequest(BaseModel):
    """Request body for POST /api/v1/policies/batch."""

    documents: list[CreatePolicyRequest] = Field(
        ..., min_length=1, max_length=20, description="List of policy documents to process (1-20)"
    )


class BatchPolicyResult(BaseModel):
    """Result for a single document within a batch operation."""

    policy_id: str | None = None
    title: str
    status: Literal["success", "failed"]
    error: str | None = None


class BatchSummary(BaseModel):
    """Aggregate counts for a batch operation."""

    total: int
    succeeded: int
    failed: int


class BatchPolicyResponse(BaseModel):
    """Response for POST /api/v1/policies/batch."""

    results: list[BatchPolicyResult]
    summary: BatchSummary
