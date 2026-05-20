"""
Data models for the OCR Orchestrator.

This module contains Pydantic models for the OCR orchestration pipeline results.

Models:
    - SubFileProcessingResult: Result of processing a single sub-file
    - OcrOrchestrationResult: Result of the complete OCR orchestration pipeline
    - FallbackStats: Statistics about fallback usage across all sub-files
"""

from typing import Literal

from pydantic import BaseModel, Field

# Type alias for model names
ModelName = Literal["mistral", "llm"]


class SubFileProcessingResult(BaseModel):
    """Result of processing a single sub-file."""

    gcs_uri: str = Field(..., description="GCS URI of the sub-file that was processed.")

    success: bool = Field(..., description="Whether the processing was successful.")

    extracted_text_uri: str | None = Field(
        default=None, description="GCS URI of the saved extracted text JSON (if successful)."
    )

    page_count: int = Field(default=0, description="Number of pages processed.")

    error: str | None = Field(default=None, description="Error message if processing failed.")

    model_used: ModelName | None = Field(
        default=None, description="Which model was used to process this sub-file."
    )

    fallback_used: bool = Field(
        default=False, description="Whether fallback from primary to secondary model was used."
    )

    primary_error: str | None = Field(
        default=None, description="Error from primary model if fallback was triggered."
    )


class FallbackStats(BaseModel):
    """Statistics about fallback usage across all processed sub-files."""

    total_processed: int = Field(default=0, description="Total number of sub-files processed.")

    primary_success_count: int = Field(
        default=0, description="Number of sub-files successfully processed by primary model."
    )

    fallback_success_count: int = Field(
        default=0, description="Number of sub-files successfully processed after fallback."
    )

    both_failed_count: int = Field(
        default=0, description="Number of sub-files where both primary and fallback failed."
    )

    mistral_used_count: int = Field(
        default=0, description="Number of sub-files processed by Mistral model."
    )

    llm_used_count: int = Field(
        default=0, description="Number of sub-files processed by LLM (Gemini) model."
    )

    fallback_rate: float = Field(
        default=0.0,
        description="Percentage of successful sub-files that required fallback (0-100).",
    )

    @classmethod
    def from_results(cls, results: list[SubFileProcessingResult]) -> "FallbackStats":
        """
        Calculate fallback statistics from a list of sub-file processing results.

        Args:
            results: List of SubFileProcessingResult from orchestration

        Returns:
            FallbackStats with computed values
        """
        total = len(results)
        if total == 0:
            return cls()

        primary_success = sum(1 for r in results if r.success and not r.fallback_used)
        fallback_success = sum(1 for r in results if r.success and r.fallback_used)
        both_failed = sum(1 for r in results if not r.success and r.fallback_used)

        mistral_count = sum(1 for r in results if r.model_used == "mistral")
        llm_count = sum(1 for r in results if r.model_used == "llm")

        successful_count = primary_success + fallback_success
        fallback_rate = (fallback_success / successful_count * 100) if successful_count > 0 else 0.0

        return cls(
            total_processed=total,
            primary_success_count=primary_success,
            fallback_success_count=fallback_success,
            both_failed_count=both_failed,
            mistral_used_count=mistral_count,
            llm_used_count=llm_count,
            fallback_rate=round(fallback_rate, 2),
        )


class OcrOrchestrationResult(BaseModel):
    """Result of the complete OCR orchestration pipeline."""

    session_id: str = Field(..., description="The unique key used for this processing session.")

    source_uri: str = Field(..., description="Original source file URI.")

    success: bool = Field(
        ..., description="Overall success status (True if all sub-files processed successfully)."
    )

    total_sub_files: int = Field(default=0, description="Total number of sub-files processed.")

    successful_sub_files: int = Field(
        default=0, description="Number of sub-files processed successfully."
    )

    failed_sub_files: int = Field(
        default=0, description="Number of sub-files that failed processing."
    )

    total_pages: int = Field(
        default=0, description="Total number of pages processed across all sub-files."
    )

    sub_file_results: list[SubFileProcessingResult] = Field(
        default_factory=list, description="Individual results for each sub-file."
    )

    extracted_text_uris: list[str] = Field(
        default_factory=list, description="GCS URIs of all successfully saved extracted text JSONs."
    )

    error: str | None = Field(
        default=None, description="Error message if orchestration failed at a high level."
    )

    fallback_stats: FallbackStats | None = Field(
        default=None, description="Statistics about fallback usage across all sub-files."
    )

    def compute_fallback_stats(self) -> "OcrOrchestrationResult":
        """
        Compute and set fallback statistics from sub-file results.

        Returns:
            Self with fallback_stats populated.
        """
        self.fallback_stats = FallbackStats.from_results(self.sub_file_results)
        return self
