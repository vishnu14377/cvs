"""
Data models for the OCR pipeline.

This package contains Pydantic models and dataclasses for OCR responses
and processing results.

OCR Response Models (by layer):

    Mistral OCR (raw response) - from mistral_response.py:
        - MistralPageResponse: Page with 'index' and 'markdown'
        - MistralOcrResponse: Collection of Mistral pages

    LLM OCR (structured output) - from llm_response.py:
        - PageExtraction: Page with 'index' and 'extracted_text'
        - DocumentExtraction: Collection of pages for LangChain structured output

    OcrModelClient (normalized response) - from normalized_response.py:
        - NormalizedPage: Unified page with 'index' and 'extracted_text'
        - NormalizedOcrResponse: Normalized result consumed by SubFileHandler

SubFile Handler Models - from sub_file_models.py:
    - PageInfo: Page with original document mapping
    - SubFileMetadata: Document name and page range info
    - SubFileResult: Complete sub-file processing result

Orchestrator Models - from orchestrator_models.py:
    - SubFileProcessingResult: Result of processing a single sub-file
    - OcrOrchestrationResult: Result of the complete OCR orchestration pipeline
"""

# Mistral OCR models
# LLM OCR models (for structured output)
from src.ocr.data_models.llm_response import (
    DocumentExtraction,
    PageExtraction,
)
from src.ocr.data_models.mistral_response import (
    MistralOcrResponse,
    MistralPageResponse,
)

# Normalized models (OcrModelClient output)
from src.ocr.data_models.normalized_response import (
    NormalizedOcrResponse,
    NormalizedPage,
)

# Orchestrator Models
from src.ocr.data_models.orchestrator_models import (
    OcrOrchestrationResult,
    SubFileProcessingResult,
)

# SubFile Handler Models
from src.ocr.data_models.sub_file_models import (
    PageInfo,
    SubFileMetadata,
    SubFileResult,
    map_page_to_original,
)

__all__ = [
    # Mistral OCR models
    "MistralPageResponse",
    "MistralOcrResponse",
    # LLM OCR models
    "PageExtraction",
    "DocumentExtraction",
    # Normalized models (OcrModelClient output)
    "NormalizedPage",
    "NormalizedOcrResponse",
    # SubFile Handler Models
    "PageInfo",
    "SubFileMetadata",
    "SubFileResult",
    "map_page_to_original",
    # Orchestrator Models
    "SubFileProcessingResult",
    "OcrOrchestrationResult",
]
