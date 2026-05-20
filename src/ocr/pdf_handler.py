"""
PDF splitting utility for OCR processing.

Flow (both public entry points):
  1. Validate the local PDF path.
  2. Resolve GCS object prefix (session-scoped under GCS_WORKING_FOLDER).
  3. Open the PDF, decide how many pages go in each chunk (by size or fixed count).
  4. For each chunk: write a temporary PDF on disk, upload to GCS, then delete temps.

GCS layout per upload:
  gs://<bucket>/<GCS_WORKING_FOLDER>/<unique_key>/tmp/<stem>_p<start>-<end>.pdf
"""

import math
import os
import tempfile
from pathlib import Path

from PyPDF2 import PdfReader, PdfWriter

from src.core.config import ocr_config
from src.core.gcs_client import upload_to_gcs
from src.core.logger import get_logger

logger = get_logger(__name__)


def calculate_pages_by_size(file_size_mb: float, total_pages: int, size_limit_mb: float = 5) -> int:
    """
    Pages per chunk when splitting by target file size.

    Example: 20 MB PDF, 5 MB limit → ~4 files → spread 100 pages as ~25 pages/chunk.
    Uses ceil so we never undershoot the number of splits (avoids oversized chunks).
    """
    # How many output files we need if each is at most size_limit_mb
    num_sub_files = max(1, math.ceil(file_size_mb / size_limit_mb))
    # Spread all pages across that many files as evenly as possible
    pages_per_chunk = math.ceil(total_pages / num_sub_files)
    return max(1, pages_per_chunk)


def _validate_pdf_input(pdf_path: str) -> None:
    """Ensure the path exists and looks like a PDF before we read it."""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"File not found: {pdf_path}")
    if not pdf_path.lower().endswith(".pdf"):
        raise ValueError(f"File must be a PDF: {pdf_path}")


def _get_gcs_prefix(unique_key: str) -> str:
    """
    Build the GCS object prefix for this job's split files.

    Note: The base folder (GCS_WORKING_FOLDER) is automatically prepended by gcs_client.

    Returns:
        gcs_prefix: Relative path prefix, e.g. ``session-abc/tmp``.

    ``unique_key`` isolates one session/job so splits from different runs do not collide.
    """
    uk = str(unique_key).strip()
    if not uk:
        raise ValueError("unique_key must not be empty")
    if "/" in uk or "\\" in uk or ".." in uk:
        raise ValueError("unique_key must not contain slashes or '..'")

    # Relative prefix: session / temp_folder (actual PDF names appended per chunk)
    # Base folder is handled by gcs_client
    # Use configured temp folder name from ocr_config
    temp_folder = ocr_config.GCS_TEMP_FOLDER
    prefix = f"{uk}/{temp_folder}"
    return prefix


def _write_page_range_to_file(reader: PdfReader, start: int, end: int, local_path: str) -> None:
    """
    Build one PDF containing pages start..end-1 (0-based, end exclusive).

    PyPDF2 copies page objects into a new writer; we flush to disk so upload_file_to_gcs
    can read a real file path.
    """
    writer = PdfWriter()
    for page_num in range(start, end):
        writer.add_page(reader.pages[page_num])
    with open(local_path, "wb") as f:
        writer.write(f)


def _split_reader_to_gcs(
    reader: PdfReader,
    pdf_stem: str,
    pages_per_chunk: int,
    gcs_prefix: str,
) -> list[str]:
    """
    Core split + upload loop shared by split_pdf_by_size and split_pdf_by_pages.

    Walks the PDF in windows of ``pages_per_chunk`` pages. Each window becomes one file
    named ``{stem}_p{first}-{last}.pdf``, uploaded under ``gcs_prefix``.
    Local files live in a single temp directory and are removed in ``finally`` so we
    do not leave chunks on disk after upload (or after failure partway through).
    """
    total_pages = len(reader.pages)
    temp_dir = tempfile.mkdtemp()
    temp_files: list[str] = []
    gcs_urls: list[str] = []

    try:
        chunk_num = 1
        # Step through the document in fixed-size page windows
        for start in range(0, total_pages, pages_per_chunk):
            end = min(start + pages_per_chunk, total_pages)
            filename = f"{pdf_stem}_p{start + 1}-{end}.pdf"
            local_path = os.path.join(temp_dir, filename)

            _write_page_range_to_file(reader, start, end, local_path)
            temp_files.append(local_path)

            size_mb = os.path.getsize(local_path) / (1024 * 1024)
            logger.info(
                "Chunk %s: %s (pages %s-%s, %.2f MB)",
                chunk_num,
                filename,
                start + 1,
                end,
                size_mb,
            )

            # Full object key = prefix + filename; returns gs://bucket/key
            gcs_urls.append(upload_to_gcs(local_path, f"{gcs_prefix}/{filename}"))
            chunk_num += 1

        logger.info("Uploaded %s chunk(s) to GCS", len(gcs_urls))
        return gcs_urls

    finally:
        # Always tear down local copies; GCS already has the bytes
        for path in temp_files:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except OSError as e:
                logger.warning("Could not remove temp file %s: %s", path, e)
        try:
            if os.path.exists(temp_dir):
                os.rmdir(temp_dir)
        except OSError as e:
            logger.warning("Could not remove temp dir %s: %s", temp_dir, e)


def split_pdf_by_size(
    pdf_path: str,
    unique_key: str,
    size_limit_mb: float = 5,
) -> list[str]:
    """
    Split so each output PDF is roughly under ``size_limit_mb`` (best effort).

    Page count per chunk is derived from total file size vs limit, then the same
    upload pipeline as fixed-page splitting runs.
    """
    _validate_pdf_input(pdf_path)

    gcs_prefix = _get_gcs_prefix(unique_key)
    logger.info("GCS upload prefix: %s/", gcs_prefix)

    reader = PdfReader(pdf_path)
    total_pages = len(reader.pages)
    file_size_mb = os.path.getsize(pdf_path) / (1024 * 1024)
    logger.info(
        "PDF %s: %s pages, %.2f MB",
        os.path.basename(pdf_path),
        total_pages,
        file_size_mb,
    )

    # Derive chunk size from MB limits, then reuse shared upload logic
    pages_per_chunk = calculate_pages_by_size(file_size_mb, total_pages, size_limit_mb)
    logger.info("Using %s pages per chunk (limit %s MB)", pages_per_chunk, size_limit_mb)

    return _split_reader_to_gcs(
        reader,
        Path(pdf_path).stem,
        pages_per_chunk,
        gcs_prefix,
    )


def split_pdf_by_pages(
    pdf_path: str,
    pages_per_chunk: int,
    unique_key: str,
) -> list[str]:
    """
    Split every ``pages_per_chunk`` pages into one output PDF (last chunk may be smaller).

    Same GCS session layout as split_pdf_by_size; only the chunking rule differs.
    """
    _validate_pdf_input(pdf_path)
    if pages_per_chunk <= 0:
        raise ValueError(f"pages_per_chunk must be greater than 0, got {pages_per_chunk}")

    gcs_prefix = _get_gcs_prefix(unique_key)
    logger.info("GCS upload prefix: %s/", gcs_prefix)

    reader = PdfReader(pdf_path)
    total_pages = len(reader.pages)
    logger.info(
        "PDF %s: %s pages, %s pages per chunk",
        os.path.basename(pdf_path),
        total_pages,
        pages_per_chunk,
    )

    return _split_reader_to_gcs(
        reader,
        Path(pdf_path).stem,
        pages_per_chunk,
        gcs_prefix,
    )


if __name__ == "__main__":
    pdf_path = "test1.pdf"
    unique_key1 = "test11"
    unique_key2 = "test3"
    size_limit_mb = 5
    pages_per_chunk = 10
    split_pdf_by_size(pdf_path, unique_key1, size_limit_mb)
    split_pdf_by_pages(pdf_path, pages_per_chunk, unique_key2)
