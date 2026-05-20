"""Utility to load OCR prompt templates from .txt files."""

from pathlib import Path

PROMPTS_DIR = Path(__file__).parent

def load_full_extraction_prompt() -> str:
    """Load the full extraction prompt (text + summary + insights)."""
    return (PROMPTS_DIR / "full_extraction_prompt.txt").read_text(encoding="utf-8")

def load_enrichment_prompt(page_number: int, page_text: str) -> str:
    """
    Load and format the enrichment prompt (summary + insights only).
    
    Args:
        page_number: The page number being enriched
        page_text: Already extracted text for the page
        
    Returns:
        Formatted prompt string
    """
    template = (PROMPTS_DIR / "enrichment_prompt.txt").read_text(encoding="utf-8")
    return template.replace("{page_number}", str(page_number)).replace("{page_text}", page_text)