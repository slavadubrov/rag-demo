"""Paths, environment, and tunable defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CORPUS_DIR = PROJECT_ROOT / "corpus"
UPLOAD_DIR = CORPUS_DIR / "uploads"
DATA_DIR = PROJECT_ROOT / "data"
PAGE_IMAGE_DIR = DATA_DIR / "pages"
CROP_IMAGE_DIR = DATA_DIR / "crops"
DOCLING_DIR = DATA_DIR / "docling"
CHUNKS_DIR = DATA_DIR / "chunks"
QDRANT_DIR = DATA_DIR / "qdrant"
MANIFEST_PATH = DATA_DIR / "manifest.json"
EVAL_DIR = DATA_DIR / "eval"

for _p in (
    DATA_DIR,
    PAGE_IMAGE_DIR,
    CROP_IMAGE_DIR,
    DOCLING_DIR,
    CHUNKS_DIR,
    QDRANT_DIR,
    UPLOAD_DIR,
    EVAL_DIR,
):
    _p.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Settings:
    openai_api_key: str | None
    embed_model: str
    embed_dim: int
    answer_model: str
    page_render_dpi: int
    multimodal_collection: str
    baseline_collection: str
    embed_backend: str  # "openai" or "hash"
    max_pages_per_doc: int


def load_settings() -> Settings:
    api_key = os.getenv("OPENAI_API_KEY")
    backend = os.getenv("EMBED_BACKEND", "openai" if api_key else "hash").lower()
    return Settings(
        openai_api_key=api_key,
        embed_model=os.getenv("EMBED_MODEL", "text-embedding-3-small"),
        embed_dim=int(os.getenv("EMBED_DIM", "1536")),
        answer_model=os.getenv("ANSWER_MODEL", "gpt-4o-mini"),
        page_render_dpi=int(os.getenv("PAGE_RENDER_DPI", "150")),
        multimodal_collection=os.getenv("MULTIMODAL_COLLECTION", "multimodal_chunks"),
        baseline_collection=os.getenv("BASELINE_COLLECTION", "baseline_chunks"),
        embed_backend=backend,
        max_pages_per_doc=int(os.getenv("MAX_PAGES_PER_DOC", "60")),
    )


SETTINGS = load_settings()
