"""Pydantic schemas for retrieval units, evidence, and answers."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

ElementType = Literal[
    "section_chunk",
    "table_chunk",
    "figure_chunk",
    "caption_chunk",
    "page_fallback_chunk",
]


class BBox(BaseModel):
    l: float = 0.0
    t: float = 0.0
    r: float = 0.0
    b: float = 0.0


class Chunk(BaseModel):
    """One typed retrieval unit."""

    chunk_id: str
    doc_id: str
    page_num: int
    element_type: ElementType
    text: str
    section_path: list[str] = Field(default_factory=list)
    bbox: BBox | None = None
    parent_id: str | None = None
    # Path to the full page image (always present if rendering succeeded)
    image_ref: str | None = None
    # Path to a cropped figure/table image extracted by Docling (when available).
    # Preferred over `image_ref` when attaching images to the vision model.
    crop_ref: str | None = None
    table_html: str | None = None
    table_markdown: str | None = None
    extra: dict = Field(default_factory=dict)


class DocumentSummary(BaseModel):
    doc_id: str
    title: str
    pdf_path: str
    page_count: int
    chunk_count: int = 0


class Evidence(BaseModel):
    chunk: Chunk
    score: float
    page_image_path: str | None = None
    crop_image_path: str | None = None


class AnswerPayload(BaseModel):
    question: str
    mode: Literal["multimodal", "baseline"]
    answer: str
    citations: list[dict] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    debug: dict = Field(default_factory=dict)


class IngestReport(BaseModel):
    documents: list[DocumentSummary]
    multimodal_chunks: int
    baseline_chunks: int
    elapsed_seconds: float
