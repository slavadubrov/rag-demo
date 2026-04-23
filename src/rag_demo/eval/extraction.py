"""Extraction stats: how many chunk types Docling produced per doc."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..ingest import load_chunks


@dataclass
class ExtractionStats:
    filename: str
    doc_id: str | None
    ingested: bool
    page_count: int = 0
    chunk_count: int = 0
    chunk_type_counts: dict = field(default_factory=dict)
    pages_with_typed_visual: int = 0
    pages_total_with_chunks: int = 0
    notes: list[str] = field(default_factory=list)


def filename_to_doc_id(filename: str, doc_manifest: dict) -> str | None:
    stem = Path(filename).stem.lower()
    for doc_id, info in doc_manifest.items():
        pdf = info.get("pdf_path", "")
        if Path(pdf).stem.lower() == stem:
            return doc_id
        if doc_id.lower().startswith(stem[:20]):
            return doc_id
    return None


def compute_extraction_stats(
    corpus_manifest: dict, doc_manifest: dict
) -> list[ExtractionStats]:
    out: list[ExtractionStats] = []
    for entry in corpus_manifest.get("documents", []):
        filename = entry["filename"]
        doc_id = filename_to_doc_id(filename, doc_manifest)
        stats = ExtractionStats(filename=filename, doc_id=doc_id, ingested=bool(doc_id))
        if not doc_id:
            stats.notes.append("not yet ingested — run rebuild after downloading the PDF")
            out.append(stats)
            continue

        info = doc_manifest[doc_id]
        stats.page_count = info.get("page_count", 0)
        stats.chunk_count = info.get("chunk_count", 0)

        chunks = load_chunks(doc_id)
        type_counts: dict[str, int] = {}
        pages_with_visual: set[int] = set()
        pages_with_chunks: set[int] = set()
        for c in chunks:
            type_counts[c.element_type] = type_counts.get(c.element_type, 0) + 1
            pages_with_chunks.add(c.page_num)
            if c.element_type in ("table_chunk", "figure_chunk"):
                pages_with_visual.add(c.page_num)
        stats.chunk_type_counts = type_counts
        stats.pages_with_typed_visual = len(pages_with_visual)
        stats.pages_total_with_chunks = len(pages_with_chunks)
        if not type_counts.get("table_chunk") and not type_counts.get("figure_chunk"):
            stats.notes.append(
                "no table or figure chunks extracted — Docling may have missed visuals"
            )
        out.append(stats)
    return out
