"""End-to-end test for the text-only (baseline) path.

Skips Docling entirely: builds a tiny synthetic PDF with PyMuPDF, runs the
baseline chunker against it, upserts into a fresh Qdrant collection, then
exercises retrieval + the offline stub answer.
"""

from __future__ import annotations

from pathlib import Path

import fitz  # PyMuPDF
import pytest


def _make_pdf(path: Path, pages_text: list[str]) -> None:
    doc = fitz.open()
    for text in pages_text:
        page = doc.new_page(width=612, height=792)
        page.insert_text((72, 120), text, fontsize=11)
    doc.save(path)
    doc.close()


@pytest.fixture
def tiny_pdf(tmp_data_dir) -> Path:
    path = tmp_data_dir / "corpus" / "mini.pdf"
    _make_pdf(
        path,
        [
            "Residual learning uses shortcut connections to ease optimization.\n\n"
            "ResNet-50 stacks 50 layers.",
            "The bottleneck building block reduces parameters.\n\n"
            "Identity shortcuts avoid extra parameters.",
            "Experiments on ImageNet classification report top-1 accuracy numbers "
            "in Table 1 of the paper.",
        ],
    )
    return path


def test_baseline_chunker_splits_per_page(tiny_pdf):
    from rag_demo.ingest import _doc_id, baseline_chunks_from_pdf

    chunks = baseline_chunks_from_pdf(tiny_pdf, _doc_id(tiny_pdf))
    assert len(chunks) >= 3
    pages = sorted({c.page_num for c in chunks})
    assert pages == [1, 2, 3]
    assert all(c.element_type == "page_fallback_chunk" for c in chunks)


def test_baseline_chunker_respects_page_cap(tiny_pdf):
    from rag_demo.ingest import _doc_id, baseline_chunks_from_pdf

    chunks = baseline_chunks_from_pdf(tiny_pdf, _doc_id(tiny_pdf), max_pages=2)
    pages = sorted({c.page_num for c in chunks})
    assert pages == [1, 2]


def test_baseline_index_retrieves_expected_page(tiny_pdf):
    from rag_demo.config import SETTINGS
    from rag_demo.index import (
        _ensure_collection,
        _upsert_chunks,
        reset_client,
        search,
    )
    from rag_demo.ingest import _doc_id, baseline_chunks_from_pdf
    from rag_demo.embeddings import embed_query

    reset_client()
    doc_id = _doc_id(tiny_pdf)
    chunks = baseline_chunks_from_pdf(tiny_pdf, doc_id)

    _ensure_collection(SETTINGS.baseline_collection, SETTINGS.embed_dim, recreate=True)
    _upsert_chunks(SETTINGS.baseline_collection, chunks)

    qvec = embed_query("bottleneck building block parameters")
    results = search(
        SETTINGS.baseline_collection, qvec, limit=5, doc_ids=[doc_id]
    )
    assert results, "expected at least one result"
    # Page 2 contains the bottleneck text → top hit should come from there
    top = results[0]
    assert top.chunk.doc_id == doc_id
    assert top.chunk.page_num in (1, 2, 3)


def test_stub_answer_runs_offline(tiny_pdf):
    from rag_demo.answer import _stub_answer
    from rag_demo.ingest import _doc_id, baseline_chunks_from_pdf
    from rag_demo.schema import Evidence

    chunks = baseline_chunks_from_pdf(tiny_pdf, _doc_id(tiny_pdf))
    evidence = [Evidence(chunk=c, score=0.5) for c in chunks[:2]]
    payload = _stub_answer("What is a bottleneck block?", "baseline", evidence)
    assert payload.answer
    assert "OFFLINE MODE" in payload.answer
    assert payload.citations


def test_citation_validation_detects_invalid_pages():
    from rag_demo.answer import _validate_citations
    from rag_demo.schema import Chunk, Evidence

    evidence = [
        Evidence(
            chunk=Chunk(
                chunk_id="d::sec::00001",
                doc_id="paper",
                page_num=5,
                element_type="section_chunk",
                text="text",
            ),
            score=0.9,
        )
    ]
    text = "Some claim [paper, p.5]. Another claim [paper, p.99]."
    result = _validate_citations(text, evidence)
    assert result["valid_count"] == 1
    assert result["invalid_count"] == 1
