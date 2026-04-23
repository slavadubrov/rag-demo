"""Pydantic schema round-trips."""

from __future__ import annotations

from rag_demo.schema import BBox, Chunk, DocumentSummary, Evidence, IngestReport


def test_chunk_minimal_defaults():
    c = Chunk(
        chunk_id="doc::sec::00001",
        doc_id="doc",
        page_num=3,
        element_type="section_chunk",
        text="hello",
    )
    assert c.section_path == []
    assert c.bbox is None
    assert c.crop_ref is None


def test_chunk_round_trip_json():
    c = Chunk(
        chunk_id="doc::tbl::00001",
        doc_id="doc",
        page_num=1,
        element_type="table_chunk",
        text="table text",
        bbox=BBox(l=0, t=0, r=10, b=10),
        table_markdown="|a|b|\n|-|-|\n|1|2|",
        crop_ref="/tmp/crop.png",
        extra={"caption": "sample"},
    )
    raw = c.model_dump_json()
    reloaded = Chunk.model_validate_json(raw)
    assert reloaded == c


def test_evidence_accepts_optional_paths():
    c = Chunk(
        chunk_id="x", doc_id="d", page_num=1, element_type="figure_chunk", text="fig"
    )
    e = Evidence(chunk=c, score=0.7)
    assert e.page_image_path is None
    assert e.crop_image_path is None


def test_ingest_report_serializes():
    report = IngestReport(
        documents=[
            DocumentSummary(
                doc_id="d1", title="t", pdf_path="x.pdf", page_count=3, chunk_count=9
            )
        ],
        multimodal_chunks=9,
        baseline_chunks=4,
        elapsed_seconds=1.5,
    )
    d = report.model_dump()
    assert d["multimodal_chunks"] == 9
    assert d["documents"][0]["chunk_count"] == 9
