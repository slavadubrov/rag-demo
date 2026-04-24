"""Unit tests for intent classification."""

from __future__ import annotations

import gradio as gr

from rag_demo.retrieve import _allowed_types_for, infer_query_kind


def test_infer_table():
    assert infer_query_kind("How many parameters does ResNet-50 have?") == "table"
    assert infer_query_kind("Compare BLEU scores across models.") == "table"


def test_infer_figure():
    assert infer_query_kind("What does Figure 3 show about attention?") == "figure"
    assert infer_query_kind("Describe the architecture diagram.") == "figure"


def test_infer_mixed():
    assert infer_query_kind("Show me the chart and the accompanying table.") == "mixed"


def test_infer_text_default():
    # "summarize this" has no table/figure hint — must be text.
    assert infer_query_kind("Summarize this paper.") == "text"


def test_allowed_types_table_first_pass_prefers_tables():
    passes = _allowed_types_for("table")
    assert "table_chunk" in passes[0]
    assert "caption_chunk" in passes[0]


def test_allowed_types_figure_first_pass_prefers_figures():
    passes = _allowed_types_for("figure")
    assert "figure_chunk" in passes[0]


def test_allowed_types_text_omits_visual_in_first_pass():
    passes = _allowed_types_for("text")
    assert "table_chunk" not in passes[0]
    assert "figure_chunk" not in passes[0]


def test_run_full_rebuild_yields_two_outputs(monkeypatch):
    from rag_demo import app
    from rag_demo.schema import DocumentSummary, IngestReport

    report = IngestReport(
        documents=[
            DocumentSummary(
                doc_id="doc",
                title="Doc",
                pdf_path="/tmp/doc.pdf",
                page_count=2,
                chunk_count=5,
            )
        ],
        multimodal_chunks=5,
        baseline_chunks=3,
        elapsed_seconds=1.2,
    )

    monkeypatch.setattr(app, "reset_client", lambda: None)
    monkeypatch.setattr(app, "rebuild_index", lambda: report)
    monkeypatch.setattr(
        app, "_doc_choices", lambda: [("Doc (doc, 2p, 5 chunks)", "doc")]
    )

    events = list(app._run_full_rebuild())
    assert len(events) == 2
    assert len(events[0]) == 2
    assert events[0][0].startswith("⏳ Rebuilding index")
    assert events[0][1] == gr.skip()
    assert len(events[1]) == 2
    assert events[1][0].startswith("✅ Rebuilt.")
