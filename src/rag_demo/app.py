"""Gradio Blocks UI for the unified multimodal PDF RAG demo.

Features combined from all three reference builds:

- **Claude**: tabbed layout (Answer / Evidence / Page Preview / Tables /
  Baseline Comparison / Debug), Dataframe for evidence details, retrieved
  chunks table, corpus checkbox selector, side-by-side mode, offline-mode
  warnings.
- **Gemini**: real LLM token streaming (answer renders incrementally), a live
  "Run Ingestion & Indexing" button, and a dynamic **PDF upload** component
  that ingests new PDFs on the fly.
- **Codex**: custom CSS / typography, "Rebuild Artifacts" button that refreshes
  the selector choices in place, status bar that reports what just happened.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path

import gradio as gr

from .baseline import query as run_query
from .baseline import query_stream
from .config import SETTINGS, UPLOAD_DIR
from .index import list_documents, rebuild_index
from .ingest import docling_available
from .schema import AnswerPayload, Evidence

try:
    # Used to refresh the singleton client after a rebuild
    from .index import ingest_single_pdf, reset_client
except ImportError:  # pragma: no cover
    ingest_single_pdf = None  # type: ignore
    reset_client = None  # type: ignore

logger = logging.getLogger(__name__)


SAMPLE_QUESTIONS = [
    "What does Figure 1 in the ResNet paper actually show?",
    "What is the architecture of the Transformer encoder block?",
    "Compare the parameters and depth of ResNet-50 vs ResNet-152.",
    "How does the RAG paper combine the retriever with the generator at training time?",
    "Which BLEU scores does the Transformer report on WMT 2014 EN-DE and EN-FR?",
    "Which chart in the OECD report shows the largest gap between target and actual?",
]


CSS = """
:root {
  --paper: #f7f4ec;
  --ink: #17212b;
  --teal: #0f766e;
  --amber: #d97706;
  --panel: rgba(255, 255, 255, 0.94);
}
body, .gradio-container {
  background:
    radial-gradient(circle at top left, rgba(15,118,110,0.15), transparent 32%),
    radial-gradient(circle at bottom right, rgba(217,119,6,0.14), transparent 30%),
    var(--paper);
  color: var(--ink);
  font-family: "Iowan Old Style", Georgia, serif;
}
.app-shell {
  border: 1px solid rgba(23,33,43,0.08);
  border-radius: 22px;
  background: var(--panel);
  box-shadow: 0 18px 60px rgba(23,33,43,0.08);
  padding: 18px 22px;
}
.eyebrow {
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--teal);
  font-size: 0.78rem;
}
.warn {
  border-left: 4px solid var(--amber);
  background: rgba(217, 119, 6, 0.08);
  padding: 10px 14px;
  border-radius: 8px;
}
footer {visibility: hidden}
"""


# --------------------------------------------------------------------------- #
# Formatting helpers
# --------------------------------------------------------------------------- #


def _doc_choices() -> list[tuple[str, str]]:
    docs = list_documents()
    return [
        (f"{d.title} ({d.doc_id}, {d.page_count}p, {d.chunk_count} chunks)", d.doc_id)
        for d in docs
    ]


def _format_citations(payload: AnswerPayload | None) -> str:
    if not payload or not payload.citations:
        return "_No citations._"
    seen = set()
    lines = []
    for c in payload.citations:
        key = (c["doc_id"], c["page_num"])
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"- **{c['doc_id']}**, p.{c['page_num']} ({c.get('type', '?')})")
    return "\n".join(lines)


def _format_evidence_md(evidence: list[Evidence]) -> str:
    if not evidence:
        return "_No evidence retrieved._"
    out = []
    for i, e in enumerate(evidence, 1):
        c = e.chunk
        section = " / ".join(c.section_path) if c.section_path else ""
        snippet = c.text.strip().replace("\n", " ")
        if len(snippet) > 350:
            snippet = snippet[:350] + "…"
        out.append(
            f"### {i}. `{c.element_type}` — page {c.page_num} (score {e.score:.3f})\n"
            f"**Doc:** `{c.doc_id}`  \n"
            + (f"**Section:** {section}  \n" if section else "")
            + f"**Chunk:** `{c.chunk_id}`\n\n> {snippet}\n"
        )
    return "\n---\n".join(out)


def _evidence_table_rows(evidence: list[Evidence]) -> list[list]:
    return [
        [
            i + 1,
            e.chunk.element_type,
            e.chunk.doc_id,
            e.chunk.page_num,
            f"{e.score:.3f}",
            (e.chunk.text or "").strip().replace("\n", " ")[:120],
        ]
        for i, e in enumerate(evidence)
    ]


def _gallery_items(evidence: list[Evidence]) -> list[tuple[str, str]]:
    """Prefer crops (figures/tables) over page images; dedup."""
    items: list[tuple[str, str]] = []
    seen: set[str] = set()

    for e in evidence:
        for path in (e.crop_image_path, e.chunk.crop_ref):
            if not path or not Path(path).exists() or path in seen:
                continue
            seen.add(path)
            items.append(
                (
                    path,
                    f"{e.chunk.doc_id} — {e.chunk.element_type} (p.{e.chunk.page_num})",
                )
            )
            break

    for e in evidence:
        for path in (e.page_image_path, e.chunk.image_ref):
            if not path or not Path(path).exists() or path in seen:
                continue
            seen.add(path)
            items.append((path, f"{e.chunk.doc_id} — p.{e.chunk.page_num}"))
            break
    return items


def _table_evidence_markdown(evidence: list[Evidence]) -> str:
    parts = []
    for e in evidence:
        if e.chunk.element_type != "table_chunk":
            continue
        cap = (e.chunk.extra or {}).get("caption", "") or ""
        parts.append(
            f"#### Page {e.chunk.page_num} — {e.chunk.doc_id}\n"
            + (f"_{cap}_\n\n" if cap else "")
            + (e.chunk.table_markdown or "_(table markdown unavailable)_")
        )
    return "\n\n".join(parts) or "_No table evidence retrieved._"


def _validation_line(payload: AnswerPayload | None) -> str:
    if not payload:
        return ""
    v = (payload.debug or {}).get("citation_validation") or {}
    if not v:
        return ""
    ok = v.get("valid_count", 0)
    bad = v.get("invalid_count", 0)
    total = ok + bad
    if total == 0:
        return "_No inline citations detected in the answer._"
    badge = "✅" if bad == 0 else "⚠️"
    return f"{badge} citation check: {ok}/{total} resolved against retrieved pages"


# --------------------------------------------------------------------------- #
# Ask handler — streams tokens for the primary mode, falls back to blocking
# for the comparison mode.
# --------------------------------------------------------------------------- #


def _empty_outputs(message: str):
    return (
        message,
        "",
        "",
        [],
        [],
        "",
        "",
        "",
        "",
        [],
        [],
        "",
    )


def _ask_stream(question: str, doc_ids: list[str], mode: str, top_k: int):
    if not question.strip():
        yield _empty_outputs("Please enter a question.")
        return

    doc_ids = doc_ids or None

    if mode == "Side-by-side":
        # Run the multimodal side with streaming, then run baseline blocking.
        multimodal_payload: AnswerPayload | None = None
        partial_mm = ""
        for partial, payload, debug in query_stream(
            question, mode="multimodal", doc_ids=doc_ids, top_k=top_k
        ):
            partial_mm = partial
            if payload is not None:
                multimodal_payload = payload
            yield (
                partial_mm,
                "_Waiting for completion…_",
                "",
                [],
                [],
                "",
                f"### Multimodal answer (streaming)\n\n{partial_mm}",
                "_Running baseline…_",
                f"```json\n{json.dumps(debug, indent=2)}\n```\n",
                [],
                [],
                "",
            )

        baseline_payload = run_query(
            question, mode="baseline", doc_ids=doc_ids, top_k=top_k
        )
        primary = multimodal_payload or baseline_payload

        evidence = primary.evidence
        yield (
            primary.answer,
            _format_citations(primary),
            _format_evidence_md(evidence),
            _evidence_table_rows(evidence),
            _gallery_items(evidence),
            _table_evidence_markdown(evidence),
            f"### Multimodal answer\n\n{multimodal_payload.answer if multimodal_payload else '_Not run._'}\n\n"
            f"**Citations**\n{_format_citations(multimodal_payload)}",
            f"### Text-only baseline answer\n\n{baseline_payload.answer}\n\n"
            f"**Citations**\n{_format_citations(baseline_payload)}",
            _debug_markdown(multimodal_payload, baseline_payload),
            _gallery_items(evidence),
            _evidence_table_rows(evidence),
            _validation_line(primary),
        )
        return

    actual_mode = "multimodal" if mode == "Multimodal" else "baseline"
    final_payload: AnswerPayload | None = None
    partial = ""
    for chunk, payload, debug in query_stream(
        question, mode=actual_mode, doc_ids=doc_ids, top_k=top_k
    ):
        partial = chunk
        if payload is not None:
            final_payload = payload
        if final_payload is None:
            yield (
                partial,
                "_Waiting for completion…_",
                "",
                [],
                [],
                "",
                f"### {actual_mode.capitalize()} (streaming)\n\n{partial}"
                if actual_mode == "multimodal"
                else "_Not run._",
                "_Not run._"
                if actual_mode == "multimodal"
                else f"### Baseline (streaming)\n\n{partial}",
                f"```json\n{json.dumps(debug, indent=2)}\n```",
                [],
                [],
                "",
            )

    if final_payload is None:
        yield _empty_outputs("Generation failed.")
        return

    evidence = final_payload.evidence
    multimodal = final_payload if actual_mode == "multimodal" else None
    baseline = final_payload if actual_mode == "baseline" else None
    yield (
        final_payload.answer,
        _format_citations(final_payload),
        _format_evidence_md(evidence),
        _evidence_table_rows(evidence),
        _gallery_items(evidence),
        _table_evidence_markdown(evidence),
        f"### Multimodal answer\n\n{multimodal.answer}\n\n**Citations**\n{_format_citations(multimodal)}"
        if multimodal
        else "_Not run._",
        f"### Text-only baseline answer\n\n{baseline.answer}\n\n**Citations**\n{_format_citations(baseline)}"
        if baseline
        else "_Not run._",
        _debug_markdown(multimodal, baseline),
        _gallery_items(evidence),
        _evidence_table_rows(evidence),
        _validation_line(final_payload),
    )


def _debug_markdown(multimodal, baseline) -> str:
    parts = []
    for label, p in [("multimodal", multimodal), ("baseline", baseline)]:
        if not p:
            continue
        parts.append(f"### {label}\n```json\n{json.dumps(p.debug, indent=2)}\n```")
    return "\n\n".join(parts) or "_No debug info._"


# --------------------------------------------------------------------------- #
# In-UI ingestion & upload
# --------------------------------------------------------------------------- #


def _run_full_rebuild():
    yield (
        "⏳ Rebuilding index from corpus/… (this can take a few minutes on first run)",
        gr.skip(),
    )
    try:
        if reset_client is not None:
            reset_client()
        report = rebuild_index()
        choices = _doc_choices()
        default_ids = [c[1] for c in choices]
        status = (
            f"✅ Rebuilt. {len(report.documents)} docs, "
            f"{report.multimodal_chunks} multimodal chunks, "
            f"{report.baseline_chunks} baseline chunks, "
            f"{report.elapsed_seconds:.1f}s."
        )
        yield (
            status,
            gr.CheckboxGroup(choices=choices, value=default_ids),
        )
    except Exception as e:  # noqa: BLE001
        logger.exception("Rebuild failed")
        yield (f"❌ Rebuild failed: {e}", gr.skip())


def _ingest_uploaded(files):
    if not files:
        yield "No file selected.", gr.CheckboxGroup()
        return
    if ingest_single_pdf is None or not docling_available():
        yield (
            "❌ Docling is not installed. Run `uv sync` and relaunch.",
            gr.CheckboxGroup(),
        )
        return
    try:
        total = 0
        for file_obj in files:
            src = Path(getattr(file_obj, "name", str(file_obj)))
            if not src.exists() or src.suffix.lower() != ".pdf":
                continue
            dest = UPLOAD_DIR / src.name
            if src.resolve() != dest.resolve():
                shutil.copy2(src, dest)
            yield f"⏳ Ingesting {dest.name}…", gr.CheckboxGroup()
            summary = ingest_single_pdf(dest)
            total += 1
            yield (
                f"✅ {summary.title} — {summary.page_count}p, {summary.chunk_count} chunks",
                gr.CheckboxGroup(),
            )
        choices = _doc_choices()
        ids = [c[1] for c in choices]
        yield (
            f"✅ Ingested {total} file(s). Corpus now has {len(choices)} document(s).",
            gr.CheckboxGroup(choices=choices, value=ids),
        )
    except Exception as e:  # noqa: BLE001
        logger.exception("Upload ingest failed")
        yield f"❌ Ingest failed: {e}", gr.CheckboxGroup()


# --------------------------------------------------------------------------- #
# Build UI
# --------------------------------------------------------------------------- #


def build_ui() -> gr.Blocks:
    docs = _doc_choices()
    default_doc_ids = [d[1] for d in docs]
    warnings: list[str] = []
    if SETTINGS.embed_backend != "openai":
        warnings.append(
            "Embedding backend: **hash** (offline mode). Set `OPENAI_API_KEY` "
            "and press **Run Ingestion & Indexing** for real semantic retrieval."
        )
    if not SETTINGS.openai_api_key:
        warnings.append(
            "Answer generation: **stub mode** — no `OPENAI_API_KEY` is set, "
            "so answers will surface the top retrieved evidence instead of a "
            "synthesized response."
        )
    if not docling_available():
        warnings.append(
            "**Docling is not installed.** Ingestion is disabled. "
            "Run `uv sync` to install it (querying an already-built index still works)."
        )

    with gr.Blocks(title="Unified Multimodal PDF RAG with Docling") as demo:
        with gr.Column(elem_classes="app-shell"):
            gr.Markdown(
                """
<div class="eyebrow">Chart, Table, and Diagram Grounding</div>

# Unified Multimodal PDF RAG with Docling

Compare a **text-only baseline** against a **multimodal Docling pipeline** that
preserves tables, figures, and page visuals — with real LLM token streaming
and in-UI ingestion.
""".strip()
            )
            for w in warnings:
                gr.Markdown(f'<div class="warn">⚠ {w}</div>')

            status = gr.Markdown("Ready.")

            with gr.Row(equal_height=False):
                with gr.Column(scale=1, min_width=320):
                    gr.Markdown("### 1. Corpus")
                    doc_select = gr.CheckboxGroup(
                        choices=docs,
                        value=default_doc_ids,
                        label="Pick documents to query",
                    )

                    with gr.Accordion("📂 Upload / rebuild", open=False):
                        upload = gr.File(
                            label="Upload PDF(s)",
                            file_count="multiple",
                            file_types=[".pdf"],
                        )
                        upload_btn = gr.Button("Ingest uploaded PDF(s)")
                        rebuild_btn = gr.Button(
                            "Run full Ingestion & Indexing on corpus/"
                        )

                    gr.Markdown("### 2. Sample questions")
                    samples = gr.Radio(choices=SAMPLE_QUESTIONS, label=None, value=None)

                    gr.Markdown("### 3. Mode")
                    mode = gr.Radio(
                        choices=["Multimodal", "Baseline (text-only)", "Side-by-side"],
                        value="Side-by-side",
                        label=None,
                    )
                    top_k = gr.Slider(1, 12, value=6, step=1, label="Top-k retrieval")

                    gr.Markdown("### 4. Ask")
                    question = gr.Textbox(
                        label="Question",
                        placeholder="e.g. What architectural change does the residual block introduce?",
                        lines=3,
                    )
                    ask_btn = gr.Button("Ask", variant="primary")

                with gr.Column(scale=2):
                    with gr.Tabs():
                        with gr.Tab("Answer"):
                            answer_md = gr.Markdown(label="Answer")
                            validation_md = gr.Markdown()
                            gr.Markdown("**Citations**")
                            citations_md = gr.Markdown()
                        with gr.Tab("Evidence"):
                            evidence_md = gr.Markdown()
                            evidence_table = gr.Dataframe(
                                headers=[
                                    "#",
                                    "type",
                                    "doc",
                                    "page",
                                    "score",
                                    "snippet",
                                ],
                                datatype=[
                                    "number",
                                    "str",
                                    "str",
                                    "number",
                                    "str",
                                    "str",
                                ],
                                interactive=False,
                                wrap=True,
                                label="Retrieved chunks",
                            )
                        with gr.Tab("Page Preview"):
                            page_gallery = gr.Gallery(
                                label="Cited pages & crops",
                                columns=2,
                                height=520,
                            )
                        with gr.Tab("Tables"):
                            table_md = gr.Markdown()
                        with gr.Tab("Baseline Comparison"):
                            with gr.Row():
                                with gr.Column():
                                    multimodal_md = gr.Markdown()
                                with gr.Column():
                                    baseline_md = gr.Markdown()
                        with gr.Tab("Debug"):
                            debug_md = gr.Markdown()
                            debug_table = gr.Dataframe(
                                headers=[
                                    "#",
                                    "type",
                                    "doc",
                                    "page",
                                    "score",
                                    "snippet",
                                ],
                                datatype=[
                                    "number",
                                    "str",
                                    "str",
                                    "number",
                                    "str",
                                    "str",
                                ],
                                interactive=False,
                                wrap=True,
                                label="Final ranked evidence",
                            )

                with gr.Column(scale=1, min_width=280):
                    gr.Markdown("### Evidence preview")
                    preview_gallery = gr.Gallery(
                        label="Top cited pages & crops",
                        columns=1,
                        height=420,
                    )

        samples.change(lambda s: s or "", inputs=samples, outputs=question)

        ask_btn.click(
            _ask_stream,
            inputs=[question, doc_select, mode, top_k],
            outputs=[
                answer_md,
                citations_md,
                evidence_md,
                evidence_table,
                page_gallery,
                table_md,
                multimodal_md,
                baseline_md,
                debug_md,
                preview_gallery,
                debug_table,
                validation_md,
            ],
        )

        rebuild_btn.click(_run_full_rebuild, outputs=[status, doc_select])
        upload_btn.click(
            _ingest_uploaded,
            inputs=[upload],
            outputs=[status, doc_select],
        )

    return demo


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    demo = build_ui()
    port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
    demo.queue().launch(
        server_name="127.0.0.1",
        server_port=port,
        show_error=True,
        css=CSS,
    )


if __name__ == "__main__":
    main()
