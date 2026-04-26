# Unified Multimodal PDF RAG with Docling and Gradio

A local-first demo that shows why chart, table, and diagram-heavy PDFs should
not be treated as plain text. The pipeline ingests a curated PDF corpus with
[Docling](https://github.com/DS4SD/docling), preserves both structured layout
and page-level visuals *and cropped figures/tables*, retrieves typed evidence
(sections, tables, figures, captions, page fallbacks) from a local Qdrant
index, and answers questions with a vision-capable OpenAI model using **real
token-level streaming**. A baseline text-only path — with its own separate
index — runs in parallel for side-by-side comparison.


## Highlights

- **Real Docling pipeline on real PDFs** — typed chunks with section hierarchy,
  `parent_id`, `bbox`, and Docling-exported figure/table crops (not just
  full-page PNGs).
- **Two separate Qdrant collections** — the multimodal vs text-only comparison
  is *structurally* fair, not a mode flag on the same chunks.
- **Intent-routed retrieval** — table / figure / mixed / text queries trigger
  layered passes with different `element_type` filters, followed by a
  page-neighborhood expansion via direct `scroll` (no wasted vector search).
- **Vision-capable answers with real streaming** — the OpenAI SDK streams
  tokens into the Gradio Markdown in real time; figure/table **crops** are
  preferred over full-page images when available.
- **Post-generation citation validation** — `[doc_id, p.X]` citations emitted
  by the model are parsed and cross-checked against the retrieved evidence
  pages; the UI surfaces a check badge.
- **In-UI ingestion & upload** — upload your own PDFs or trigger a full
  rebuild from the Gradio tab, no terminal required. Existing documents are
  preserved; only the uploaded doc's rows are refreshed.
- **Offline-safe** — missing `OPENAI_API_KEY` degrades gracefully to
  deterministic hash embeddings + an evidence-only stub answer.
- **Pytest suite** — runs fully offline (no OpenAI required) via a synthetic
  PyMuPDF-generated PDF; Docling itself isn't touched by the tests.
- **Polished UI** — custom CSS, Iowan Old Style typography, tabbed layout
  (Answer / Evidence / Page Preview / Tables / Baseline Comparison / Debug).

## Documentation

A full guide with SVG diagrams lives in [`docs/`](docs/README.md) — start
with the [Overview](docs/01-overview.md) and [Architecture](docs/02-architecture.md)
chapters for the big picture, or jump straight to
[Retrieval](docs/04-retrieval.md) / [Answering](docs/05-answering.md) /
[Evaluation](docs/06-evaluation.md) for a specific subsystem. The roadmap
chapter ([`08-roadmap.md`](docs/08-roadmap.md)) tracks known gaps and
candidate next steps.

## Quickstart

```bash
# 1. Install (Docling + Qdrant + Gradio + OpenAI SDK + PyMuPDF, etc.)
uv sync                            # or: make install

# 2. Optional: configure your OpenAI key
cp .env.example .env
# then edit .env and set OPENAI_API_KEY=sk-...

# 3. Download the curated corpus
uv run rag-demo download           # or: make download

# 4. Build the index
uv run rag-demo rebuild            # or: make ingest

# 5. Launch the Gradio UI
uv run rag-demo app                # or: make app
```

The UI opens at http://127.0.0.1:7860 (override with `GRADIO_SERVER_PORT`).

### One-shot CLI

```bash
uv run rag-demo query "What does Figure 1 in the ResNet paper show?"
uv run rag-demo query "Compare ResNet-50 vs ResNet-152 layers" --mode baseline
uv run rag-demo list
```

### Evaluation

```bash
uv run rag-demo eval --max-questions 1       # extraction + routing + citation grounding
uv run rag-demo eval --judge                 # also runs LLM judge (extra OpenAI cost)
```

Reports land in `data/eval/eval.md` and `data/eval/eval.json`.

## Running tests

Tests build a synthetic PDF via PyMuPDF and do **not** require Docling or an
OpenAI key:

```bash
uv run pytest -q                   # or: make test
```

## Layout

```
corpus/                    # Source PDFs (curated + uploaded)
  uploads/                 # PDFs ingested via the UI
data/
  pages/<doc_id>/          # Rendered page PNGs (one per PDF page)
  crops/<doc_id>/          # Docling figure/table crops (figure_00001.png, table_00001.png, …)
  docling/<doc_id>.json    # Saved DoclingDocument
  chunks/<doc_id>.jsonl    # Typed retrieval units
  qdrant/                  # Local Qdrant store (multimodal + baseline collections)
  eval/                    # Evaluation reports
src/rag_demo/
  config.py                # Paths, environment, settings
  schema.py                # Pydantic models
  ingest.py                # Docling + PyMuPDF ingestion + typed chunking + baseline chunker
  embeddings.py            # OpenAI embeddings (hash-backend fallback)
  index.py                 # Local Qdrant (rebuild, single-doc ingest, search, scroll)
  retrieve.py              # Intent-routed layered retrieval + page-neighborhood expansion
  answer.py                # Vision-capable grounded answer + real token streaming
  baseline.py              # query() / query_both() / query_stream()
  app.py                   # Gradio Blocks UI (tabs, streaming, upload, in-UI ingest)
  corpus.py                # Manifest-driven downloader
  cli.py                   # rebuild / list / query / app / download / eval
  eval/
    __init__.py            # Public eval surface
    extraction.py          # Docling extraction stats
    judge.py               # Optional LLM judge
    retrieval_eval.py      # Routing + citation eval, aggregation, rendering
files/
  corpus_manifest.json     # Curated 10-PDF corpus + benchmark questions
  corpus_manifest.md       # Human-readable version of the manifest
tests/                     # Offline pytest suite
scripts/
  download_corpus.py       # Standalone downloader wrapper
```

## Configuration

Environment variables (all optional):

| Variable | Default | Notes |
|----------|---------|-------|
| `OPENAI_API_KEY` | — | Required for real embeddings/answers. Hash-backend + stub answers used otherwise. |
| `EMBED_MODEL` | `text-embedding-3-small` | OpenAI embedding model. |
| `EMBED_DIM` | `1536` | Must match the model. |
| `ANSWER_MODEL` | `gpt-4o-mini` | Vision-capable chat model. |
| `EMBED_BACKEND` | auto (`openai` if key set) | Force `hash` for deterministic offline runs. |
| `PAGE_RENDER_DPI` | `150` | Page-image rendering DPI. |
| `MAX_PAGES_PER_DOC` | `60` | Safety cap during ingestion. |
| `GRADIO_SERVER_PORT` | `7860` | UI port. |
| `MULTIMODAL_COLLECTION` | `multimodal_chunks` | Qdrant collection name. |
| `BASELINE_COLLECTION` | `baseline_chunks` | Qdrant collection name. |

## Modes

- **Multimodal (default)** — typed Docling chunks, figure/table crops or page
  images attached to the answer model.
- **Baseline** — naive PyMuPDF page text chunks, no visuals, no typed filtering,
  stored in a *separate* Qdrant collection.
- **Side-by-side** — runs both pipelines and shows them next to each other.
  The primary (multimodal) answer streams; baseline completes after.

## Attribution

The unified build picks from each of the three reference implementations:

- **`rag-demo-claude`** (base) — full Docling pipeline with typed chunks and a
  section stack; intent-routed layered retrieval; separate Qdrant collections
  for multimodal vs baseline; strict citation prompt rules and offline stub;
  robust manifest-driven corpus downloader; tab-rich UI layout with Dataframes
  for evidence; thorough multi-axis evaluator.
- **`rag-demo-gemini`** — Docling `PdfPipelineOptions(generate_picture_images=True)`
  + `item.image.pil_image.save(...)` so figure/table **crops** are persisted
  and passed to the vision model; real OpenAI streaming; the in-UI
  "Run Ingestion & Indexing" button.
- **`rag-demo-codex`** — Pytest suite shape; custom
  CSS + typography for the UI; eval module split (extraction / judge /
  retrieval-eval) to tame the single 700-LOC file.

## Notes

- Without `OPENAI_API_KEY`, the pipeline still ingests, indexes (deterministic
  hash embeddings), and renders the UI. The Answer tab will surface the top
  retrieved evidence rather than a synthesized response.
- The bundled manifest lists 10 public PDFs (arXiv papers, OECD/WHO/Federal
  Reserve/World Bank reports, NASA handbooks), each with a direct PDF URL for
  scripted download.
