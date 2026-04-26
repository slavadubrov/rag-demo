# rag-demo documentation

A guided tour of the **Unified Multimodal PDF RAG** codebase. Each chapter
stands alone but they build on one another — read them in order the first
time.

| # | Chapter | What's inside |
|---|---------|---------------|
| 1 | [Overview](01-overview.md) | Why this project exists, what it demonstrates, how the pieces fit. |
| 2 | [Architecture](02-architecture.md) | The runtime view: modules, responsibilities, control flow. |
| 3 | [Ingestion pipeline](03-ingestion.md) | How a PDF becomes typed, indexable chunks. |
| 4 | [Retrieval & routing](04-retrieval.md) | Intent inference, layered search, page-neighborhood expansion. |
| 5 | [Answering & streaming](05-answering.md) | Grounded prompt, real token streaming, citation validation. |
| 6 | [Evaluation](06-evaluation.md) | What the evaluator measures and how to read the report. |
| 7 | [Configuration & operations](07-configuration.md) | Env vars, data layout, make targets, offline mode. |

## Diagram index

All diagrams are plain SVG — they render in GitHub, in most editors, and when
embedded directly in markdown.

- [System overview](diagrams/system-overview.svg) — end-to-end flow
- [Ingestion pipeline](diagrams/ingestion-pipeline.svg) — PDF → typed chunks
- [Chunk type anatomy](diagrams/chunk-types.svg) — payload shape by class
- [Retrieval flow](diagrams/retrieval-flow.svg) — intent router + layered passes
- [Answer streaming](diagrams/answer-streaming.svg) — prompt, stream, validate
- [Evaluation flow](diagrams/evaluation-flow.svg) — extraction + routing + judge
- [Data layout](diagrams/data-layout.svg) — on-disk directory map

## Conventions

- Code references use the `path:line` form so editors can jump to them.
- "Multimodal" = the full Docling path (`multimodal_chunks` collection);
  "baseline" = the naive PyMuPDF page-text path (`baseline_chunks`
  collection). Both paths coexist in the index and are compared in the
  eval report.
- "Offline mode" means `OPENAI_API_KEY` is unset — the pipeline still ingests
  and indexes, but with deterministic hash embeddings and a stub answer.
