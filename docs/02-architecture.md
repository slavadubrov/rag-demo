# 2 · Architecture

The project is a single Python package (`rag_demo`) with tightly scoped
modules. Nothing imports anything it doesn't need, and every module has one
clear job.

## Module map

```
src/rag_demo/
├─ config.py          # Paths + environment settings (single source of truth)
├─ schema.py          # Pydantic models: Chunk, Evidence, AnswerPayload, …
├─ corpus.py          # Manifest-driven downloader
├─ ingest.py          # Docling parse + PyMuPDF render + typed chunker
├─ embeddings.py      # OpenAI + deterministic hash backend
├─ index.py           # Local Qdrant: rebuild, single-doc upsert, search, scroll
├─ retrieve.py        # Intent router + layered retrieval + page expansion
├─ answer.py          # Grounded prompt + real streaming + citation check
├─ baseline.py        # query / query_both / query_stream wrappers
├─ cli.py             # rebuild · list · query · app · download · eval
├─ app.py             # Gradio Blocks UI (tabs, streaming, upload)
└─ eval/
   ├─ extraction.py   # per-doc chunk-type stats
   ├─ retrieval_eval.py  # routing + citation grounding + aggregation
   └─ judge.py        # optional LLM judge
```

## Layering

From bottom to top — each layer only depends on the ones below it:

| Layer | Modules | Responsibility |
|-------|---------|----------------|
| L0 · Config | `config.py`, `schema.py` | Paths, settings, data types. Pure — no side effects beyond mkdir at import. |
| L1 · I/O | `corpus.py`, `embeddings.py` | Bring bytes in (PDFs, embedding API calls). |
| L2 · Indexing | `ingest.py`, `index.py` | Parse, chunk, persist, index. Knows Docling + Qdrant; nothing above does. |
| L3 · Query | `retrieve.py`, `answer.py` | Ranked evidence + grounded answer. No storage code. |
| L4 · Facade | `baseline.py` | One-shot wrappers used by the CLI, UI, and evaluator. |
| L5 · Surfaces | `cli.py`, `app.py`, `eval/*` | User-facing entry points. |

The baseline path is *not* a different layer — it's a different **collection**
at L2 and a different `mode` argument at L3+, so both modes share the L3
code.

## Control flow — a single query

```
UI / CLI
  └─ baseline.query(question, mode="multimodal")
       ├─ retrieve.retrieve(...)
       │    ├─ retrieve.infer_query_kind(question)
       │    ├─ embeddings.embed_query(question)
       │    ├─ for types in _allowed_types_for(kind):
       │    │      index.search(collection, qvec, element_types=types)
       │    ├─ _dedup + sort
       │    ├─ index.scroll_page_fallback(...)   # neighborhood expansion
       │    └─ returns (evidence, debug)
       └─ answer.generate_answer(...)  [or stream_answer(...)]
            ├─ _format_evidence_text
            ├─ _select_image_evidence
            ├─ OpenAI chat (stream=True for stream_answer)
            └─ _validate_citations
```

## Shared types (schema.py)

```python
Chunk            # one typed retrieval unit (schema.py:25)
DocumentSummary  # manifest row                   (schema.py:46)
Evidence         # chunk + score + image paths    (schema.py:54)
AnswerPayload    # answer + citations + evidence  (schema.py:61)
IngestReport     # summary of a rebuild           (schema.py:70)
```

These five models are the only cross-layer contract. Any module that touches
indexed data either emits or consumes one of them.

## Why two collections instead of one?

Early prototypes used a single collection with an `element_type` flag and
reused the same chunks in "baseline mode" by ignoring the flag. That
muddied the comparison:

- The multimodal chunks already had much better recall (they were short and
  semantically clean), so "baseline" was effectively "same chunks but filter
  differently" — not a comparison against *naive* RAG.
- It was impossible to change the baseline chunker independently.

The cost of two collections is low (double the storage, same embedding
calls — the baseline chunker re-embeds the same PDF text at ~1 KB per
chunk), and the comparison becomes meaningful: each path produces its own
chunks its own way.

## Offline determinism

The pipeline must run with **no network** so tests are fast and new
contributors can try the UI without an API key.

- `embed_backend = "hash"` produces deterministic pseudo-embeddings. Not
  semantically useful, but enough for keyword-overlap retrieval to demo the
  plumbing.
- `answer.generate_answer()` short-circuits to `_stub_answer()` and returns
  the top retrieved chunk as the "answer" with a synthetic citation.
- `tests/` generate a synthetic PDF via PyMuPDF and never import Docling.
