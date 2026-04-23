"""Unified multimodal PDF RAG demo.

Combines the strongest ideas from three independent implementations:

- `rag-demo-claude` — full Docling + typed chunks + intent-routed retrieval
  + Qdrant (multimodal + baseline collections) + vision answers + offline mode
- `rag-demo-gemini` — real Docling figure/table image crops + real LLM
  streaming + in-UI ingestion button
- `rag-demo-codex` — pytest suite + optional docling extra + polished UI theme
  + modular evaluation split
"""

__all__ = [
    "config",
    "schema",
    "embeddings",
    "index",
    "retrieve",
    "answer",
    "baseline",
]
