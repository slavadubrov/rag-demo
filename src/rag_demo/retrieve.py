"""Hybrid retrieval: query intent → typed search → page-neighborhood expansion.

Uses Claude's layered-pass + page-neighborhood design, but replaces the
"embed 'page N context'" shortcut with a direct `scroll` on the Qdrant
collection (index.scroll_page_fallback). That avoids spending an embedding call
(or, worse, running a pointless vector search) just to fetch a fallback chunk
we already know we want.
"""

from __future__ import annotations

import logging
import re
from collections import OrderedDict

from .config import SETTINGS
from .embeddings import embed_query
from .index import scroll_page_fallback, search
from .schema import Evidence

logger = logging.getLogger(__name__)


_TABLE_HINTS = re.compile(
    r"\b(table|row|column|cell|value|number|score|metric|benchmark|dataset|param(eter)?s?|"
    r"accuracy|precision|recall|f1|bleu|rouge|tokens?|layers?|how many|how much)\b",
    re.I,
)
_FIGURE_HINTS = re.compile(
    r"\b(figure|fig\.|diagram|chart|plot|graph|arrow|architecture|block|illustration|"
    r"shape|color|legend|axis|y-axis|x-axis|trend|curve|image|picture|visual|show)\b",
    re.I,
)


def infer_query_kind(question: str) -> str:
    has_table = bool(_TABLE_HINTS.search(question))
    has_figure = bool(_FIGURE_HINTS.search(question))
    if has_table and has_figure:
        return "mixed"
    if has_table:
        return "table"
    if has_figure:
        return "figure"
    return "text"


def _allowed_types_for(kind: str) -> list[list[str]]:
    """Ordered passes of ``element_type`` filters (later passes are wider)."""
    if kind == "table":
        return [
            ["table_chunk", "caption_chunk"],
            ["section_chunk", "page_fallback_chunk"],
        ]
    if kind == "figure":
        return [
            ["figure_chunk", "caption_chunk"],
            ["page_fallback_chunk", "section_chunk"],
        ]
    if kind == "mixed":
        return [
            ["table_chunk", "figure_chunk", "caption_chunk"],
            ["section_chunk", "page_fallback_chunk"],
        ]
    return [
        ["section_chunk", "caption_chunk"],
        ["table_chunk", "figure_chunk"],
        ["page_fallback_chunk"],
    ]


def _dedup(items: list[Evidence]) -> list[Evidence]:
    seen: OrderedDict[str, Evidence] = OrderedDict()
    for e in items:
        key = e.chunk.chunk_id
        if key in seen:
            if e.score > seen[key].score:
                seen[key] = e
            continue
        seen[key] = e
    return list(seen.values())


def _expand_page_neighborhood(
    collection: str,
    evidence: list[Evidence],
    max_extra: int = 4,
) -> list[Evidence]:
    """For the top-ranked results, also pull the page-fallback chunk for that page."""
    if not evidence:
        return evidence
    pages = {(e.chunk.doc_id, e.chunk.page_num) for e in evidence[:3]}
    extras: list[Evidence] = []
    for doc_id, page in pages:
        for r in scroll_page_fallback(collection, doc_id, page, limit=1):
            r.score = 0.45  # demoted so it never outranks typed evidence
            extras.append(r)
    return evidence + extras[:max_extra]


def retrieve(
    question: str,
    mode: str,
    doc_ids: list[str] | None,
    top_k: int = 6,
) -> tuple[list[Evidence], dict]:
    """Return ranked evidence plus a debug trace."""
    kind = infer_query_kind(question)
    qvec = embed_query(question)

    if mode == "baseline":
        results = search(
            SETTINGS.baseline_collection,
            qvec,
            limit=top_k,
            doc_ids=doc_ids,
        )
        return results[:top_k], {
            "kind": kind,
            "passes": [{"types": "all", "n": len(results)}],
        }

    passes_meta: list[dict] = []
    aggregated: list[Evidence] = []
    for types in _allowed_types_for(kind):
        results = search(
            SETTINGS.multimodal_collection,
            qvec,
            limit=top_k,
            doc_ids=doc_ids,
            element_types=types,
        )
        passes_meta.append({"types": types, "n": len(results)})
        aggregated.extend(results)

    aggregated = sorted(_dedup(aggregated), key=lambda e: -e.score)
    aggregated = aggregated[: max(top_k, 6)]
    aggregated = _expand_page_neighborhood(SETTINGS.multimodal_collection, aggregated)
    aggregated = sorted(_dedup(aggregated), key=lambda e: -e.score)[:top_k]

    debug = {
        "kind": kind,
        "passes": passes_meta,
        "final": [
            {
                "chunk_id": e.chunk.chunk_id,
                "type": e.chunk.element_type,
                "page": e.chunk.page_num,
                "score": round(e.score, 3),
            }
            for e in aggregated
        ],
    }
    return aggregated, debug
