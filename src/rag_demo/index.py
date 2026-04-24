"""Local Qdrant vector index: build, search, and evidence lookup.

Two collections live side by side:

- ``multimodal_chunks`` — typed Docling chunks (tables, figures, captions,
  section prose, page fallbacks) that power the multimodal answer path.
- ``baseline_chunks`` — naive PyMuPDF page-text chunks that power the
  text-only baseline. Keeping it separate is what makes the multimodal-vs-text
  comparison structurally fair.
"""

from __future__ import annotations

import atexit
import json
import logging
import shutil
import time
import uuid
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

from .config import (
    CHUNKS_DIR,
    CORPUS_DIR,
    CROP_IMAGE_DIR,
    DOCLING_DIR,
    MANIFEST_PATH,
    PAGE_IMAGE_DIR,
    QDRANT_DIR,
    SETTINGS,
)
from .embeddings import embed_texts
from .ingest import (
    baseline_chunks_from_pdf,
    chunk_document,
    ingest_corpus,
    ingest_one,
    load_docling,
    load_manifest,
    save_chunks,
)
from .schema import Chunk, DocumentSummary, Evidence, IngestReport

logger = logging.getLogger(__name__)


_client: QdrantClient | None = None


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(path=str(QDRANT_DIR))
    return _client


def reset_client() -> None:
    """Close the local Qdrant client (it holds a directory lock)."""
    global _client
    if _client is not None:
        try:
            _client.close()
        except Exception:  # noqa: BLE001
            pass
        _client = None


# Close the local Qdrant client before the interpreter starts tearing down
# modules. Without this, qdrant-client's own __del__ may fire after sys.meta_path
# is already None and log a noisy ImportError during shutdown.
atexit.register(reset_client)


def _ensure_collection(name: str, dim: int, *, recreate: bool) -> None:
    client = get_client()
    exists = client.collection_exists(name)
    if recreate and exists:
        client.delete_collection(name)
        exists = False
    if not exists:
        client.create_collection(
            collection_name=name,
            vectors_config=rest.VectorParams(size=dim, distance=rest.Distance.COSINE),
        )


def _chunk_to_payload(c: Chunk) -> dict:
    d = c.model_dump()
    if d.get("bbox") is None:
        d.pop("bbox", None)
    if d.get("table_html") and len(d["table_html"]) > 4000:
        d["table_html"] = d["table_html"][:4000]
    return d


def _upsert_chunks(collection: str, chunks: list[Chunk]) -> None:
    if not chunks:
        return
    client = get_client()
    vectors = embed_texts([c.text for c in chunks])
    points = [
        rest.PointStruct(
            id=str(uuid.uuid4()),
            vector=vec.tolist(),
            payload=_chunk_to_payload(c),
        )
        for c, vec in zip(chunks, vectors)
    ]
    BATCH = 128
    for i in range(0, len(points), BATCH):
        client.upsert(collection_name=collection, points=points[i : i + BATCH])


def _delete_by_doc_id(collection: str, doc_id: str) -> None:
    client = get_client()
    if not client.collection_exists(collection):
        return
    client.delete(
        collection_name=collection,
        points_selector=rest.FilterSelector(
            filter=rest.Filter(
                must=[
                    rest.FieldCondition(
                        key="doc_id", match=rest.MatchValue(value=doc_id)
                    )
                ]
            )
        ),
    )


def _same_pdf_path(left: str | Path | None, right: Path) -> bool:
    if not left:
        return False
    try:
        return Path(left).resolve() == right.resolve()
    except OSError:
        return Path(left) == right


def _doc_ids_for_pdf_path(manifest: dict, pdf_path: Path) -> list[str]:
    return [
        doc_id
        for doc_id, info in manifest.items()
        if _same_pdf_path(info.get("pdf_path"), pdf_path)
    ]


def _delete_local_artifacts(doc_id: str) -> None:
    shutil.rmtree(PAGE_IMAGE_DIR / doc_id, ignore_errors=True)
    shutil.rmtree(CROP_IMAGE_DIR / doc_id, ignore_errors=True)
    (DOCLING_DIR / f"{doc_id}.json").unlink(missing_ok=True)
    (CHUNKS_DIR / f"{doc_id}.jsonl").unlink(missing_ok=True)


# --------------------------------------------------------------------------- #
# Rebuild paths
# --------------------------------------------------------------------------- #


def rebuild_index() -> IngestReport:
    """Re-ingest every PDF in the corpus and rebuild both vector indexes."""
    t0 = time.time()
    reset_client()

    summaries, manifest = ingest_corpus(CORPUS_DIR)

    multimodal_total = 0
    baseline_total = 0
    multimodal_chunks: list[Chunk] = []
    baseline_chunks: list[Chunk] = []

    for s in summaries:
        info = manifest[s.doc_id]
        page_imgs = info["page_images"]
        doc = load_docling(s.doc_id)

        m_chunks = chunk_document(doc, s.doc_id, page_imgs)
        save_chunks(s.doc_id, m_chunks)
        s.chunk_count = len(m_chunks)
        multimodal_total += len(m_chunks)
        multimodal_chunks.extend(m_chunks)

        b_chunks = baseline_chunks_from_pdf(
            Path(s.pdf_path),
            s.doc_id,
            max_pages=SETTINGS.max_pages_per_doc,
        )
        baseline_total += len(b_chunks)
        baseline_chunks.extend(b_chunks)

    _ensure_collection(
        SETTINGS.multimodal_collection, SETTINGS.embed_dim, recreate=True
    )
    _ensure_collection(SETTINGS.baseline_collection, SETTINGS.embed_dim, recreate=True)

    _upsert_chunks(SETTINGS.multimodal_collection, multimodal_chunks)
    _upsert_chunks(SETTINGS.baseline_collection, baseline_chunks)

    for s in summaries:
        manifest[s.doc_id]["chunk_count"] = s.chunk_count
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return IngestReport(
        documents=summaries,
        multimodal_chunks=multimodal_total,
        baseline_chunks=baseline_total,
        elapsed_seconds=round(time.time() - t0, 2),
    )


def ingest_single_pdf(pdf_path: Path) -> DocumentSummary:
    """Ingest a single PDF (e.g. uploaded via the UI) and upsert its chunks.

    Collections are created if missing, but not dropped — existing indexed docs
    stay in place and this new one is merged in. If the same file has been
    ingested before, its prior chunks are purged first.
    """
    pdf_path = Path(pdf_path)
    reset_client()
    manifest = load_manifest()
    stale_doc_ids = _doc_ids_for_pdf_path(manifest, pdf_path)

    summary, page_imgs, doc = ingest_one(pdf_path)

    m_chunks = chunk_document(doc, summary.doc_id, page_imgs)
    save_chunks(summary.doc_id, m_chunks)
    summary.chunk_count = len(m_chunks)

    b_chunks = baseline_chunks_from_pdf(
        pdf_path,
        summary.doc_id,
        max_pages=SETTINGS.max_pages_per_doc,
    )

    _ensure_collection(
        SETTINGS.multimodal_collection, SETTINGS.embed_dim, recreate=False
    )
    _ensure_collection(SETTINGS.baseline_collection, SETTINGS.embed_dim, recreate=False)

    # Refresh this upload path (same filename can produce a new doc_id when the
    # contents change, so purge all previous rows tied to this PDF path first).
    refresh_doc_ids = set(stale_doc_ids)
    refresh_doc_ids.add(summary.doc_id)
    for doc_id in refresh_doc_ids:
        _delete_by_doc_id(SETTINGS.multimodal_collection, doc_id)
        _delete_by_doc_id(SETTINGS.baseline_collection, doc_id)

    _upsert_chunks(SETTINGS.multimodal_collection, m_chunks)
    _upsert_chunks(SETTINGS.baseline_collection, b_chunks)

    # Merge into manifest
    for doc_id in stale_doc_ids:
        if doc_id != summary.doc_id:
            _delete_local_artifacts(doc_id)
        manifest.pop(doc_id, None)
    manifest[summary.doc_id] = {
        **summary.model_dump(),
        "page_images": page_imgs,
        "docling_json": str(DOCLING_DIR / f"{summary.doc_id}.json"),
        "chunk_count": summary.chunk_count,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return summary


# --------------------------------------------------------------------------- #
# Query-side
# --------------------------------------------------------------------------- #


def list_documents() -> list[DocumentSummary]:
    manifest = load_manifest()
    return [
        DocumentSummary(
            doc_id=doc_id,
            title=info.get("title", doc_id),
            pdf_path=info.get("pdf_path", ""),
            page_count=info.get("page_count", 0),
            chunk_count=info.get("chunk_count", 0),
        )
        for doc_id, info in manifest.items()
    ]


def search(
    collection: str,
    query_vector,
    limit: int = 12,
    doc_ids: list[str] | None = None,
    element_types: list[str] | None = None,
) -> list[Evidence]:
    client = get_client()
    if not client.collection_exists(collection):
        return []
    must = []
    if doc_ids:
        must.append(rest.FieldCondition(key="doc_id", match=rest.MatchAny(any=doc_ids)))
    if element_types:
        must.append(
            rest.FieldCondition(
                key="element_type", match=rest.MatchAny(any=element_types)
            )
        )
    flt = rest.Filter(must=must) if must else None

    res = client.query_points(
        collection_name=collection,
        query=query_vector.tolist(),
        limit=limit,
        with_payload=True,
        query_filter=flt,
    )
    out: list[Evidence] = []
    for p in res.points:
        chunk = Chunk.model_validate(p.payload)
        out.append(
            Evidence(
                chunk=chunk,
                score=float(p.score),
                page_image_path=chunk.image_ref,
                crop_image_path=chunk.crop_ref,
            )
        )
    return out


def scroll_page_fallback(
    collection: str, doc_id: str, page_num: int, limit: int = 3
) -> list[Evidence]:
    """Pull page-fallback chunks for a specific (doc, page) without re-embedding.

    Replaces the "embed 'page N context'" shortcut in the original retrieve
    path — direct scroll is both faster and semantically exact.
    """
    client = get_client()
    if not client.collection_exists(collection):
        return []
    res = client.scroll(
        collection_name=collection,
        scroll_filter=rest.Filter(
            must=[
                rest.FieldCondition(key="doc_id", match=rest.MatchValue(value=doc_id)),
                rest.FieldCondition(
                    key="page_num", match=rest.MatchValue(value=page_num)
                ),
                rest.FieldCondition(
                    key="element_type",
                    match=rest.MatchValue(value="page_fallback_chunk"),
                ),
            ]
        ),
        limit=limit,
        with_payload=True,
    )
    points = res[0]
    out: list[Evidence] = []
    for p in points:
        chunk = Chunk.model_validate(p.payload)
        out.append(
            Evidence(
                chunk=chunk,
                score=0.5,
                page_image_path=chunk.image_ref,
                crop_image_path=chunk.crop_ref,
            )
        )
    return out


def get_evidence(chunk_id: str) -> Evidence | None:
    client = get_client()
    if not client.collection_exists(SETTINGS.multimodal_collection):
        return None
    res = client.scroll(
        collection_name=SETTINGS.multimodal_collection,
        scroll_filter=rest.Filter(
            must=[
                rest.FieldCondition(
                    key="chunk_id", match=rest.MatchValue(value=chunk_id)
                )
            ]
        ),
        limit=1,
        with_payload=True,
    )
    points = res[0]
    if not points:
        return None
    chunk = Chunk.model_validate(points[0].payload)
    return Evidence(
        chunk=chunk,
        score=1.0,
        page_image_path=chunk.image_ref,
        crop_image_path=chunk.crop_ref,
    )
