"""Index management behavior that doesn't require Docling or live Qdrant."""

from __future__ import annotations

import json


def test_ingest_single_pdf_replaces_existing_manifest_entry_for_same_upload_path(
    tmp_data_dir,
    monkeypatch,
):
    from rag_demo import index
    from rag_demo.config import SETTINGS
    from rag_demo.schema import Chunk, DocumentSummary

    pdf_path = tmp_data_dir / "corpus" / "uploads" / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    old_doc_id = "paper_oldhash"
    index.MANIFEST_PATH.write_text(
        json.dumps(
            {
                old_doc_id: {
                    "doc_id": old_doc_id,
                    "title": "Old upload",
                    "pdf_path": str(pdf_path),
                    "page_count": 1,
                    "chunk_count": 1,
                }
            }
        ),
        encoding="utf-8",
    )

    new_doc_id = "paper_newhash"
    mm_chunks = [
        Chunk(
            chunk_id=f"{new_doc_id}::sec::00001",
            doc_id=new_doc_id,
            page_num=1,
            element_type="section_chunk",
            text="fresh multimodal chunk",
        )
    ]
    bl_chunks = [
        Chunk(
            chunk_id=f"{new_doc_id}::base::00001",
            doc_id=new_doc_id,
            page_num=1,
            element_type="page_fallback_chunk",
            text="fresh baseline chunk",
        )
    ]

    deleted_rows: list[tuple[str, str]] = []
    deleted_artifacts: list[str] = []
    upserts: list[tuple[str, int]] = []

    monkeypatch.setattr(index, "reset_client", lambda: None)
    monkeypatch.setattr(
        index,
        "ingest_one",
        lambda path: (
            DocumentSummary(
                doc_id=new_doc_id,
                title="New upload",
                pdf_path=str(path),
                page_count=2,
            ),
            ["/tmp/page_0001.png", "/tmp/page_0002.png"],
            object(),
        ),
    )
    monkeypatch.setattr(
        index, "chunk_document", lambda doc, doc_id, page_imgs: mm_chunks
    )
    monkeypatch.setattr(index, "save_chunks", lambda doc_id, chunks: None)
    monkeypatch.setattr(
        index,
        "baseline_chunks_from_pdf",
        lambda path, doc_id, max_pages=None: bl_chunks,
    )
    monkeypatch.setattr(index, "_ensure_collection", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        index,
        "_delete_by_doc_id",
        lambda collection, doc_id: deleted_rows.append((collection, doc_id)),
    )
    monkeypatch.setattr(
        index,
        "_upsert_chunks",
        lambda collection, chunks: upserts.append((collection, len(chunks))),
    )
    monkeypatch.setattr(
        index,
        "_delete_local_artifacts",
        lambda doc_id: deleted_artifacts.append(doc_id),
    )

    summary = index.ingest_single_pdf(pdf_path)

    manifest = json.loads(index.MANIFEST_PATH.read_text(encoding="utf-8"))
    assert summary.doc_id == new_doc_id
    assert old_doc_id not in manifest
    assert new_doc_id in manifest
    assert manifest[new_doc_id]["pdf_path"] == str(pdf_path)
    assert manifest[new_doc_id]["docling_json"].endswith(f"{new_doc_id}.json")
    assert deleted_artifacts == [old_doc_id]
    assert (SETTINGS.multimodal_collection, old_doc_id) in deleted_rows
    assert (SETTINGS.baseline_collection, old_doc_id) in deleted_rows
    assert (SETTINGS.multimodal_collection, new_doc_id) in deleted_rows
    assert (SETTINGS.baseline_collection, new_doc_id) in deleted_rows
    assert upserts == [
        (SETTINGS.multimodal_collection, len(mm_chunks)),
        (SETTINGS.baseline_collection, len(bl_chunks)),
    ]
