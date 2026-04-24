"""Ingestion + typed chunking in a single module.

This merges the best of all three reference implementations:

- **Claude**: full Docling pipeline, section-stack typed chunking, page fallbacks,
  PyMuPDF-based page rendering, manifest persistence, baseline chunk index.
- **Gemini**: `PdfPipelineOptions(generate_picture_images=True)` + per-element
  `item.image.pil_image.save(...)` so figures/tables have their own crops.
- **Codex**: `max_pages_per_doc` cap + clean manifest separation so rebuilds
  are deterministic.

Docling is an *optional* dependency (see `pyproject.toml` extras). If it is
missing, ingestion fails with a clear message and the rest of the package
(retrieval, answer, UI) still imports and runs against a previously-built index.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import defaultdict
from pathlib import Path

import fitz  # PyMuPDF

from .config import (
    CHUNKS_DIR,
    CROP_IMAGE_DIR,
    DOCLING_DIR,
    MANIFEST_PATH,
    PAGE_IMAGE_DIR,
    SETTINGS,
)
from .schema import BBox, Chunk, DocumentSummary

logger = logging.getLogger(__name__)

# Group consecutive text items in the same section into a single chunk
# until reaching this character budget.
SECTION_CHUNK_TARGET_CHARS = 900
SECTION_CHUNK_MAX_CHARS = 1600


# --------------------------------------------------------------------------- #
# Docling availability
# --------------------------------------------------------------------------- #


def _import_docling():
    """Import Docling lazily so the package imports cleanly without the extra."""
    try:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling_core.types.doc.document import (
            DoclingDocument,
            PictureItem,
            SectionHeaderItem,
            TableItem,
            TextItem,
        )
    except ImportError as e:  # pragma: no cover - exercised at runtime
        raise RuntimeError(
            "Docling is not installed. Run `uv sync` (or "
            "`pip install rag-demo-unified`) to install it."
        ) from e
    return {
        "InputFormat": InputFormat,
        "PdfPipelineOptions": PdfPipelineOptions,
        "DocumentConverter": DocumentConverter,
        "PdfFormatOption": PdfFormatOption,
        "DoclingDocument": DoclingDocument,
        "PictureItem": PictureItem,
        "SectionHeaderItem": SectionHeaderItem,
        "TableItem": TableItem,
        "TextItem": TextItem,
    }


def docling_available() -> bool:
    try:
        _import_docling()
        return True
    except RuntimeError:
        return False


# --------------------------------------------------------------------------- #
# Page rendering + doc_id + Docling parse
# --------------------------------------------------------------------------- #


def _doc_id(pdf_path: Path) -> str:
    h = hashlib.sha1(pdf_path.read_bytes()).hexdigest()[:8]
    stem = pdf_path.stem.replace(" ", "_").lower()
    return f"{stem}_{h}"


def _render_pages(pdf_path: Path, doc_id: str, dpi: int, max_pages: int) -> list[str]:
    """Render each PDF page to a PNG (PyMuPDF — faster & more reliable than Docling)."""
    out_dir = PAGE_IMAGE_DIR / doc_id
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    with fitz.open(pdf_path) as doc:
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        n = min(doc.page_count, max_pages)
        for i in range(n):
            page = doc.load_page(i)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            out = out_dir / f"page_{i + 1:04d}.png"
            pix.save(str(out))
            paths.append(str(out))
    return paths


def _build_converter():
    """Build a Docling converter that preserves per-page AND per-picture images.

    Grafted from rag-demo-gemini — ``generate_picture_images`` is what makes
    ``item.image.pil_image`` available for figure/table chunks.
    """
    d = _import_docling()
    options = d["PdfPipelineOptions"]()
    options.generate_page_images = True
    options.generate_picture_images = True
    # Only some Docling versions expose generate_table_images; set defensively.
    if hasattr(options, "generate_table_images"):
        options.generate_table_images = True
    return d["DocumentConverter"](
        format_options={
            d["InputFormat"].PDF: d["PdfFormatOption"](pipeline_options=options)
        }
    )


def _docling_parse(pdf_path: Path, max_pages: int):
    converter = _build_converter()
    result = converter.convert(str(pdf_path), page_range=(1, max_pages))
    return result.document


def _save_docling(doc, doc_id: str) -> Path:
    out = DOCLING_DIR / f"{doc_id}.json"
    out.write_text(json.dumps(doc.export_to_dict(), default=str), encoding="utf-8")
    return out


def _title_from_docling(doc, fallback: str) -> str:
    try:
        for item, _ in doc.iterate_items():
            label = getattr(item, "label", None)
            label_v = getattr(label, "value", str(label)) if label is not None else ""
            text = (getattr(item, "text", "") or "").strip()
            if not text:
                continue
            if label_v == "title":
                return text[:200]
            if label_v == "section_header":
                return text[:200]
    except Exception:  # noqa: BLE001
        pass
    return fallback


# --------------------------------------------------------------------------- #
# Chunking helpers
# --------------------------------------------------------------------------- #


def _label_value(item) -> str:
    label = getattr(item, "label", None)
    return getattr(label, "value", str(label)) if label is not None else ""


def _page_no(item) -> int | None:
    prov = getattr(item, "prov", None)
    if not prov:
        return None
    return prov[0].page_no


def _bbox(item) -> BBox | None:
    prov = getattr(item, "prov", None)
    if not prov:
        return None
    bb = getattr(prov[0], "bbox", None)
    if bb is None:
        return None
    return BBox(l=float(bb.l), t=float(bb.t), r=float(bb.r), b=float(bb.b))


def _save_item_crop(item, doc_id: str, kind: str, index: int) -> str | None:
    """If Docling produced a PIL image for this item, save it as a PNG crop."""
    img = getattr(item, "image", None)
    if img is None:
        return None
    pil = getattr(img, "pil_image", None)
    if pil is None:
        return None
    out_dir = CROP_IMAGE_DIR / doc_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{kind}_{index:05d}.png"
    try:
        pil.save(out)
    except Exception as e:  # noqa: BLE001
        logger.warning("failed to save %s crop for %s: %s", kind, doc_id, e)
        return None
    return str(out)


def _flush_section_chunk(
    buf: list[str],
    section_path: list[str],
    page_num: int,
    doc_id: str,
    counter: dict,
) -> Chunk | None:
    text = " ".join(s.strip() for s in buf if s.strip()).strip()
    if not text:
        return None
    counter["section"] += 1
    return Chunk(
        chunk_id=f"{doc_id}::sec::{counter['section']:05d}",
        doc_id=doc_id,
        page_num=page_num,
        element_type="section_chunk",
        text=text,
        section_path=list(section_path),
    )


# --------------------------------------------------------------------------- #
# Chunk document
# --------------------------------------------------------------------------- #


def chunk_document(
    doc,
    doc_id: str,
    page_image_paths: list[str],
) -> list[Chunk]:
    """Convert a Docling document into typed retrieval units.

    Produces ``section_chunk`` / ``table_chunk`` / ``figure_chunk`` /
    ``caption_chunk`` / ``page_fallback_chunk`` instances with section
    hierarchy, bounding boxes, and — when Docling provides them — cropped
    figure/table images on `crop_ref` (in addition to the full-page render on
    `image_ref`).
    """
    d = _import_docling()
    SectionHeaderItem = d["SectionHeaderItem"]
    TableItem = d["TableItem"]
    PictureItem = d["PictureItem"]
    TextItem = d["TextItem"]

    chunks: list[Chunk] = []
    counter: dict[str, int] = defaultdict(int)

    section_stack: list[tuple[int, str]] = []
    buf: list[str] = []
    buf_page: int | None = None

    def current_section_path() -> list[str]:
        return [h for _, h in section_stack]

    def page_image(page_num: int | None) -> str | None:
        if not page_num:
            return None
        idx = page_num - 1
        if 0 <= idx < len(page_image_paths):
            return page_image_paths[idx]
        return None

    def flush() -> None:
        nonlocal buf, buf_page
        if not buf or buf_page is None:
            buf, buf_page = [], None
            return
        chunk = _flush_section_chunk(
            buf, current_section_path(), buf_page, doc_id, counter
        )
        if chunk is not None:
            chunk.image_ref = page_image(buf_page)
            chunks.append(chunk)
        buf, buf_page = [], None

    for item, _level in doc.iterate_items(with_groups=False, traverse_pictures=False):
        label = _label_value(item)
        page = _page_no(item)

        if isinstance(item, SectionHeaderItem):
            flush()
            level = max(1, getattr(item, "level", 1))
            while section_stack and section_stack[-1][0] >= level:
                section_stack.pop()
            header_text = (item.text or "").strip()
            if header_text:
                section_stack.append((level, header_text))
            continue

        if isinstance(item, TableItem):
            flush()
            counter["table"] += 1
            try:
                md = item.export_to_markdown(doc=doc)
            except Exception:  # noqa: BLE001
                md = ""
            try:
                html = item.export_to_html(doc=doc)
            except Exception:  # noqa: BLE001
                html = ""
            try:
                cap = item.caption_text(doc) or ""
            except Exception:  # noqa: BLE001
                cap = ""
            text = (
                f"Table caption: {cap}\n\n{md}".strip()
                if cap
                else md or html or f"[Table on page {page}]"
            )
            crop = _save_item_crop(item, doc_id, "table", counter["table"])
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}::tbl::{counter['table']:05d}",
                    doc_id=doc_id,
                    page_num=page or 1,
                    element_type="table_chunk",
                    text=text,
                    section_path=current_section_path(),
                    bbox=_bbox(item),
                    table_html=html or None,
                    table_markdown=md or None,
                    image_ref=page_image(page),
                    crop_ref=crop,
                    extra={"caption": cap},
                )
            )
            continue

        if isinstance(item, PictureItem):
            flush()
            counter["figure"] += 1
            try:
                cap = item.caption_text(doc) or ""
            except Exception:  # noqa: BLE001
                cap = ""
            text = cap.strip() or f"[Figure on page {page}]"
            crop = _save_item_crop(item, doc_id, "figure", counter["figure"])
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}::fig::{counter['figure']:05d}",
                    doc_id=doc_id,
                    page_num=page or 1,
                    element_type="figure_chunk",
                    text=text,
                    section_path=current_section_path(),
                    bbox=_bbox(item),
                    image_ref=page_image(page),
                    crop_ref=crop,
                    extra={"caption": cap},
                )
            )
            if cap.strip():
                counter["caption"] += 1
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc_id}::cap::{counter['caption']:05d}",
                        doc_id=doc_id,
                        page_num=page or 1,
                        element_type="caption_chunk",
                        text=cap.strip(),
                        section_path=current_section_path(),
                        parent_id=chunks[-1].chunk_id,
                        image_ref=page_image(page),
                        crop_ref=crop,
                    )
                )
            continue

        if isinstance(item, TextItem):
            text = (item.text or "").strip()
            if not text:
                continue

            if label == "caption":
                counter["caption"] += 1
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc_id}::cap::{counter['caption']:05d}",
                        doc_id=doc_id,
                        page_num=page or buf_page or 1,
                        element_type="caption_chunk",
                        text=text,
                        section_path=current_section_path(),
                        image_ref=page_image(page),
                    )
                )
                continue

            if label in ("page_header", "page_footer"):
                continue

            if buf_page is not None and page is not None and page != buf_page:
                flush()

            buf.append(text)
            buf_page = page or buf_page

            if sum(len(s) for s in buf) >= SECTION_CHUNK_TARGET_CHARS:
                flush()

    flush()

    # Page-fallback chunks (single walk — collect page text in one pass).
    page_text: dict[int, list[str]] = defaultdict(list)
    for item, _ in doc.iterate_items(with_groups=False, traverse_pictures=False):
        if isinstance(item, TextItem):
            label = _label_value(item)
            if label in ("page_header", "page_footer", "footnote"):
                continue
            t = (item.text or "").strip()
            if not t:
                continue
            p = _page_no(item)
            if p is None:
                continue
            page_text[p].append(t)

    for page_num, pieces in sorted(page_text.items()):
        text = "\n".join(pieces).strip()
        if not text:
            continue
        if len(text) > 4000:
            text = text[:4000]
        counter["page"] += 1
        chunks.append(
            Chunk(
                chunk_id=f"{doc_id}::pg::{page_num:04d}",
                doc_id=doc_id,
                page_num=page_num,
                element_type="page_fallback_chunk",
                text=text,
                section_path=[],
                image_ref=page_image(page_num),
            )
        )

    return chunks


# --------------------------------------------------------------------------- #
# Baseline chunker (PyMuPDF, no Docling)
# --------------------------------------------------------------------------- #


def baseline_chunks_from_pdf(
    pdf_path: Path,
    doc_id: str,
    max_pages: int | None = None,
) -> list[Chunk]:
    """Naive text-only chunking: PyMuPDF page text split at ~1 KB per chunk.

    ``max_pages`` should match the multimodal ingestion cap so the baseline path
    stays a fair side-by-side comparison and never emits image refs for pages we
    did not render.
    """
    page_image_dir = PAGE_IMAGE_DIR / doc_id
    chunks: list[Chunk] = []
    counter = 0
    with fitz.open(pdf_path) as doc:
        limit = doc.page_count if max_pages is None else min(doc.page_count, max_pages)
        for i in range(limit):
            page = doc.load_page(i)
            text = (page.get_text("text") or "").strip()
            if not text:
                continue
            buf, buflen = [], 0
            for para in [p.strip() for p in text.split("\n\n") if p.strip()]:
                if buflen + len(para) > 1000 and buf:
                    counter += 1
                    chunks.append(
                        Chunk(
                            chunk_id=f"{doc_id}::base::{counter:05d}",
                            doc_id=doc_id,
                            page_num=i + 1,
                            element_type="page_fallback_chunk",
                            text="\n\n".join(buf),
                            image_ref=str(page_image_dir / f"page_{i + 1:04d}.png"),
                        )
                    )
                    buf, buflen = [], 0
                buf.append(para)
                buflen += len(para)
            if buf:
                counter += 1
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc_id}::base::{counter:05d}",
                        doc_id=doc_id,
                        page_num=i + 1,
                        element_type="page_fallback_chunk",
                        text="\n\n".join(buf),
                        image_ref=str(page_image_dir / f"page_{i + 1:04d}.png"),
                    )
                )
    return chunks


# --------------------------------------------------------------------------- #
# Chunk persistence
# --------------------------------------------------------------------------- #


def save_chunks(doc_id: str, chunks: list[Chunk]) -> Path:
    out = CHUNKS_DIR / f"{doc_id}.jsonl"
    with out.open("w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(c.model_dump_json() + "\n")
    return out


def load_chunks(doc_id: str) -> list[Chunk]:
    path = CHUNKS_DIR / f"{doc_id}.jsonl"
    if not path.exists():
        return []
    return [
        Chunk.model_validate_json(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


# --------------------------------------------------------------------------- #
# Corpus-level ingestion
# --------------------------------------------------------------------------- #


def ingest_one(pdf_path: Path) -> tuple[DocumentSummary, list[str], object]:
    """Parse a single PDF: render pages, run Docling, save Docling JSON."""
    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)

    doc_id = _doc_id(pdf_path)
    logger.info("Ingesting %s -> %s", pdf_path.name, doc_id)
    page_imgs = _render_pages(
        pdf_path, doc_id, SETTINGS.page_render_dpi, SETTINGS.max_pages_per_doc
    )
    doc = _docling_parse(pdf_path, SETTINGS.max_pages_per_doc)
    _save_docling(doc, doc_id)
    title = _title_from_docling(doc, fallback=pdf_path.stem)
    summary = DocumentSummary(
        doc_id=doc_id,
        title=title,
        pdf_path=str(pdf_path),
        page_count=len(page_imgs),
    )
    return summary, page_imgs, doc


def ingest_corpus(corpus_dir: Path) -> tuple[list[DocumentSummary], dict]:
    """Parse every PDF in corpus_dir, persist Docling JSON + page images, write manifest."""
    pdfs = sorted(p for p in corpus_dir.rglob("*.pdf"))
    if not pdfs:
        raise RuntimeError(f"No PDFs found in {corpus_dir}")
    summaries: list[DocumentSummary] = []
    manifest: dict[str, dict] = {}
    for pdf in pdfs:
        t0 = time.time()
        summary, page_imgs, doc = ingest_one(pdf)
        summaries.append(summary)
        manifest[summary.doc_id] = {
            **summary.model_dump(),
            "page_images": page_imgs,
            "docling_json": str(DOCLING_DIR / f"{summary.doc_id}.json"),
            "ingested_seconds": round(time.time() - t0, 2),
        }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return summaries, manifest


def load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        return {}
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def load_docling(doc_id: str):
    d = _import_docling()
    path = DOCLING_DIR / f"{doc_id}.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    return d["DoclingDocument"].model_validate(raw)
