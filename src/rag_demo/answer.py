"""Grounded answer generation with real token-level streaming.

Strong Claude-style grounding rules in the prompt (strict ``[doc_id, p.X]``
citations + "insufficient evidence" fallback) combined with Gemini's real
token streaming via the OpenAI SDK. Multimodal mode prefers Docling figure/
table **crops** over full-page PNGs when available, falling back to the page
image otherwise.

The module exposes two entry points:

- ``generate_answer`` — blocking. Returns a fully-formed ``AnswerPayload``.
- ``stream_answer`` — generator. Yields partial text + a final payload.
"""

from __future__ import annotations

import base64
import logging
import re
from pathlib import Path
from typing import Generator

from .config import SETTINGS
from .schema import AnswerPayload, Evidence

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are a careful research assistant answering questions about PDFs.
Rules:
- Use ONLY the supplied evidence (text, tables, page images). Do not draw from outside knowledge.
- Every factual claim must end with a citation in the form [doc_id, p.X].
- If the evidence is insufficient, reply: "Evidence is insufficient to answer confidently." and explain what is missing.
- Tables and figures may carry the answer. Inspect images carefully when text alone is ambiguous.
- Keep answers concise (under 200 words unless the question requires depth).
"""


CITATION_RE = re.compile(r"\[([a-z0-9_\-./]+)\s*,\s*p\.?\s*(\d+)\]", re.I)


# --------------------------------------------------------------------------- #
# Evidence prep
# --------------------------------------------------------------------------- #


def _format_evidence_text(evidence: list[Evidence]) -> str:
    blocks = []
    for i, e in enumerate(evidence, 1):
        c = e.chunk
        head = (
            f"[{i}] doc={c.doc_id} page={c.page_num} type={c.element_type}"
            f"{' section=' + ' / '.join(c.section_path) if c.section_path else ''}"
            f" chunk_id={c.chunk_id}"
        )
        body = c.text.strip()
        if c.element_type == "table_chunk" and c.table_markdown:
            cap = (c.extra.get("caption") or "").strip()
            body = (cap + "\n\n" if cap else "") + c.table_markdown
        blocks.append(f"{head}\n{body}")
    return "\n\n---\n\n".join(blocks)


def _image_path_for(e: Evidence) -> str | None:
    """Prefer the Docling crop; fall back to the full page render."""
    for candidate in (e.crop_image_path, e.chunk.crop_ref, e.page_image_path, e.chunk.image_ref):
        if candidate and Path(candidate).exists():
            return candidate
    return None


def _select_image_evidence(evidence: list[Evidence], max_images: int = 3) -> list[tuple[Evidence, str]]:
    """Pick up to ``max_images`` distinct images from the top evidence.

    Priority:
      1. figure/table/caption chunks (deduped by doc+page to avoid sending the
         same page image twice).
      2. any remaining evidence with an image, as a last-ditch fallback.

    Returns a list of ``(evidence, image_path)`` pairs.
    """
    seen: set[tuple[str, int]] = set()
    picks: list[tuple[Evidence, str]] = []

    for e in evidence:
        if e.chunk.element_type not in ("figure_chunk", "table_chunk", "caption_chunk"):
            continue
        img = _image_path_for(e)
        if not img:
            continue
        key = (e.chunk.doc_id, e.chunk.page_num)
        if key in seen:
            continue
        seen.add(key)
        picks.append((e, img))
        if len(picks) >= max_images:
            return picks

    for e in evidence:
        img = _image_path_for(e)
        if not img:
            continue
        key = (e.chunk.doc_id, e.chunk.page_num)
        if key in seen:
            continue
        seen.add(key)
        picks.append((e, img))
        if len(picks) >= max_images:
            break
    return picks


def _b64_image(path: str) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode("ascii")


def _build_messages(
    question: str,
    mode: str,
    evidence: list[Evidence],
    use_images: bool,
) -> tuple[list[dict], list[tuple[Evidence, str]]]:
    text_block = _format_evidence_text(evidence)
    user_content: list[dict] = [
        {
            "type": "text",
            "text": (
                f"Question: {question}\n\n"
                f"Mode: {mode}\n\n"
                f"Text evidence:\n\n{text_block or '(none)'}"
            ),
        }
    ]
    image_evidence: list[tuple[Evidence, str]] = []
    if use_images and mode == "multimodal":
        image_evidence = _select_image_evidence(evidence)
        for e, img in image_evidence:
            label = "Crop" if (e.crop_image_path or e.chunk.crop_ref) == img else "Page"
            user_content.append(
                {
                    "type": "text",
                    "text": (
                        f"{label} image from {e.chunk.doc_id}, page {e.chunk.page_num}, "
                        f"element {e.chunk.element_type}:"
                    ),
                }
            )
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{_b64_image(img)}",
                        "detail": "high",
                    },
                }
            )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    return messages, image_evidence


def _structured_citations(evidence: list[Evidence]) -> list[dict]:
    return [
        {
            "doc_id": e.chunk.doc_id,
            "page_num": e.chunk.page_num,
            "chunk_id": e.chunk.chunk_id,
            "type": e.chunk.element_type,
        }
        for e in evidence[:6]
    ]


def _validate_citations(answer_text: str, evidence: list[Evidence]) -> dict:
    """Post-parse ``[doc, p.X]`` citations and flag any that point outside retrieval."""
    retrieved_pages: dict[str, set[int]] = {}
    for e in evidence:
        retrieved_pages.setdefault(e.chunk.doc_id, set()).add(e.chunk.page_num)

    matches = CITATION_RE.findall(answer_text or "")
    valid: list[dict] = []
    invalid: list[dict] = []
    for doc_id, page in matches:
        try:
            p = int(page)
        except ValueError:
            continue
        ok = any(
            (doc_id == d) or (doc_id in d) or (d in doc_id)
            for d in retrieved_pages.keys()
        ) and any(p in retrieved_pages[d] for d in retrieved_pages if doc_id in d or d in doc_id)
        record = {"doc_id": doc_id, "page_num": p}
        (valid if ok else invalid).append(record)
    return {
        "cited": valid + invalid,
        "valid_count": len(valid),
        "invalid_count": len(invalid),
    }


# --------------------------------------------------------------------------- #
# Stub (offline) answer
# --------------------------------------------------------------------------- #


def _stub_answer(question: str, mode: str, evidence: list[Evidence]) -> AnswerPayload:
    if not evidence:
        msg = (
            "[OFFLINE MODE] No OpenAI key configured and no evidence retrieved. "
            "Set OPENAI_API_KEY in .env and rebuild the index for real answers."
        )
    else:
        top = evidence[0]
        excerpt = top.chunk.text.strip().replace("\n", " ")[:400]
        msg = (
            "[OFFLINE MODE — no OPENAI_API_KEY set]\n\n"
            f"Top retrieved evidence (page {top.chunk.page_num}, "
            f"type={top.chunk.element_type}):\n\n> {excerpt}\n\n"
            f"[{top.chunk.doc_id}, p.{top.chunk.page_num}]"
        )
    return AnswerPayload(
        question=question,
        mode=mode,
        answer=msg,
        citations=_structured_citations(evidence),
        evidence=evidence,
    )


# --------------------------------------------------------------------------- #
# Public entry points
# --------------------------------------------------------------------------- #


def generate_answer(
    question: str,
    mode: str,
    evidence: list[Evidence],
    use_images: bool = True,
) -> AnswerPayload:
    """Blocking answer synthesis (used by the CLI)."""
    if not SETTINGS.openai_api_key:
        return _stub_answer(question, mode, evidence)

    from openai import OpenAI

    client = OpenAI(api_key=SETTINGS.openai_api_key)
    messages, image_evidence = _build_messages(question, mode, evidence, use_images)

    try:
        completion = client.chat.completions.create(
            model=SETTINGS.answer_model,
            messages=messages,
            temperature=0.1,
        )
        answer_text = (completion.choices[0].message.content or "").strip()
    except Exception as e:  # noqa: BLE001
        logger.exception("OpenAI call failed")
        answer_text = f"[Generation failed: {e}]"

    return AnswerPayload(
        question=question,
        mode=mode,
        answer=answer_text,
        citations=_structured_citations(evidence),
        evidence=evidence,
        debug={
            "images_attached": len(image_evidence),
            "citation_validation": _validate_citations(answer_text, evidence),
        },
    )


def stream_answer(
    question: str,
    mode: str,
    evidence: list[Evidence],
    use_images: bool = True,
) -> Generator[tuple[str, AnswerPayload | None], None, None]:
    """Real token streaming.

    Yields ``(partial_text, None)`` as tokens arrive, then one final
    ``(full_text, AnswerPayload)`` tuple with citations attached. The caller
    can simply keep the first element to render progressively.
    """
    if not SETTINGS.openai_api_key:
        payload = _stub_answer(question, mode, evidence)
        yield payload.answer, payload
        return

    from openai import OpenAI

    client = OpenAI(api_key=SETTINGS.openai_api_key)
    messages, image_evidence = _build_messages(question, mode, evidence, use_images)

    full = ""
    try:
        resp = client.chat.completions.create(
            model=SETTINGS.answer_model,
            messages=messages,
            temperature=0.1,
            stream=True,
        )
        for chunk in resp:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                full += delta
                yield full, None
    except Exception as e:  # noqa: BLE001
        logger.exception("OpenAI streaming call failed")
        full = f"[Generation failed: {e}]"
        yield full, None

    payload = AnswerPayload(
        question=question,
        mode=mode,
        answer=full.strip(),
        citations=_structured_citations(evidence),
        evidence=evidence,
        debug={
            "images_attached": len(image_evidence),
            "citation_validation": _validate_citations(full, evidence),
        },
    )
    yield full, payload
