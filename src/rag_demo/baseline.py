"""Wrappers that run retrieval + answer in one shot (both modes)."""

from __future__ import annotations

from typing import Generator

from .answer import generate_answer, stream_answer
from .retrieve import retrieve
from .schema import AnswerPayload


def query(
    question: str,
    mode: str,
    doc_ids: list[str] | None,
    top_k: int = 6,
) -> AnswerPayload:
    evidence, debug = retrieve(question, mode=mode, doc_ids=doc_ids, top_k=top_k)
    payload = generate_answer(question, mode=mode, evidence=evidence)
    payload.debug.update(debug)
    return payload


def query_both(
    question: str,
    doc_ids: list[str] | None,
    top_k: int = 6,
) -> tuple[AnswerPayload, AnswerPayload]:
    multimodal = query(question, mode="multimodal", doc_ids=doc_ids, top_k=top_k)
    baseline = query(question, mode="baseline", doc_ids=doc_ids, top_k=top_k)
    return multimodal, baseline


def query_stream(
    question: str,
    mode: str,
    doc_ids: list[str] | None,
    top_k: int = 6,
) -> Generator[tuple[str, AnswerPayload | None, dict], None, None]:
    """Streaming variant. Yields (partial_text, final_payload_or_None, retrieval_debug)."""
    evidence, debug = retrieve(question, mode=mode, doc_ids=doc_ids, top_k=top_k)
    for partial, payload in stream_answer(question, mode=mode, evidence=evidence):
        if payload is not None:
            payload.debug.update(debug)
        yield partial, payload, debug
