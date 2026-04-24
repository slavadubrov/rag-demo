"""Retrieval + generation eval: does the pipeline surface + cite the right evidence?"""

from __future__ import annotations

import json
import logging
import re
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..baseline import query as run_query
from ..config import SETTINGS
from ..corpus import DEFAULT_MANIFEST
from ..corpus import load_manifest as load_corpus_manifest
from ..ingest import load_manifest as load_doc_manifest
from ..retrieve import infer_query_kind
from ..schema import AnswerPayload, Evidence
from .extraction import ExtractionStats, compute_extraction_stats
from .judge import judge_pair

logger = logging.getLogger(__name__)


CITATION_PAGE_RE = re.compile(r"p\.?\s*(\d+)", re.I)
_INSUFFICIENT_RE = re.compile(r"insufficient|not enough evidence|cannot.*answer", re.I)


@dataclass
class QuestionResult:
    filename: str
    doc_id: str | None
    question: str
    expected_kind: str

    mm_inferred_kind: str | None = None
    mm_evidence_types: list[str] = field(default_factory=list)
    mm_evidence_pages: list[int] = field(default_factory=list)
    mm_has_expected_type: bool = False
    mm_answer_chars: int = 0
    mm_has_citation: bool = False
    mm_cited_pages: list[int] = field(default_factory=list)
    mm_text_cited_pages: list[int] = field(default_factory=list)
    mm_citation_page_match: bool = False
    mm_text_citation_page_match: bool | None = None
    mm_insufficient_evidence: bool = False
    mm_seconds: float = 0.0

    bl_evidence_types: list[str] = field(default_factory=list)
    bl_evidence_pages: list[int] = field(default_factory=list)
    bl_has_expected_type: bool = False
    bl_answer_chars: int = 0
    bl_has_citation: bool = False
    bl_cited_pages: list[int] = field(default_factory=list)
    bl_text_cited_pages: list[int] = field(default_factory=list)
    bl_citation_page_match: bool = False
    bl_text_citation_page_match: bool | None = None
    bl_insufficient_evidence: bool = False
    bl_seconds: float = 0.0

    judge_multimodal_score: float | None = None
    judge_baseline_score: float | None = None
    judge_winner: str | None = None
    judge_rationale: str | None = None

    error: str | None = None


@dataclass
class EvalReport:
    started_at: str
    settings: dict
    extraction: list[ExtractionStats]
    questions: list[QuestionResult]
    aggregates: dict
    elapsed_seconds: float


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _expected_kinds_for_question(question: str) -> set[str]:
    q = question.lower()
    expected: set[str] = set()
    if re.search(
        r"\b(table|column|row|cell|appendix entry|attainment|labour-market|values?)\b",
        q,
    ):
        expected.add("table_chunk")
    if re.search(
        r"\b(chart|figure|fig\.|diagram|plot|graph|architecture|block diagram|"
        r"visual|legend|axis|menu flow|model diagram|trend|gap|topic)\b",
        q,
    ):
        expected.add("figure_chunk")
    if "caption" in q:
        expected.add("caption_chunk")
    if not expected:
        expected.add("section_chunk")
    return expected


def _kind_label(expected: set[str]) -> str:
    if "table_chunk" in expected and "figure_chunk" in expected:
        return "mixed"
    if "table_chunk" in expected:
        return "table"
    if "figure_chunk" in expected:
        return "figure"
    if "caption_chunk" in expected:
        return "figure"
    return "text"


def _has_expected(evidence_types: list[str], expected: set[str]) -> bool:
    return any(t in expected for t in evidence_types)


def _insufficient_evidence(answer: str) -> bool:
    return bool(_INSUFFICIENT_RE.search(answer or ""))


def _payload_cited_pages(payload: AnswerPayload) -> list[int]:
    seen: set[int] = set()
    out: list[int] = []
    for c in payload.citations:
        p = c.get("page_num")
        if p in seen or p is None:
            continue
        seen.add(p)
        out.append(p)
    return out


def _text_cited_pages(answer: str) -> list[int]:
    seen: set[int] = set()
    out: list[int] = []
    for m in CITATION_PAGE_RE.finditer(answer or ""):
        try:
            p = int(m.group(1))
        except ValueError:
            continue
        if p in seen:
            continue
        seen.add(p)
        out.append(p)
    return out


def _retrieved_pages(evidence: list[Evidence]) -> list[int]:
    return list({e.chunk.page_num for e in evidence})


# --------------------------------------------------------------------------- #
# per-question
# --------------------------------------------------------------------------- #


def evaluate_question(
    filename: str,
    doc_id: str | None,
    question: str,
    top_k: int,
    use_judge: bool,
) -> QuestionResult:
    expected = _expected_kinds_for_question(question)
    expected_kind = _kind_label(expected)
    result = QuestionResult(
        filename=filename, doc_id=doc_id, question=question, expected_kind=expected_kind
    )
    if not doc_id:
        result.error = "document not ingested yet"
        return result

    doc_filter = [doc_id]

    try:
        t0 = time.time()
        mm = run_query(question, mode="multimodal", doc_ids=doc_filter, top_k=top_k)
        result.mm_seconds = round(time.time() - t0, 2)
        mm_types = [e.chunk.element_type for e in mm.evidence]
        result.mm_inferred_kind = (
            mm.debug.get("kind") if mm.debug else infer_query_kind(question)
        )
        result.mm_evidence_types = mm_types
        result.mm_evidence_pages = _retrieved_pages(mm.evidence)
        result.mm_has_expected_type = _has_expected(mm_types, expected)
        result.mm_answer_chars = len(mm.answer or "")
        result.mm_cited_pages = _payload_cited_pages(mm)
        result.mm_text_cited_pages = _text_cited_pages(mm.answer)
        result.mm_has_citation = bool(
            result.mm_text_cited_pages or result.mm_cited_pages
        )
        result.mm_citation_page_match = any(
            p in result.mm_evidence_pages for p in result.mm_cited_pages
        )
        result.mm_text_citation_page_match = (
            any(p in result.mm_evidence_pages for p in result.mm_text_cited_pages)
            if result.mm_text_cited_pages
            else None
        )
        result.mm_insufficient_evidence = _insufficient_evidence(mm.answer)
    except Exception as e:  # noqa: BLE001
        logger.exception("multimodal query failed for %s", question)
        result.error = f"multimodal: {e}"
        return result

    try:
        t0 = time.time()
        bl = run_query(question, mode="baseline", doc_ids=doc_filter, top_k=top_k)
        result.bl_seconds = round(time.time() - t0, 2)
        bl_types = [e.chunk.element_type for e in bl.evidence]
        result.bl_evidence_types = bl_types
        result.bl_evidence_pages = _retrieved_pages(bl.evidence)
        result.bl_has_expected_type = _has_expected(bl_types, expected)
        result.bl_answer_chars = len(bl.answer or "")
        result.bl_cited_pages = _payload_cited_pages(bl)
        result.bl_text_cited_pages = _text_cited_pages(bl.answer)
        result.bl_has_citation = bool(
            result.bl_text_cited_pages or result.bl_cited_pages
        )
        result.bl_citation_page_match = any(
            p in result.bl_evidence_pages for p in result.bl_cited_pages
        )
        result.bl_text_citation_page_match = (
            any(p in result.bl_evidence_pages for p in result.bl_text_cited_pages)
            if result.bl_text_cited_pages
            else None
        )
        result.bl_insufficient_evidence = _insufficient_evidence(bl.answer)
    except Exception as e:  # noqa: BLE001
        logger.exception("baseline query failed for %s", question)
        result.error = f"baseline: {e}"
        return result

    if use_judge:
        verdict = judge_pair(question, mm, bl)
        if verdict:
            try:
                result.judge_multimodal_score = float(verdict.get("multimodal_score"))
                result.judge_baseline_score = float(verdict.get("baseline_score"))
                result.judge_winner = verdict.get("winner")
                result.judge_rationale = verdict.get("rationale")
            except (TypeError, ValueError):
                logger.warning("LLM judge returned bad scores: %s", verdict)

    return result


def aggregate(results: list[QuestionResult]) -> dict:
    if not results:
        return {}
    runnable = [r for r in results if not r.error]
    if not runnable:
        return {"runnable": 0, "errors": len(results)}

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return round(statistics.mean(xs), 3) if xs else None

    agg: dict = {
        "runnable": len(runnable),
        "errors": len(results) - len(runnable),
        "multimodal": {
            "expected_type_hit_rate": mean(
                int(r.mm_has_expected_type) for r in runnable
            ),
            "citation_present_rate": mean(int(r.mm_has_citation) for r in runnable),
            "model_emitted_citation_rate": mean(
                int(bool(r.mm_text_cited_pages)) for r in runnable
            ),
            "insufficient_evidence_rate": mean(
                int(r.mm_insufficient_evidence) for r in runnable
            ),
            "mean_answer_chars": mean(r.mm_answer_chars for r in runnable),
            "mean_seconds": mean(r.mm_seconds for r in runnable),
        },
        "baseline": {
            "expected_type_hit_rate": mean(
                int(r.bl_has_expected_type) for r in runnable
            ),
            "citation_present_rate": mean(int(r.bl_has_citation) for r in runnable),
            "model_emitted_citation_rate": mean(
                int(bool(r.bl_text_cited_pages)) for r in runnable
            ),
            "insufficient_evidence_rate": mean(
                int(r.bl_insufficient_evidence) for r in runnable
            ),
            "mean_answer_chars": mean(r.bl_answer_chars for r in runnable),
            "mean_seconds": mean(r.bl_seconds for r in runnable),
        },
    }
    judge_results = [r for r in runnable if r.judge_winner is not None]
    if judge_results:
        agg["judge"] = {
            "n": len(judge_results),
            "multimodal_mean_score": mean(
                r.judge_multimodal_score for r in judge_results
            ),
            "baseline_mean_score": mean(r.judge_baseline_score for r in judge_results),
            "multimodal_wins": sum(
                1 for r in judge_results if r.judge_winner == "multimodal"
            ),
            "baseline_wins": sum(
                1 for r in judge_results if r.judge_winner == "baseline"
            ),
            "ties": sum(1 for r in judge_results if r.judge_winner == "tie"),
        }
    return agg


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #


def run_eval(
    corpus_manifest_path: Path = DEFAULT_MANIFEST,
    only_filenames: list[str] | None = None,
    max_questions_per_doc: int | None = None,
    top_k: int = 6,
    use_judge: bool = False,
) -> EvalReport:
    t0 = time.time()
    corpus_manifest = load_corpus_manifest(corpus_manifest_path)
    doc_manifest = load_doc_manifest()
    extraction = compute_extraction_stats(corpus_manifest, doc_manifest)
    extraction_index = {x.filename: x for x in extraction}

    documents = corpus_manifest.get("documents", [])
    if only_filenames:
        keep = set(only_filenames)
        documents = [d for d in documents if d["filename"] in keep]

    questions: list[QuestionResult] = []
    for entry in documents:
        filename = entry["filename"]
        ext = extraction_index.get(filename)
        doc_id = ext.doc_id if ext else None
        qs = entry.get("benchmark_questions", [])
        if max_questions_per_doc:
            qs = qs[:max_questions_per_doc]
        for q in qs:
            logger.info("eval: %s :: %s", filename, q[:60])
            qr = evaluate_question(
                filename, doc_id, q, top_k=top_k, use_judge=use_judge
            )
            questions.append(qr)

    return EvalReport(
        started_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        settings={
            "embed_backend": SETTINGS.embed_backend,
            "embed_model": SETTINGS.embed_model,
            "answer_model": SETTINGS.answer_model,
            "top_k": top_k,
            "use_judge": use_judge,
            "openai_key_set": bool(SETTINGS.openai_api_key),
        },
        extraction=extraction,
        questions=questions,
        aggregates=aggregate(questions),
        elapsed_seconds=round(time.time() - t0, 2),
    )


# --------------------------------------------------------------------------- #
# render
# --------------------------------------------------------------------------- #


def report_to_dict(report: EvalReport) -> dict:
    return {
        "started_at": report.started_at,
        "settings": report.settings,
        "extraction": [asdict(x) for x in report.extraction],
        "questions": [asdict(q) for q in report.questions],
        "aggregates": report.aggregates,
        "elapsed_seconds": report.elapsed_seconds,
    }


def render_markdown(report: EvalReport) -> str:
    lines: list[str] = []
    lines.append(f"# Evaluation report — {report.started_at}")
    lines.append("")
    lines.append(f"_elapsed: {report.elapsed_seconds:.1f}s_")
    lines.append("")
    lines.append("## Settings")
    lines.append("```json")
    lines.append(json.dumps(report.settings, indent=2))
    lines.append("```")
    lines.append("")

    lines.append("## Extraction stats (per document)")
    lines.append("")
    lines.append(
        "| filename | ingested | pages | chunks | section | table | figure | caption | fallback | pages w/ visual | notes |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for x in report.extraction:
        ct = x.chunk_type_counts
        lines.append(
            "| `{f}` | {ing} | {pg} | {ch} | {s} | {t} | {fg} | {c} | {fb} | {v} | {n} |".format(
                f=x.filename,
                ing="yes" if x.ingested else "—",
                pg=x.page_count,
                ch=x.chunk_count,
                s=ct.get("section_chunk", 0),
                t=ct.get("table_chunk", 0),
                fg=ct.get("figure_chunk", 0),
                c=ct.get("caption_chunk", 0),
                fb=ct.get("page_fallback_chunk", 0),
                v=x.pages_with_typed_visual,
                n="; ".join(x.notes) or "",
            )
        )
    lines.append("")

    lines.append("## Aggregates")
    lines.append("```json")
    lines.append(json.dumps(report.aggregates, indent=2))
    lines.append("```")
    lines.append("")

    lines.append("## Per-question results")
    lines.append("")
    lines.append(
        "| doc | question | expected | mm type hit | mm cite | mm cite-grounded | bl type hit | bl cite | bl cite-grounded | judge |"
    )
    lines.append("|---|---|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|")

    def _cell(flag):
        if flag is None:
            return "—"
        return "yes" if flag else "no"

    for q in report.questions:
        if q.error:
            lines.append(
                f"| `{q.filename}` | {q.question[:80]} | — | error | — | — | error | — | — | `{q.error}` |"
            )
            continue
        judge_cell = ""
        if q.judge_winner:
            judge_cell = f"{q.judge_winner} ({q.judge_multimodal_score}/{q.judge_baseline_score})"
        lines.append(
            "| `{f}` | {q} | {ek} | {mh} | {mc} | {mp} | {bh} | {bc} | {bp} | {jg} |".format(
                f=q.filename,
                q=q.question.replace("|", "\\|")[:80],
                ek=q.expected_kind,
                mh=_cell(q.mm_has_expected_type),
                mc="yes" if q.mm_text_cited_pages else "no",
                mp=_cell(q.mm_text_citation_page_match),
                bh=_cell(q.bl_has_expected_type),
                bc="yes" if q.bl_text_cited_pages else "no",
                bp=_cell(q.bl_text_citation_page_match),
                jg=judge_cell,
            )
        )
    lines.append("")
    return "\n".join(lines)


def write_reports(report: EvalReport, json_path: Path, md_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report_to_dict(report), indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
