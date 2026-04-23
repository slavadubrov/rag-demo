"""Evaluation harness, split into extraction / retrieval / judge modules.

Addressing the "eval.py is ~700 LOC in a single file" feedback from the
combined review.

Public surface mirrors the original Claude module so existing callers keep
working:

- ``run_eval`` — drive extraction + per-question routing/citation checks.
- ``render_markdown`` — Markdown report.
- ``report_to_dict`` — JSON-serializable dict.
- ``write_reports`` — write both to disk.
- ``DEFAULT_REPORT_DIR`` — ``data/eval/``.
"""

from __future__ import annotations

from ..config import EVAL_DIR
from .extraction import ExtractionStats, compute_extraction_stats
from .judge import JUDGE_SYSTEM, judge_pair
from .retrieval_eval import (
    EvalReport,
    QuestionResult,
    aggregate,
    evaluate_question,
    render_markdown,
    report_to_dict,
    run_eval,
    write_reports,
)

DEFAULT_REPORT_DIR = EVAL_DIR

__all__ = [
    "DEFAULT_REPORT_DIR",
    "EvalReport",
    "ExtractionStats",
    "JUDGE_SYSTEM",
    "QuestionResult",
    "aggregate",
    "compute_extraction_stats",
    "evaluate_question",
    "judge_pair",
    "render_markdown",
    "report_to_dict",
    "run_eval",
    "write_reports",
]
