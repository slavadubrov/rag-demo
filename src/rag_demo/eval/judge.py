"""Optional LLM judge: scores a multimodal/baseline answer pair."""

from __future__ import annotations

import json
import logging

from ..config import SETTINGS
from ..schema import AnswerPayload

logger = logging.getLogger(__name__)


JUDGE_SYSTEM = """You are a strict evaluator comparing two RAG answers.
Score each answer 1-5 on:
- groundedness (does it stay tied to the cited evidence?),
- completeness (does it actually answer the question?),
- visual reasoning (does it correctly use figure/table evidence when relevant?).

Then return JSON only, with this exact schema:
{
  "multimodal_score": <int 1-5>,
  "baseline_score": <int 1-5>,
  "winner": "multimodal" | "baseline" | "tie",
  "rationale": "<= 60 words>"
}
"""


def judge_pair(
    question: str,
    multimodal: AnswerPayload,
    baseline: AnswerPayload,
) -> dict | None:
    """Return the judge's verdict dict, or None if unavailable."""
    if not SETTINGS.openai_api_key:
        return None
    from openai import OpenAI

    client = OpenAI(api_key=SETTINGS.openai_api_key)
    prompt = (
        f"Question:\n{question}\n\n"
        f"Multimodal answer:\n{multimodal.answer}\n\n"
        f"Baseline answer:\n{baseline.answer}\n"
    )
    try:
        resp = client.chat.completions.create(
            model=SETTINGS.answer_model,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        return json.loads(resp.choices[0].message.content or "{}")
    except Exception as e:  # noqa: BLE001
        logger.warning("LLM judge failed: %s", e)
        return None
