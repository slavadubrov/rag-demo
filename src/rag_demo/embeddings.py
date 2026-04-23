"""Embeddings: OpenAI by default, deterministic hash backend for offline development."""

from __future__ import annotations

import hashlib
import logging
import time
from typing import Sequence

import numpy as np

from .config import SETTINGS

logger = logging.getLogger(__name__)


def _hash_embed(text: str, dim: int) -> np.ndarray:
    """Deterministic pseudo-embedding from token hashes.

    Not semantically useful, but lets the rest of the pipeline run without
    network access. We include token-level signal so retrieval is at least
    keyword-overlap reasonable for smoke tests.
    """
    rng_seed = int(hashlib.sha256(text.encode("utf-8", "ignore")).hexdigest()[:16], 16)
    rng = np.random.default_rng(rng_seed)
    base = rng.standard_normal(dim).astype(np.float32) * 0.1

    for tok in set(text.lower().split()):
        h = int(hashlib.sha256(tok.encode("utf-8")).hexdigest()[:16], 16)
        base[h % dim] += 1.0
    n = np.linalg.norm(base)
    if n > 0:
        base /= n
    return base


def _openai_embed_batch(texts: Sequence[str], model: str) -> np.ndarray:
    from openai import OpenAI

    client = OpenAI(api_key=SETTINGS.openai_api_key)
    out: list[list[float]] = []
    BATCH = 64
    for i in range(0, len(texts), BATCH):
        batch = list(texts[i : i + BATCH])
        batch = [t if t.strip() else " " for t in batch]
        for attempt in range(3):
            try:
                resp = client.embeddings.create(model=model, input=batch)
                break
            except Exception as e:  # noqa: BLE001
                if attempt == 2:
                    raise
                logger.warning("embed retry %d: %s", attempt, e)
                time.sleep(2 ** attempt)
        out.extend([d.embedding for d in resp.data])
    return np.array(out, dtype=np.float32)


def embed_texts(texts: Sequence[str]) -> np.ndarray:
    """Return an (N, dim) array of embeddings for the given texts."""
    if not texts:
        return np.zeros((0, SETTINGS.embed_dim), dtype=np.float32)

    backend = SETTINGS.embed_backend
    if backend == "openai":
        if not SETTINGS.openai_api_key:
            logger.warning("OPENAI_API_KEY missing; falling back to hash embeddings")
            backend = "hash"
        else:
            return _openai_embed_batch(texts, SETTINGS.embed_model)

    return np.stack([_hash_embed(t, SETTINGS.embed_dim) for t in texts]).astype(np.float32)


def embed_query(text: str) -> np.ndarray:
    return embed_texts([text])[0]
