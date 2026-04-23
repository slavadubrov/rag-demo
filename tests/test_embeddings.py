"""Deterministic hash embedding behavior."""

from __future__ import annotations

import numpy as np

from rag_demo.embeddings import embed_query, embed_texts


def test_hash_embed_is_deterministic():
    v1 = embed_query("transformer attention layer")
    v2 = embed_query("transformer attention layer")
    np.testing.assert_array_equal(v1, v2)


def test_hash_embed_is_normalized():
    v = embed_query("hello world")
    assert abs(float(np.linalg.norm(v)) - 1.0) < 1e-3


def test_hash_embed_different_inputs_produce_different_vectors():
    v1 = embed_query("attention mechanism")
    v2 = embed_query("convolutional networks")
    # Vectors should differ meaningfully
    diff = float(np.linalg.norm(v1 - v2))
    assert diff > 0.5


def test_embed_texts_shape():
    arr = embed_texts(["a", "b", "c"])
    assert arr.shape[0] == 3
    assert arr.shape[1] > 0
