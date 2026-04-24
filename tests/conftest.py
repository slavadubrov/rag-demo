"""Shared pytest fixtures — force offline mode so tests never touch OpenAI."""

from __future__ import annotations

# IMPORTANT: these env tweaks must run BEFORE anything imports rag_demo.config,
# because config snapshots os.environ into a frozen Settings dataclass the
# moment it is first imported. That import happens the first time any test
# collects a rag_demo symbol, so we force the overrides at module-load time
# (conftest.py is executed before test modules).
import os as _os

_os.environ["EMBED_BACKEND"] = "hash"
_os.environ.pop("OPENAI_API_KEY", None)

import os
from pathlib import Path

import pytest


@pytest.fixture(autouse=True, scope="session")
def _offline_env():
    """Reassert hash backend / no OpenAI key for the session (belt + suspenders)."""
    os.environ["EMBED_BACKEND"] = "hash"
    os.environ.pop("OPENAI_API_KEY", None)

    # If config was already imported, refresh its snapshot.
    try:
        from rag_demo import config as _config

        _config.SETTINGS = _config.load_settings()
    except ImportError:
        pass
    yield


@pytest.fixture
def tmp_data_dir(monkeypatch, tmp_path: Path):
    """Point every project path at a fresh temp directory for test isolation."""
    from rag_demo import config

    new_data = tmp_path / "data"
    new_corpus = tmp_path / "corpus"
    new_uploads = new_corpus / "uploads"

    for p in (new_data, new_corpus, new_uploads):
        p.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(config, "DATA_DIR", new_data, raising=False)
    monkeypatch.setattr(config, "PAGE_IMAGE_DIR", new_data / "pages", raising=False)
    monkeypatch.setattr(config, "CROP_IMAGE_DIR", new_data / "crops", raising=False)
    monkeypatch.setattr(config, "DOCLING_DIR", new_data / "docling", raising=False)
    monkeypatch.setattr(config, "CHUNKS_DIR", new_data / "chunks", raising=False)
    monkeypatch.setattr(config, "QDRANT_DIR", new_data / "qdrant", raising=False)
    monkeypatch.setattr(
        config, "MANIFEST_PATH", new_data / "manifest.json", raising=False
    )
    monkeypatch.setattr(config, "CORPUS_DIR", new_corpus, raising=False)
    monkeypatch.setattr(config, "UPLOAD_DIR", new_uploads, raising=False)
    monkeypatch.setattr(config, "EVAL_DIR", new_data / "eval", raising=False)

    # Mirror into downstream modules that captured the originals at import time
    from rag_demo import index, ingest

    monkeypatch.setattr(index, "CORPUS_DIR", new_corpus, raising=False)
    monkeypatch.setattr(index, "CHUNKS_DIR", new_data / "chunks", raising=False)
    monkeypatch.setattr(index, "CROP_IMAGE_DIR", new_data / "crops", raising=False)
    monkeypatch.setattr(index, "DOCLING_DIR", new_data / "docling", raising=False)
    monkeypatch.setattr(index, "QDRANT_DIR", new_data / "qdrant", raising=False)
    monkeypatch.setattr(
        index, "MANIFEST_PATH", new_data / "manifest.json", raising=False
    )
    monkeypatch.setattr(index, "PAGE_IMAGE_DIR", new_data / "pages", raising=False)
    monkeypatch.setattr(ingest, "CHUNKS_DIR", new_data / "chunks", raising=False)
    monkeypatch.setattr(ingest, "DOCLING_DIR", new_data / "docling", raising=False)
    monkeypatch.setattr(ingest, "PAGE_IMAGE_DIR", new_data / "pages", raising=False)
    monkeypatch.setattr(ingest, "CROP_IMAGE_DIR", new_data / "crops", raising=False)
    monkeypatch.setattr(
        ingest, "MANIFEST_PATH", new_data / "manifest.json", raising=False
    )

    yield tmp_path

    # Ensure any open Qdrant client is released before teardown
    try:
        index.reset_client()
    except Exception:
        pass
