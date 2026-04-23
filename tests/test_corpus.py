"""Corpus downloader edge cases (no network access required)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag_demo.corpus import (
    DownloadResult,
    download_corpus,
    load_manifest,
    render_report,
)


@pytest.fixture
def mini_manifest(tmp_path: Path) -> Path:
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "created_at": "2026-04-23",
                "documents": [
                    {
                        "filename": "needs_manual.pdf",
                        "source_page": "https://example.com/manual",
                        "direct_pdf_url": None,
                        "download_note": "manual download required",
                    }
                ],
            }
        )
    )
    return path


def test_load_manifest_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_manifest(tmp_path / "nope.json")


def test_manual_required_entries_are_reported(mini_manifest, tmp_path):
    target = tmp_path / "out"
    results = download_corpus(manifest_path=mini_manifest, target_dir=target)
    assert len(results) == 1
    assert results[0].status == "manual_required"
    assert results[0].note == "manual download required"


def test_render_report_groups_by_status():
    results = [
        DownloadResult(filename="a.pdf", status="downloaded", size_bytes=1024),
        DownloadResult(filename="b.pdf", status="manual_required"),
        DownloadResult(filename="c.pdf", status="downloaded", size_bytes=2048),
    ]
    md = render_report(results)
    assert "downloaded" in md
    assert "manual_required" in md
    assert "a.pdf" in md
