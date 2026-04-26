"""Download the curated PDF corpus from the manifest.

Claude's robust version: realistic UA, `.part` tempfile, `%PDF-` header
verification, manual-download handling, and a Markdown report.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import requests

from .config import CORPUS_DIR, PROJECT_ROOT

logger = logging.getLogger(__name__)

DEFAULT_MANIFEST = PROJECT_ROOT / "files" / "corpus_manifest.json"
DEFAULT_TARGET_DIR = CORPUS_DIR

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/537.36 "
    "(KHTML, like Gecko) rag-demo-unified/0.1 Safari/537.36"
)
DOWNLOAD_ATTEMPTS = 3
DOWNLOAD_TIMEOUT = (10, 120)


@dataclass
class DownloadResult:
    filename: str
    status: str  # "downloaded", "skipped_existing", "manual_required", "failed"
    size_bytes: int = 0
    url: str | None = None
    error: str | None = None
    note: str | None = None

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None and v != ""}


def load_manifest(path: Path = DEFAULT_MANIFEST) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _stream_download(
    url: str, dest: Path, timeout: tuple[int, int] = DOWNLOAD_TIMEOUT
) -> int:
    headers = {"User-Agent": USER_AGENT, "Accept": "application/pdf,*/*;q=0.5"}
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.unlink(missing_ok=True)
    with requests.get(
        url, headers=headers, stream=True, timeout=timeout, allow_redirects=True
    ) as r:
        r.raise_for_status()
        ctype = r.headers.get("Content-Type", "")
        if "pdf" not in ctype.lower() and "octet-stream" not in ctype.lower():
            logger.warning("%s returned content-type %s (may not be a PDF)", url, ctype)
        n = 0
        try:
            with tmp.open("wb") as fh:
                for chunk in r.iter_content(chunk_size=64 * 1024):
                    if not chunk:
                        continue
                    fh.write(chunk)
                    n += len(chunk)
        except Exception:
            tmp.unlink(missing_ok=True)
            raise
        tmp.replace(dest)
        return n


def _looks_like_pdf(path: Path) -> bool:
    try:
        with path.open("rb") as fh:
            return fh.read(5) == b"%PDF-"
    except OSError:
        return False


def _download_valid_pdf(url: str, dest: Path) -> int:
    last_error = "download failed"
    for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
        try:
            size = _stream_download(url, dest)
            if _looks_like_pdf(dest):
                return size
            dest.unlink(missing_ok=True)
            last_error = (
                "downloaded file is not a valid PDF "
                "(server may have returned HTML); use the source page instead"
            )
        except Exception as e:  # noqa: BLE001
            last_error = str(e)
            dest.with_suffix(dest.suffix + ".part").unlink(missing_ok=True)

        if attempt < DOWNLOAD_ATTEMPTS:
            logger.warning(
                "Download failed for %s (attempt %s/%s): %s; retrying",
                dest.name,
                attempt,
                DOWNLOAD_ATTEMPTS,
                last_error,
            )
            time.sleep(min(2**attempt, 5))

    raise RuntimeError(last_error)


def download_corpus(
    manifest_path: Path = DEFAULT_MANIFEST,
    target_dir: Path = DEFAULT_TARGET_DIR,
    overwrite: bool = False,
    only: Iterable[str] | None = None,
) -> list[DownloadResult]:
    target_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(manifest_path)
    documents = manifest.get("documents", [])
    if only:
        wanted = set(only)
        documents = [d for d in documents if d["filename"] in wanted]

    results: list[DownloadResult] = []

    for doc in documents:
        filename = doc["filename"]
        dest = target_dir / filename
        url = doc.get("direct_pdf_url")
        note = doc.get("download_note") or doc.get("source_page")

        if dest.exists() and not overwrite and _looks_like_pdf(dest):
            results.append(
                DownloadResult(
                    filename=filename,
                    status="skipped_existing",
                    size_bytes=dest.stat().st_size,
                    url=url,
                    note="already on disk; pass overwrite=True to refetch",
                )
            )
            continue
        if dest.exists() and not overwrite:
            logger.warning("%s exists but is not a valid PDF; re-downloading", filename)
            dest.unlink(missing_ok=True)

        if not url:
            results.append(
                DownloadResult(
                    filename=filename,
                    status="manual_required",
                    url=doc.get("source_page"),
                    note=note,
                )
            )
            continue

        logger.info("Downloading %s <- %s", filename, url)
        try:
            size = _download_valid_pdf(url, dest)
            results.append(
                DownloadResult(
                    filename=filename, status="downloaded", size_bytes=size, url=url
                )
            )
        except Exception as e:  # noqa: BLE001
            results.append(
                DownloadResult(
                    filename=filename, status="failed", url=url, error=str(e)
                )
            )
        time.sleep(0.5)

    return results


def render_report(results: list[DownloadResult]) -> str:
    counts: dict[str, int] = {}
    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1

    lines = ["# Corpus download report", "", "## Summary"]
    for status, n in sorted(counts.items()):
        lines.append(f"- **{status}**: {n}")
    lines.append("")
    lines.append("## Detail")
    for r in results:
        line = f"- `{r.filename}` — **{r.status}**"
        if r.size_bytes:
            line += f" ({r.size_bytes / 1_048_576:.2f} MB)"
        if r.url:
            line += f"  \n  url: <{r.url}>"
        if r.note:
            line += f"  \n  note: {r.note}"
        if r.error:
            line += f"  \n  error: `{r.error}`"
        lines.append(line)
    return "\n".join(lines) + "\n"
