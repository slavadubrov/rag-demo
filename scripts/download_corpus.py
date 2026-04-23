"""Standalone wrapper around rag_demo.corpus.download_corpus.

Usage:
    uv run python scripts/download_corpus.py --help
    uv run python scripts/download_corpus.py
    uv run python scripts/download_corpus.py --target-dir corpus --overwrite
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from rag_demo.corpus import (  # noqa: E402
    DEFAULT_MANIFEST,
    DEFAULT_TARGET_DIR,
    download_corpus,
    render_report,
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--target-dir", type=Path, default=DEFAULT_TARGET_DIR)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--only", nargs="*", default=None)
    p.add_argument("--report", type=Path, default=None)
    p.add_argument("--json", type=Path, default=None)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    results = download_corpus(
        manifest_path=args.manifest,
        target_dir=args.target_dir,
        overwrite=args.overwrite,
        only=args.only,
    )
    md = render_report(results)
    print(md)

    if args.report:
        args.report.write_text(md, encoding="utf-8")
    if args.json:
        args.json.write_text(
            json.dumps([r.to_dict() for r in results], indent=2), encoding="utf-8"
        )

    failed = sum(1 for r in results if r.status == "failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
