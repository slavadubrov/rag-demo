"""Command-line entry points."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from .baseline import query as run_query
from .corpus import (
    DEFAULT_MANIFEST,
    DEFAULT_TARGET_DIR,
    download_corpus,
    render_report as render_download_report,
)
from .index import list_documents, rebuild_index, reset_client


def cmd_rebuild(args: argparse.Namespace) -> None:
    report = rebuild_index()
    print(json.dumps(report.model_dump(), indent=2))


def cmd_list(args: argparse.Namespace) -> None:
    for d in list_documents():
        print(f"{d.doc_id}\t{d.page_count}p\t{d.chunk_count} chunks\t{d.title}")


def cmd_query(args: argparse.Namespace) -> None:
    payload = run_query(
        args.question,
        mode=args.mode,
        doc_ids=args.docs.split(",") if args.docs else None,
        top_k=args.top_k,
    )
    print(payload.answer)
    print()
    print("Citations:")
    for c in payload.citations:
        print(" ", c)
    print()
    print("Debug:", json.dumps(payload.debug, indent=2))


def cmd_app(args: argparse.Namespace) -> None:
    from .app import main as launch

    reset_client()
    launch()


def cmd_download(args: argparse.Namespace) -> None:
    results = download_corpus(
        manifest_path=args.manifest,
        target_dir=args.target_dir,
        overwrite=args.overwrite,
        only=args.only,
    )
    md = render_download_report(results)
    print(md)
    if args.report:
        args.report.write_text(md, encoding="utf-8")
    if args.json:
        args.json.write_text(
            json.dumps([r.to_dict() for r in results], indent=2), encoding="utf-8"
        )


def cmd_eval(args: argparse.Namespace) -> None:
    from .eval import (
        DEFAULT_REPORT_DIR,
        render_markdown,
        run_eval,
        write_reports,
    )

    report = run_eval(
        corpus_manifest_path=args.manifest,
        only_filenames=args.only,
        max_questions_per_doc=args.max_questions,
        top_k=args.top_k,
        use_judge=args.judge,
    )
    json_out = args.json or (DEFAULT_REPORT_DIR / "eval.json")
    md_out = args.report or (DEFAULT_REPORT_DIR / "eval.md")
    write_reports(report, json_out, md_out)
    print(render_markdown(report))
    print(f"\nWrote {md_out}\nWrote {json_out}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rag-demo")
    sub = p.add_subparsers(dest="cmd", required=True)

    rb = sub.add_parser(
        "rebuild", help="Re-ingest the corpus and rebuild both vector indexes."
    )
    rb.set_defaults(func=cmd_rebuild)

    ls = sub.add_parser("list", help="List ingested documents.")
    ls.set_defaults(func=cmd_list)

    q = sub.add_parser("query", help="Ask a single question.")
    q.add_argument("question")
    q.add_argument("--mode", choices=["multimodal", "baseline"], default="multimodal")
    q.add_argument("--docs", default="", help="Comma-separated doc_ids; default = all")
    q.add_argument("--top-k", type=int, default=6)
    q.set_defaults(func=cmd_query)

    app = sub.add_parser("app", help="Launch the Gradio UI.")
    app.set_defaults(func=cmd_app)

    dl = sub.add_parser(
        "download", help="Download the curated corpus from the manifest."
    )
    dl.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    dl.add_argument("--target-dir", type=Path, default=DEFAULT_TARGET_DIR)
    dl.add_argument("--overwrite", action="store_true")
    dl.add_argument("--only", nargs="*", default=None)
    dl.add_argument("--report", type=Path, default=None)
    dl.add_argument("--json", type=Path, default=None)
    dl.set_defaults(func=cmd_download)

    ev = sub.add_parser(
        "eval",
        help="Evaluate extraction + generation against the manifest's benchmark questions.",
    )
    ev.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ev.add_argument(
        "--only", nargs="*", default=None, help="restrict to these filenames"
    )
    ev.add_argument(
        "--max-questions", type=int, default=None, help="cap questions per doc"
    )
    ev.add_argument("--top-k", type=int, default=6)
    ev.add_argument(
        "--judge", action="store_true", help="use the LLM judge (extra OpenAI cost)"
    )
    ev.add_argument("--report", type=Path, default=None, help="markdown report path")
    ev.add_argument("--json", type=Path, default=None, help="JSON report path")
    ev.set_defaults(func=cmd_eval)

    return p


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
