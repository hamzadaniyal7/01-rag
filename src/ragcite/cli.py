"""`ragcite` -- the command line.

    ragcite init                      write a starter ragcite.toml
    ragcite ingest corpus/            load, chunk, embed, index
    ragcite search "refunds"          retrieval only -- no model, no cost
    ragcite ask "how long for a refund?"
    ragcite stats                     what is in the index
    ragcite eval                      score the pipeline against eval/questions.jsonl
    ragcite serve                     HTTP API (pip install -e '.[api]')

`search` exists because most "the RAG is bad" bugs are retrieval bugs, and you
can see those without spending a token.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ragcite.config import CONFIG_FILENAME, Config
from ragcite.pipeline import Rag
from ragcite.retrieval import format_hits
from ragcite.store import EmbedderMismatch

STARTER_CONFIG = """# ragcite configuration -- see docs/03-pipeline.md for what each knob does.
[ragcite]
db_path = ".ragcite/index.db"
corpus_dir = "corpus"

# Ingestion
chunk_size = 512        # target chunk size in tokens
chunk_overlap = 64

# Embeddings: hash:512 (offline, lexical) | st:all-MiniLM-L6-v2 (local, semantic)
#             | voyage:voyage-3 (hosted, best quality)
embedder = "hash:512"

# Retrieval
retrieval_mode = "hybrid"   # dense | lexical | hybrid
top_k = 8
candidate_k = 40
max_per_doc = 3

# Generation
model = "claude-opus-5"
effort = "medium"           # low | medium | high | xhigh | max
rerank = false
"""


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 1
    try:
        return args.func(args)
    except EmbedderMismatch as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:  # pragma: no cover
        return 130
    except (FileNotFoundError, ValueError, ImportError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #
def cmd_init(args: argparse.Namespace) -> int:
    path = Path(CONFIG_FILENAME)
    if path.exists() and not args.force:
        print(f"{path} already exists (use --force to overwrite)")
        return 1
    path.write_text(STARTER_CONFIG, encoding="utf-8")
    Path("corpus").mkdir(exist_ok=True)
    print(f"wrote {path}\nnext: put documents in corpus/ then run `ragcite ingest corpus/`")
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    cfg = _config(args)
    paths = args.paths or [cfg.corpus_dir]
    with Rag(cfg) as rag:
        report = rag.ingest(
            paths,
            force=args.force,
            reset=args.reset,
            on_file=lambda p, n: print(f"  + {p} ({n} chunks)"),
        )
        print(report.summary())
        print(f"index: {rag.store.path} (embedder {rag.embedder.name})")
    return 1 if report.failures and report.documents_indexed == 0 else 0


def cmd_search(args: argparse.Namespace) -> int:
    cfg = _config(args)
    with Rag(cfg) as rag:
        hits = rag.search(args.question, k=args.k, mode=args.mode)
        if args.json:
            print(
                json.dumps(
                    [
                        {
                            "chunk_id": h.chunk.chunk_id,
                            "path": h.chunk.doc_path,
                            "locator": h.chunk.locator,
                            "score": h.score,
                            "why": h.why(),
                            "text": h.chunk.text,
                        }
                        for h in hits
                    ],
                    indent=2,
                )
            )
        else:
            print(format_hits(hits))
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    cfg = _config(args)
    # Stream to the terminal unless the caller wants machine-readable output.
    stream_to = None if (args.json or args.quiet) else sys.stdout
    with Rag(cfg, stream_to=stream_to) as rag:
        hits = rag.search(
            args.question,
            k=max(cfg.candidate_k, cfg.top_k) if (args.rerank or cfg.rerank) else args.k,
            mode=args.mode,
        )
        if args.show_context:
            print(format_hits(hits), file=sys.stderr)
            print("---", file=sys.stderr)

        if args.dry_run:
            tokens = rag.count_prompt_tokens(args.question, hits)
            print(f"retrieved {len(hits)} chunk(s)")
            print(f"prompt tokens: {tokens if tokens is not None else 'n/a (fake llm)'}")
            print(f"model: {cfg.model}  effort: {cfg.effort}")
            return 0

        answer = rag.ask(
            args.question,
            k=args.k,
            mode=args.mode,
            rerank=args.rerank or None,
            structured=args.structured,
        )
        if args.json:
            print(answer.to_json())
        else:
            if stream_to is not None:
                print()  # the answer text already streamed; close the line
                if answer.citations or answer.dangling_markers:
                    print()
                    print("Sources:")
                    for citation in answer.citations:
                        print("  " + citation.render())
                    if answer.dangling_markers:
                        print(
                            "  !! cited markers with no matching source: "
                            + ", ".join(f"[{m}]" for m in answer.dangling_markers)
                        )
            else:
                print(answer.render())
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    cfg = _config(args)
    with Rag(cfg) as rag:
        stats = rag.stats()
        width = max(len(k) for k in stats)
        for key, value in stats.items():
            print(f"{key:<{width}}  {value}")
        if args.documents:
            print()
            for row in rag.store.documents():
                print(f"  {row['path']}  ({row['n_pages'] or '-'} pages, {row['ingested_at']})")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from ragcite.evaluate import evaluate, load_cases, render_report, save_report

    cfg = _config(args)
    cases = load_cases(args.cases)
    with Rag(cfg) as rag:
        report = evaluate(rag, cases, answer=not args.retrieval_only, judge=args.judge, k=args.k)
    print(render_report(report, verbose=args.verbose))
    if not args.no_save:
        path = save_report(report)
        print(f"\nreport: {path}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ImportError:
        print("error: pip install -e '.[api]'", file=sys.stderr)
        return 1
    from ragcite.api import create_app

    uvicorn.run(create_app(_config(args)), host=args.host, port=args.port)
    return 0


# --------------------------------------------------------------------------- #
# parser
# --------------------------------------------------------------------------- #
def _config(args: argparse.Namespace) -> Config:
    overrides = {
        key: getattr(args, key, None)
        for key in ("embedder", "model", "db_path", "llm", "effort", "top_k")
    }
    if getattr(args, "k", None):
        overrides["top_k"] = args.k
    return Config.load(getattr(args, "config", None), **overrides)


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", help=f"path to {CONFIG_FILENAME}")
    p.add_argument("--db-path", dest="db_path", help="override the index location")
    p.add_argument("--embedder", help="hash:512 | st:<model> | voyage:<model>")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ragcite",
        description="RAG over your own documents, with citations you can check.",
    )
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("init", help="write a starter ragcite.toml")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_init)

    p = sub.add_parser("ingest", help="load, chunk, embed and index documents")
    _add_common(p)
    p.add_argument("paths", nargs="*", help="files or directories (default: corpus_dir)")
    p.add_argument("--force", action="store_true", help="re-index even if unchanged")
    p.add_argument("--reset", action="store_true", help="wipe the index first")
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("search", help="retrieval only -- no model call")
    _add_common(p)
    p.add_argument("question")
    p.add_argument("-k", type=int, help="how many chunks to return")
    p.add_argument("--mode", choices=["dense", "lexical", "hybrid"])
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("ask", help="retrieve, then answer with citations")
    _add_common(p)
    p.add_argument("question")
    p.add_argument("-k", type=int)
    p.add_argument("--mode", choices=["dense", "lexical", "hybrid"])
    p.add_argument("--model", help="e.g. claude-opus-5")
    p.add_argument("--effort", choices=["low", "medium", "high", "xhigh", "max"])
    p.add_argument("--llm", choices=["anthropic", "fake"])
    p.add_argument("--rerank", action="store_true", help="LLM-rerank the candidate pool")
    p.add_argument(
        "--structured", action="store_true", help="constrain the model to the JSON answer schema"
    )
    p.add_argument(
        "--show-context", action="store_true", help="print the retrieved chunks to stderr"
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="retrieve and count prompt tokens, but do not generate",
    )
    p.add_argument("--json", action="store_true")
    p.add_argument("--quiet", action="store_true", help="do not stream tokens")
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("stats", help="what is in the index")
    _add_common(p)
    p.add_argument("--documents", action="store_true", help="list indexed documents")
    p.set_defaults(func=cmd_stats)

    p = sub.add_parser("eval", help="score the pipeline against an eval set")
    _add_common(p)
    p.add_argument("--cases", default="eval/questions.jsonl")
    p.add_argument("-k", type=int)
    p.add_argument("--llm", choices=["anthropic", "fake"])
    p.add_argument("--model")
    p.add_argument("--effort", choices=["low", "medium", "high", "xhigh", "max"])
    p.add_argument(
        "--retrieval-only", action="store_true", help="skip generation: fast, free, deterministic"
    )
    p.add_argument("--judge", action="store_true", help="also grade answers with an LLM judge")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--no-save", action="store_true")
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("serve", help="run the HTTP API")
    _add_common(p)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--llm", choices=["anthropic", "fake"])
    p.add_argument("--model")
    p.set_defaults(func=cmd_serve)

    return parser


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
