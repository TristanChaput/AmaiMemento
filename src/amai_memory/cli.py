from __future__ import annotations

import argparse
import json
import sys

from amai_memory.memora_adapter import (
    EMBEDDINGGEMMA2_MODEL,
    QWEN_EMBEDDING_MODEL,
    MemoraUnavailableError,
)
from amai_memory.service import (
    MemoryPaths,
    benchmark_embeddings,
    build_index,
    current_manifest,
    query_index,
    scan,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local Memora index for AmaiKnowledge")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("scan", help="List the Markdown notes that would be indexed")
    commands.add_parser("status", help="Show the active index manifest")
    rebuild = commands.add_parser(
        "rebuild", help="Build or resume and activate an index"
    )
    rebuild.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Parallel LLM extraction workers (default: MEMORA_REBUILD_WORKERS or 2)",
    )
    rebuild.add_argument(
        "--resume-after",
        metavar="NOTE_PATH",
        help="Resume a checkpointed rebuild after this completed note",
    )
    cue_mode = rebuild.add_mutually_exclusive_group()
    cue_mode.add_argument(
        "--cues",
        dest="enable_cue_index",
        action="store_true",
        help="Generate richer cue indices (slower; disabled by default)",
    )
    cue_mode.add_argument(
        "--no-cues",
        dest="enable_cue_index",
        action="store_false",
        help="Skip cue-index generation for a faster rebuild",
    )
    rebuild.set_defaults(enable_cue_index=None)
    query = commands.add_parser(
        "query", help="Run semantic search against the active index"
    )
    query.add_argument("question")
    query.add_argument("--top-k", type=int, default=5)
    benchmark = commands.add_parser(
        "benchmark-embeddings",
        help="Compare embedding models over the active index's memories",
    )
    benchmark.add_argument(
        "questions",
        nargs="*",
        help="Probe questions (default: sample of the indexed memories)",
    )
    benchmark.add_argument("--top-k", type=int, default=5)
    benchmark.add_argument(
        "--models",
        nargs="+",
        default=[QWEN_EMBEDDING_MODEL, EMBEDDINGGEMMA2_MODEL],
        help="Embedding models to compare",
    )
    benchmark.add_argument("--probe-count", type=int, default=8)
    return parser


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = _parser().parse_args()
    paths = MemoryPaths.from_environment()
    try:
        if args.command == "scan":
            documents = scan(paths)
            output = {
                "vault": str(paths.vault),
                "count": len(documents),
                "documents": [document.relative_path for document in documents],
            }
            print(json.dumps(output, indent=2, ensure_ascii=False))
        elif args.command == "status":
            print(
                json.dumps(current_manifest(paths) or {"status": "not_built"}, indent=2)
            )
        elif args.command == "rebuild":
            documents = scan(paths)
            manifest = build_index(
                paths,
                documents,
                progress=lambda current, total, path: print(
                    f"[{current}/{total}] {path}", flush=True
                ),
                workers=args.workers,
                enable_cue_index=args.enable_cue_index,
                resume_after=args.resume_after,
            )
            print(json.dumps(manifest, indent=2, ensure_ascii=False))
        elif args.command == "query":
            print(
                json.dumps(
                    query_index(paths, args.question, args.top_k),
                    indent=2,
                    ensure_ascii=False,
                )
            )
        elif args.command == "benchmark-embeddings":
            print(
                json.dumps(
                    benchmark_embeddings(
                        paths,
                        questions=args.questions,
                        top_k=args.top_k,
                        models=args.models,
                        probe_count=args.probe_count,
                    ),
                    indent=2,
                    ensure_ascii=False,
                )
            )
    except (FileNotFoundError, MemoraUnavailableError, ValueError) as error:
        print(f"error: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
