"""Command-line interface for document indexing and local RAG queries."""

from __future__ import annotations

import argparse
import json
import sys
from numbers import Real
from pathlib import Path

from rag_cliente.bedrock_parser import BedrockMarkdownParser
from rag_cliente.config import get_settings, resolve_local_model_profile
from rag_cliente.diagnostics import run_doctor
from rag_cliente.model_manifest import check_models, download_models, plan_models
from rag_cliente.pipeline import RagPipeline, grounded_answer


def _help_formatter(prog: str) -> argparse.HelpFormatter:
    return argparse.HelpFormatter(prog, max_help_position=32, width=100)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag.bat",
        description="Bedrock + local models RAG",
        formatter_class=_help_formatter,
    )
    commands = parser.add_subparsers(dest="command", required=True)

    index_parser = commands.add_parser("index", help="Index PDF/Markdown files.")
    index_parser.add_argument("--doc-dir", type=Path, required=True)
    index_parser.add_argument("--tag", default=None)
    index_parser.add_argument(
        "--refresh-bedrock",
        action="store_true",
        help="Ignore a valid PDF Markdown cache and parse it again with Bedrock.",
    )

    preview_parser = commands.add_parser(
        "bedrock-preview",
        help="Extract selected PDF target pages without changing cache or index.",
    )
    preview_parser.add_argument("pdf", type=Path)
    preview_parser.add_argument("pages", type=int, nargs="+")

    ask_parser = commands.add_parser(
        "ask",
        help="Ask against the current index.",
        formatter_class=_help_formatter,
    )
    ask_parser.add_argument("question")
    ask_parser.add_argument("--top-k", type=int, default=None)
    ask_parser.add_argument("--tag", default=None)
    ask_parser.add_argument("--stream", action="store_true")
    ask_parser.add_argument(
        "--no-query-augmentation",
        dest="query_augmentation",
        action="store_false",
        help="Retrieve only with the original question.",
    )
    ask_parser.add_argument(
        "--show-queries",
        action="store_true",
        help="Show every query used for retrieval.",
    )
    ask_parser.set_defaults(query_augmentation=True)
    ask_parser.add_argument(
        "--show-top-k",
        action="store_true",
        help="Show ranked pages and available retrieval scores.",
    )

    commands.add_parser("doctor", help="Validate configuration without loading models.")

    models = commands.add_parser("models", help="Plan, download or check local GGUF files.")
    model_commands = models.add_subparsers(dest="models_command", required=True)
    for command in ("plan", "download"):
        subcommand = model_commands.add_parser(command)
        subcommand.add_argument("profile", choices=("cpu", "gpu"))
    check = model_commands.add_parser("check")
    check.add_argument("--profile", choices=("cpu", "gpu"), default=None)
    return parser


def _print_sources(citations: list[dict]) -> None:
    print("\nSources:\n")
    if not citations:
        print("[No retrieved source was selected as direct support for the answer.]")
        return
    for citation in citations:
        print(
            f"- {citation['source']} [{citation['source_type']}] "
            f"(page {citation['page_start']}, "
            f"page chunk {citation.get('page_chunk_index', 0)}, "
            f"chunk {citation['chunk_index']}, "
            f"path: {citation['source_path']})"
        )


def _print_queries(queries: list[str]) -> None:
    print("\nRetrieval queries:\n")
    for index, query in enumerate(queries, start=1):
        print(f"{index}. {query}")


def _score_parts(match: dict) -> list[str]:
    parts: list[str] = []
    retrieval_sources = match.get("_retrieval_sources")
    if isinstance(retrieval_sources, list) and retrieval_sources:
        parts.append(f"sources={','.join(str(item) for item in retrieval_sources)}")

    rrf_score = match.get("_rrf_score")
    if isinstance(rrf_score, Real):
        parts.append(f"rrf_score={float(rrf_score):.6g} (higher is better)")

    vector_rank = match.get("_vector_rank")
    if isinstance(vector_rank, Real):
        parts.append(f"vector_rank={int(vector_rank)}")
    distance = match.get("_distance")
    if isinstance(distance, Real):
        parts.append(f"distance={float(distance):.6g} (lower is better)")

    bm25_rank = match.get("_bm25_rank")
    if isinstance(bm25_rank, Real):
        parts.append(f"bm25_rank={int(bm25_rank)}")
    bm25_score = match.get("_bm25_raw_score")
    if not isinstance(bm25_score, Real):
        bm25_score = match.get("_bm25_score")
    if isinstance(bm25_score, Real):
        parts.append(f"bm25_score={float(bm25_score):.6g} (higher is better)")
    return parts


def _print_top_k(matches: list[dict]) -> None:
    print("\nTop-k retrieved pages:\n")
    if not matches:
        print("[No pages were retrieved.]")
        return
    for rank, match in enumerate(matches, start=1):
        print(
            f"- #{rank} {match.get('source', 'unknown')} "
            f"(page {match.get('page_start', '?')}, "
            f"page chunk {match.get('page_chunk_index', 0)}, "
            f"chunk {match.get('chunk_index', '?')}, "
            f"path: {match.get('source_path', 'unknown')})"
        )
        scores = _score_parts(match)
        if scores:
            print(f"  {' · '.join(scores)}")


def _print_query_diagnostics(result: dict, show_queries: bool) -> None:
    error = result.get("query_augmentation_error")
    if error:
        print(
            "WARNING: query augmentation failed; retrieval continued with the "
            f"original question. Detail: {error}",
            file=sys.stderr,
        )
    if show_queries:
        _print_queries(result.get("retrieval_queries", []))


def _print_model_reports(reports: list[dict]) -> None:
    for report in reports:
        state = "OK" if report["valid"] else "PENDIENTE"
        print(f"[{state}] {report['label']} ({report['quantization']})")
        print(f"  repo: {report['repository']}")
        for artifact in report["artifacts"]:
            artifact_state = "OK" if artifact["valid"] else "--"
            patterns = ", ".join(artifact.get("patterns", []))
            print(
                f"  [{artifact_state}] model: {artifact['path']} "
                f"({artifact['expected_size']}; archivos: {patterns}; "
                f"{artifact['message']}; reutilizable="
                f"{'si' if artifact['valid'] else 'no'})"
            )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    settings = get_settings()

    if args.command == "doctor":
        report = run_doctor(settings)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if not report["ok"]:
            raise SystemExit(1)
        return

    if args.command == "models":
        if args.models_command == "plan":
            _print_model_reports(plan_models(settings, args.profile))
            return
        if args.models_command == "download":
            results = download_models(settings, args.profile)
            for result in results:
                state = "DESCARGADO" if result["downloaded"] else "ERROR"
                print(f"[{state}] {result['role']}: {result['message']}")
            if not all(result["downloaded"] for result in results):
                raise SystemExit(1)
            return
        profile = args.profile or resolve_local_model_profile(settings)
        reports = check_models(settings, profile)
        _print_model_reports(reports)
        if not all(report["valid"] for report in reports):
            raise SystemExit(1)
        return

    if args.command == "bedrock-preview":
        try:
            pages = BedrockMarkdownParser(settings).preview_pdf_pages(
                args.pdf,
                args.pages,
                progress_callback=print,
            )
        except (FileNotFoundError, RuntimeError, TimeoutError, ValueError) as exc:
            parser.exit(1, f"ERROR: {exc}\n")
        for page in pages:
            print(f"\n<!-- PAGE {page.page_number} -->\n{page.markdown}")
        return

    settings.lancedb_path.mkdir(parents=True, exist_ok=True)
    pipeline = RagPipeline(settings)

    if args.command == "index":
        try:
            count = pipeline.index_documents(
                args.doc_dir,
                tag=args.tag,
                progress_callback=print,
                refresh_bedrock=args.refresh_bedrock,
            )
        except (FileNotFoundError, RuntimeError, TimeoutError) as exc:
            parser.exit(1, f"ERROR: {exc}\n")
        print(f"Indexed {count} chunks into '{settings.lancedb_table}'.")
        return

    if args.stream:
        result = pipeline.stream_answer(
            args.question,
            top_k=args.top_k,
            tag=args.tag,
            query_augmentation=args.query_augmentation,
        )
        _print_query_diagnostics(result, args.show_queries)
        answer_parts: list[str] = []
        for event in result["answer_stream"]:
            if event["type"] == "answer":
                answer_parts.append(event["delta"])
        if not answer_parts:
            for event in result["fallback_stream"]():
                if event["type"] == "answer":
                    answer_parts.append(event["delta"])
        raw_answer = "".join(answer_parts)
        citations = result["resolve_citations"](raw_answer)
        print(grounded_answer(raw_answer, citations))
    else:
        result = pipeline.ask(
            args.question,
            top_k=args.top_k,
            tag=args.tag,
            query_augmentation=args.query_augmentation,
        )
        _print_query_diagnostics(result, args.show_queries)
        citations = result["citations"]
        print("\nAnswer:\n")
        print(grounded_answer(result["answer"], citations))
    _print_sources(citations)
    if args.show_top_k:
        _print_top_k(result.get("matches", []))


if __name__ == "__main__":
    main()
