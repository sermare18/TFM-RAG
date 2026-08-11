"""Command-line interface for document indexing and local RAG queries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag_cliente.config import get_settings, resolve_local_model_profile
from rag_cliente.diagnostics import run_doctor
from rag_cliente.model_manifest import check_models, download_models, plan_models
from rag_cliente.pipeline import RagPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bedrock + local models RAG")
    commands = parser.add_subparsers(dest="command", required=True)

    index_parser = commands.add_parser("index", help="Index PDF/Markdown files.")
    index_parser.add_argument("--doc-dir", type=Path, required=True)
    index_parser.add_argument("--tag", default=None)
    index_parser.add_argument(
        "--refresh-bedrock",
        action="store_true",
        help="Ignore a valid PDF Markdown cache and parse it again with Bedrock.",
    )

    ask_parser = commands.add_parser("ask", help="Ask against the current index.")
    ask_parser.add_argument("question")
    ask_parser.add_argument("--top-k", type=int, default=None)
    ask_parser.add_argument("--tag", default=None)
    ask_parser.add_argument("--stream", action="store_true")
    ask_parser.add_argument("--show-reasoning", action="store_true")

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
    for citation in citations:
        print(
            f"- {citation['source']} [{citation['source_type']}] "
            f"(page {citation['page_start']}, chunk {citation['chunk_index']}, "
            f"path: {citation['source_path']})"
        )


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
            enable_reasoning=args.show_reasoning,
        )
        answer_parts: list[str] = []
        for event in result["answer_stream"]:
            if event["type"] == "answer":
                answer_parts.append(event["delta"])
                print(event["delta"], end="", flush=True)
            elif event["type"] == "reasoning" and args.show_reasoning:
                print(event["delta"], end="", flush=True)
        print()
        citations = result["resolve_citations"]("".join(answer_parts))
    else:
        result = pipeline.ask(
            args.question,
            top_k=args.top_k,
            tag=args.tag,
            enable_reasoning=args.show_reasoning,
        )
        print("\nAnswer:\n")
        print(result["answer"] or "[No text returned by the model]")
        if args.show_reasoning and result["reasoning"]:
            print("\nReasoning:\n")
            print(result["reasoning"])
        citations = result["citations"]
    _print_sources(citations)


if __name__ == "__main__":
    main()
