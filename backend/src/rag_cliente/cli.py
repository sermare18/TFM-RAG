"""Interfaz de línea de comandos del proyecto.

Comandos disponibles:
- `index`: indexa documentos desde una carpeta.
- `ask`: consulta el índice vectorial ya construido.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag_cliente.config import get_settings, resolve_marker_profile
from rag_cliente.diagnostics import run_doctor
from rag_cliente.model_manifest import check_models, download_models, plan_models
from rag_cliente.pipeline import RagPipeline


def run_smoke_parser(pdf_path: Path, settings):
    """Importación perezosa para no cargar el parser en comandos de chat/API."""
    from rag_cliente.smoke_parser import run_smoke_parser as run

    return run(pdf_path, settings)


def build_parser() -> argparse.ArgumentParser:
    """Construye el parser principal y sus subcomandos."""
    parser = argparse.ArgumentParser(description="Local RAG CLI for document indexing and Q&A.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index", help="Index documents supported by Marker 2 from a directory.")
    index_parser.add_argument(
        "--doc-dir",
        type=Path,
        required=True,
        help="Directory containing PDF, Office, EPUB, HTML, TXT or image files.",
    )
    index_parser.add_argument("--tag", type=str, default=None, help="Metadata tag to assign to indexed chunks.")

    ask_parser = subparsers.add_parser("ask", help="Ask a question against the indexed documents.")
    ask_parser.add_argument("question", type=str, help="Question to answer.")
    ask_parser.add_argument("--top-k", type=int, default=None, help="Number of chunks to retrieve.")
    ask_parser.add_argument("--tag", type=str, default=None, help="Only retrieve chunks with this metadata tag.")
    ask_parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream the answer token by token as it is generated.",
    )
    ask_parser.add_argument(
        "--show-reasoning",
        action="store_true",
        help="Enable model thinking for this request and show its reasoning.",
    )

    subparsers.add_parser(
        "doctor",
        help="Validate hardware, llama.cpp, disk, profiles and local configuration.",
    )

    models_parser = subparsers.add_parser("models", help="Plan, download or validate local GGUF files.")
    models_subparsers = models_parser.add_subparsers(dest="models_command", required=True)
    for command in ("plan", "download"):
        command_parser = models_subparsers.add_parser(command)
        command_parser.add_argument("profile", choices=("cpu", "gpu"))
    check_parser = models_subparsers.add_parser("check")
    check_parser.add_argument("--profile", choices=("cpu", "gpu"), default=None)

    smoke_parser = subparsers.add_parser(
        "smoke-parser",
        help="Manually parse a bounded PDF page range and emit structured JSON.",
    )
    smoke_parser.add_argument("pdf", type=Path, help="PDF file to parse.")
    smoke_parser.add_argument(
        "--profile",
        required=True,
        choices=("cpu-digital", "cpu-quality", "gpu-quality", "auto"),
    )
    smoke_parser.add_argument(
        "--pages",
        required=True,
        help="Marker 0-based page range, for example 0-2 or 0,3,5-6.",
    )

    return parser


def _print_sources(citations: list[dict]) -> None:
    print("\nSources:\n")
    for citation in citations:
        ocr_label = "sí" if citation.get("ocr_used") else "no"
        print(
            f"- {citation['source']} [{citation['source_type']}] "
            f"(pages {citation['page_start']}-{citation['page_end']}, "
            f"chunk {citation['chunk_index']}, OCR: {ocr_label}, "
            f"path: {citation['source_path']})"
        )


def _print_model_reports(reports: list[dict]) -> None:
    for report in reports:
        state = "OK" if report["valid"] else "PENDIENTE"
        repository = report.get("repository") or "ruta local configurable"
        print(f"[{state}] {report['label']} ({report['quantization']})")
        print(f"  repo: {repository}")
        for artifact in report["artifacts"]:
            artifact_state = "OK" if artifact["valid"] else "--"
            expected = artifact.get("expected_size", "tamaño configurable")
            patterns = ", ".join(artifact.get("patterns", []))
            pattern_label = f"; archivos: {patterns}" if patterns else ""
            print(
                f"  [{artifact_state}] {artifact['kind']}: {artifact['path']} "
                f"({expected}{pattern_label}; {artifact['message']}; "
                f"reutilizable={'sí' if artifact['valid'] else 'no'})"
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
                state = "DESCARGADO" if result["downloaded"] else "REUTILIZAR"
                print(f"[{state}] {result['role']}: {result['message']}")
            return
        if args.models_command == "check":
            selected_profile = args.profile
            if selected_profile is None:
                selected_profile = (
                    "gpu"
                    if resolve_marker_profile(settings).name == "gpu-quality"
                    else "cpu"
                )
            reports = check_models(settings, selected_profile)
            _print_model_reports(reports)
            if not all(report["valid"] for report in reports):
                raise SystemExit(1)
            return

    if args.command == "smoke-parser":
        smoke_settings = settings.model_copy(
            update={
                "marker_profile": args.profile,
                "marker_page_range": args.pages,
            }
        )
        report = run_smoke_parser(args.pdf, smoke_settings)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    settings.lancedb_path.mkdir(parents=True, exist_ok=True)
    pipeline = RagPipeline(settings)

    if args.command == "index":
        indexed_chunks = pipeline.index_documents(args.doc_dir, tag=args.tag, progress_callback=print)
        print(
            f"Indexed {indexed_chunks} chunks into LanceDB table '{settings.lancedb_table}' "
            f"with source metadata."
        )
        return

    if args.command == "ask":
        if args.stream:
            result = pipeline.stream_answer(
                args.question,
                top_k=args.top_k,
                tag=args.tag,
                enable_reasoning=args.show_reasoning,
            )
            received_answer = False
            answer_parts: list[str] = []
            reasoning_parts: list[str] = []
            answer_heading_printed = False
            reasoning_heading_printed = False

            for event in result["answer_stream"]:
                if event["type"] == "answer":
                    received_answer = True
                    answer_parts.append(event["delta"])
                    if not answer_heading_printed:
                        print("\nAnswer:\n")
                        answer_heading_printed = True
                    print(event["delta"], end="", flush=True)
                    continue

                if event["type"] == "reasoning" and args.show_reasoning:
                    reasoning_parts.append(event["delta"])
                    if not reasoning_heading_printed:
                        print("\nReasoning:\n")
                        reasoning_heading_printed = True
                    print(event["delta"], end="", flush=True)

            if not received_answer:
                for event in result["fallback_stream"]():
                    if event["type"] != "answer":
                        continue
                    received_answer = True
                    answer_parts.append(event["delta"])
                    if not answer_heading_printed:
                        print("\n\nAnswer (fallback sin thinking):\n")
                        answer_heading_printed = True
                    print(event["delta"], end="", flush=True)

                if not received_answer:
                    if not answer_heading_printed:
                        print("\nAnswer:\n")
                    print("[No text returned by the model]")

            print("\n")

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
        return

    parser.error(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
