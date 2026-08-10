"""Interfaz de línea de comandos del proyecto.

Comandos disponibles:
- `index`: indexa documentos desde una carpeta.
- `ask`: consulta el índice vectorial ya construido.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rag_cliente.config import get_settings
from rag_cliente.pipeline import RagPipeline


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


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    settings = get_settings()
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
