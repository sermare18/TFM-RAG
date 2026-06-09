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

    index_parser = subparsers.add_parser("index", help="Index PDF, DOCX, TXT and image files from a directory.")
    index_parser.add_argument(
        "--doc-dir",
        type=Path,
        required=True,
        help="Directory containing PDF, DOCX, TXT or image files.",
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
        help="Show model reasoning_content when the backend returns it. Hidden by default.",
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
            result = pipeline.stream_answer(args.question, top_k=args.top_k, tag=args.tag)
            print("\nAnswer:\n")
            received_answer = False
            reasoning_parts: list[str] = []

            for event in result["answer_stream"]:
                if event["type"] == "answer":
                    received_answer = True
                    print(event["delta"], end="", flush=True)
                    continue

                if event["type"] == "reasoning" and args.show_reasoning:
                    reasoning_parts.append(event["delta"])

            if not received_answer:
                fallback_response = result["fallback_response"]()
                if fallback_response["answer"]:
                    print(fallback_response["answer"], end="")
                else:
                    print("[No text returned by the model]")
                if args.show_reasoning and fallback_response["reasoning"]:
                    reasoning_parts.append(fallback_response["reasoning"])

            print("\n")
            if args.show_reasoning and reasoning_parts:
                print("\nReasoning:\n")
                print("".join(reasoning_parts))
        else:
            result = pipeline.ask(args.question, top_k=args.top_k, tag=args.tag)
            print("\nAnswer:\n")
            print(result["answer"] or "[No text returned by the model]")
            if args.show_reasoning and result["reasoning"]:
                print("\nReasoning:\n")
                print(result["reasoning"])

        _print_sources(result["citations"])
        return

    parser.error(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
