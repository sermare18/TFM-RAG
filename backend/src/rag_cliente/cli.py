"""Interfaz de línea de comandos para indexar documentos y consultar el RAG."""

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


class SpanishArgumentParser(argparse.ArgumentParser):
    """Localiza los textos automáticos de argparse sin cambiar comandos ni flags."""

    @staticmethod
    def _translate_help(text: str) -> str:
        replacements = (
            ("usage:", "uso:"),
            ("positional arguments:", "argumentos posicionales:"),
            ("options:", "opciones:"),
            ("show this help message and exit", "muestra esta ayuda y termina"),
        )
        for source, target in replacements:
            text = text.replace(source, target)
        return text

    def format_help(self) -> str:
        return self._translate_help(super().format_help())

    def format_usage(self) -> str:
        return self._translate_help(super().format_usage())

    def error(self, message: str) -> None:
        replacements = (
            ("unrecognized arguments:", "argumentos no reconocidos:"),
            (
                "the following arguments are required:",
                "faltan los siguientes argumentos obligatorios:",
            ),
            ("invalid choice:", "opción no válida:"),
            ("expected one argument", "se esperaba un argumento"),
            ("invalid int value:", "valor entero no válido:"),
        )
        for source, target in replacements:
            message = message.replace(source, target)
        self.print_usage(sys.stderr)
        self.exit(2, f"{self.prog}: error: {message}\n")


def build_parser() -> argparse.ArgumentParser:
    parser = SpanishArgumentParser(
        prog="rag.bat",
        description="RAG con Bedrock y modelos locales",
        formatter_class=_help_formatter,
    )
    commands = parser.add_subparsers(dest="command", required=True)

    index_parser = commands.add_parser("index", help="Indexa archivos PDF y Markdown.")
    index_parser.add_argument(
        "--doc-dir",
        type=Path,
        required=True,
        help="Directorio que contiene los documentos.",
    )
    index_parser.add_argument("--tag", default=None, help="Etiqueta asignada a los documentos.")
    index_parser.add_argument(
        "--refresh-bedrock",
        action="store_true",
        help="Ignora una caché Markdown válida y vuelve a procesar el PDF con Bedrock.",
    )

    preview_parser = commands.add_parser(
        "bedrock-preview",
        help="Extrae páginas concretas de un PDF sin cambiar la caché ni el índice.",
    )
    preview_parser.add_argument("pdf", type=Path, help="Ruta del PDF que se va a previsualizar.")
    preview_parser.add_argument("pages", type=int, nargs="+", help="Páginas que se van a extraer.")

    ask_parser = commands.add_parser(
        "ask",
        help="Consulta el índice actual.",
        formatter_class=_help_formatter,
    )
    ask_parser.add_argument("question", help="Pregunta que se enviará al RAG.")
    ask_parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Número máximo de páginas que se recuperarán.",
    )
    ask_parser.add_argument("--tag", default=None, help="Filtra la recuperación por etiqueta.")
    ask_parser.add_argument(
        "--stream",
        action="store_true",
        help="Usa el flujo de respuesta y la entrega después de auditarla.",
    )
    ask_parser.add_argument(
        "--no-query-augmentation",
        dest="query_augmentation",
        action="store_false",
        help="Recupera únicamente con la pregunta original.",
    )
    ask_parser.add_argument(
        "--show-queries",
        action="store_true",
        help="Muestra todas las consultas utilizadas en la recuperación.",
    )
    ask_parser.set_defaults(query_augmentation=True)
    ask_parser.add_argument(
        "--show-top-k",
        action="store_true",
        help="Muestra las páginas ordenadas y sus puntuaciones de recuperación.",
    )

    commands.add_parser("doctor", help="Valida la configuración sin cargar modelos.")

    models = commands.add_parser("models", help="Planifica, descarga o comprueba los GGUF locales.")
    model_commands = models.add_subparsers(dest="models_command", required=True)
    model_help = {
        "plan": "Muestra los modelos necesarios sin descargarlos.",
        "download": "Descarga los modelos del perfil seleccionado.",
    }
    for command in ("plan", "download"):
        subcommand = model_commands.add_parser(command, help=model_help[command])
        subcommand.add_argument(
            "profile",
            choices=("cpu", "gpu"),
            help="Perfil de hardware local.",
        )
    check = model_commands.add_parser("check", help="Comprueba los modelos locales.")
    check.add_argument(
        "--profile",
        choices=("cpu", "gpu"),
        default=None,
        help="Limita la comprobación a un perfil.",
    )
    return parser


def _print_sources(citations: list[dict]) -> None:
    print("\nFuentes:\n")
    if not citations:
        print("[Ninguna fuente recuperada respalda directamente la respuesta.]")
        return
    for citation in citations:
        print(
            f"- {citation['source']} [{citation['source_type']}] "
            f"(página {citation['page_start']}, "
            f"chunk de página {citation.get('page_chunk_index', 0)}, "
            f"chunk {citation['chunk_index']}, "
            f"ruta: {citation['source_path']})"
        )


def _print_queries(queries: list[str]) -> None:
    print("\nConsultas de recuperación:\n")
    for index, query in enumerate(queries, start=1):
        print(f"{index}. {query}")


def _score_parts(match: dict) -> list[str]:
    parts: list[str] = []
    retrieval_sources = match.get("_retrieval_sources")
    if isinstance(retrieval_sources, list) and retrieval_sources:
        parts.append(f"fuentes={','.join(str(item) for item in retrieval_sources)}")

    rrf_score = match.get("_rrf_score")
    if isinstance(rrf_score, Real):
        parts.append(f"puntuación_rrf={float(rrf_score):.6g} (mayor es mejor)")

    vector_rank = match.get("_vector_rank")
    if isinstance(vector_rank, Real):
        parts.append(f"posición_vectorial={int(vector_rank)}")
    distance = match.get("_distance")
    if isinstance(distance, Real):
        parts.append(f"distancia={float(distance):.6g} (menor es mejor)")

    bm25_rank = match.get("_bm25_rank")
    if isinstance(bm25_rank, Real):
        parts.append(f"posición_bm25={int(bm25_rank)}")
    bm25_score = match.get("_bm25_raw_score")
    if not isinstance(bm25_score, Real):
        bm25_score = match.get("_bm25_score")
    if isinstance(bm25_score, Real):
        parts.append(f"puntuación_bm25={float(bm25_score):.6g} (mayor es mejor)")
    return parts


def _print_top_k(matches: list[dict]) -> None:
    print("\nPáginas recuperadas en el top-k:\n")
    if not matches:
        print("[No se recuperó ninguna página.]")
        return
    for rank, match in enumerate(matches, start=1):
        print(
            f"- #{rank} {match.get('source', 'desconocida')} "
            f"(página {match.get('page_start', '?')}, "
            f"chunk de página {match.get('page_chunk_index', 0)}, "
            f"chunk {match.get('chunk_index', '?')}, "
            f"ruta: {match.get('source_path', 'desconocida')})"
        )
        scores = _score_parts(match)
        if scores:
            print(f"  {' · '.join(scores)}")


def _print_query_diagnostics(result: dict, show_queries: bool) -> None:
    error = result.get("query_augmentation_error")
    if error:
        print(
            "ADVERTENCIA: falló el aumento de consultas; la recuperación continuó "
            f"con la pregunta original. Detalle: {error}",
            file=sys.stderr,
        )
    if show_queries:
        _print_queries(result.get("retrieval_queries", []))


def _print_model_reports(reports: list[dict]) -> None:
    for report in reports:
        state = "OK" if report["valid"] else "PENDIENTE"
        print(f"[{state}] {report['label']} ({report['quantization']})")
        print(f"  repositorio: {report['repository']}")
        for artifact in report["artifacts"]:
            artifact_state = "OK" if artifact["valid"] else "--"
            patterns = ", ".join(artifact.get("patterns", []))
            print(
                f"  [{artifact_state}] modelo: {artifact['path']} "
                f"({artifact['expected_size']}; archivos: {patterns}; "
                f"{artifact['message']}; reutilizable="
                f"{'sí' if artifact['valid'] else 'no'})"
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
            print(f"\n<!-- PÁGINA {page.page_number} -->\n{page.markdown}")
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
        print(f"Se han indexado {count} chunks en '{settings.lancedb_table}'.")
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
        print("\nRespuesta:\n")
        print(grounded_answer(result["answer"], citations))
    _print_sources(citations)
    if args.show_top_k:
        _print_top_k(result.get("matches", []))


if __name__ == "__main__":
    main()
