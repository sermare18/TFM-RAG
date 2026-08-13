from __future__ import annotations

import argparse
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from rag_cliente.cli import build_parser, main
from rag_cliente.config import Settings


class CliStreamingTests(unittest.TestCase):
    def test_stream_resolves_citations_from_complete_answer(self) -> None:
        resolved_answers: list[str] = []
        citation = {
            "source": "guia.pdf",
            "source_type": "pdf",
            "page_start": 22,
            "page_end": 22,
            "chunk_index": 76,
            "source_path": "data/pdfs/guia.pdf",
        }

        class FakePipeline:
            def stream_answer(
                self,
                question,
                top_k=None,
                tag=None,
                query_augmentation=False,
            ):
                self.query_augmentation = query_augmentation
                return {
                    "answer_stream": iter(
                        [
                            {"type": "answer", "delta": "Respuesta "},
                            {"type": "answer", "delta": "completa"},
                        ]
                    ),
                    "fallback_stream": lambda: iter(()),
                    "resolve_citations": lambda answer: (
                        resolved_answers.append(answer) or [citation]
                    ),
                    "retrieval_queries": [question, "variante 1", "variante 2"],
                    "query_augmentation_error": None,
                }

        fake = FakePipeline()

        with tempfile.TemporaryDirectory() as tmp_dir:
            settings = Settings(lancedb_uri=str(Path(tmp_dir) / "lancedb"))
            with (
                patch("rag_cliente.cli.get_settings", return_value=settings),
                patch("rag_cliente.cli.RagPipeline", return_value=fake),
                patch.object(
                    sys,
                    "argv",
                    ["rag-cli", "ask", "pregunta", "--stream", "--show-queries"],
                ),
                redirect_stdout(io.StringIO()) as output,
            ):
                main()

        self.assertEqual(resolved_answers, ["Respuesta completa"])
        self.assertIn("guia.pdf", output.getvalue())
        self.assertIn("2. variante 1", output.getvalue())
        self.assertIn("3. variante 2", output.getvalue())
        self.assertTrue(fake.query_augmentation)

    def test_show_top_k_prints_retrieved_page_and_chunk_when_sources_are_empty(self) -> None:
        match = {
            "source": "guia.pdf",
            "source_path": "data/pdfs/guia.pdf",
            "page_start": 19,
            "page_chunk_index": 2,
            "chunk_index": 76,
            "_retrieval_sources": ["vector", "bm25"],
            "_rrf_score": 0.0325,
            "_vector_rank": 1,
            "_distance": 0.3284,
            "_bm25_rank": 3,
            "_bm25_raw_score": 12.4381,
        }

        class FakePipeline:
            def stream_answer(
                self,
                question,
                top_k=None,
                tag=None,
                query_augmentation=False,
            ):
                return {
                    "answer_stream": iter(
                        [{"type": "answer", "delta": "No consta en el documento."}]
                    ),
                    "fallback_stream": lambda: iter(()),
                    "resolve_citations": lambda _answer: [],
                    "matches": [match],
                    "retrieval_queries": [question, "variante 1", "variante 2"],
                    "query_augmentation_error": None,
                }

        with tempfile.TemporaryDirectory() as tmp_dir:
            settings = Settings(lancedb_uri=str(Path(tmp_dir) / "lancedb"))
            with (
                patch("rag_cliente.cli.get_settings", return_value=settings),
                patch("rag_cliente.cli.RagPipeline", return_value=FakePipeline()),
                patch.object(
                    sys,
                    "argv",
                    ["rag-cli", "ask", "pregunta", "--stream", "--show-top-k"],
                ),
                redirect_stdout(io.StringIO()) as output,
            ):
                main()

        rendered = output.getvalue()
        self.assertIn("No retrieved source was selected", rendered)
        self.assertIn("Top-k retrieved pages", rendered)
        self.assertIn("page 19", rendered)
        self.assertIn("page chunk 2", rendered)
        self.assertIn("chunk 76", rendered)
        self.assertIn("sources=vector,bm25", rendered)
        self.assertIn("rrf_score=0.0325", rendered)
        self.assertIn("distance=0.3284", rendered)
        self.assertIn("bm25_score=12.4381", rendered)

    def test_non_streaming_can_disable_augmentation_and_show_original_query(self) -> None:
        class FakePipeline:
            def __init__(self) -> None:
                self.query_augmentation = None

            def ask(
                self,
                question,
                top_k=None,
                tag=None,
                query_augmentation=False,
            ):
                self.query_augmentation = query_augmentation
                return {
                    "answer": "respuesta",
                    "reasoning": "no debe mostrarse",
                    "citations": [],
                    "matches": [],
                    "retrieval_queries": [question],
                    "query_augmentation_error": None,
                }

        fake = FakePipeline()
        with tempfile.TemporaryDirectory() as tmp_dir:
            settings = Settings(lancedb_uri=str(Path(tmp_dir) / "lancedb"))
            with (
                patch("rag_cliente.cli.get_settings", return_value=settings),
                patch("rag_cliente.cli.RagPipeline", return_value=fake),
                patch.object(
                    sys,
                    "argv",
                    [
                        "rag-cli",
                        "ask",
                        "pregunta original",
                        "--no-query-augmentation",
                        "--show-queries",
                    ],
                ),
                redirect_stdout(io.StringIO()) as output,
            ):
                main()

        self.assertFalse(fake.query_augmentation)
        rendered = output.getvalue()
        self.assertIn("Retrieval queries", rendered)
        self.assertIn("1. pregunta original", rendered)
        self.assertNotIn("Reasoning", rendered)
        self.assertNotIn("no debe mostrarse", rendered)

    def test_augmentation_failure_warns_and_returns_grounded_refusal(self) -> None:
        class FakePipeline:
            def ask(self, question, **_kwargs):
                return {
                    "answer": "respuesta disponible",
                    "reasoning": "",
                    "citations": [],
                    "matches": [],
                    "retrieval_queries": [question],
                    "query_augmentation_error": "JSON invalido",
                }

        with tempfile.TemporaryDirectory() as tmp_dir:
            settings = Settings(lancedb_uri=str(Path(tmp_dir) / "lancedb"))
            with (
                patch("rag_cliente.cli.get_settings", return_value=settings),
                patch("rag_cliente.cli.RagPipeline", return_value=FakePipeline()),
                patch.object(sys, "argv", ["rag-cli", "ask", "pregunta"]),
                redirect_stdout(io.StringIO()) as output,
                redirect_stderr(io.StringIO()) as errors,
            ):
                main()

        self.assertIn("No consta en los documentos recuperados", output.getvalue())
        self.assertNotIn("respuesta disponible", output.getvalue())
        self.assertIn("query augmentation failed", errors.getvalue())
        self.assertIn("JSON invalido", errors.getvalue())

    def test_show_reasoning_is_no_longer_a_valid_cli_option(self) -> None:
        parser = build_parser()
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
            parser.parse_args(["ask", "pregunta", "--show-reasoning"])
        self.assertEqual(raised.exception.code, 2)

    def test_help_keeps_long_option_and_description_on_the_same_line(self) -> None:
        parser = build_parser()
        ask_parser = next(
            action.choices["ask"]
            for action in parser._actions
            if isinstance(action, argparse._SubParsersAction)
        )
        help_text = ask_parser.format_help()
        option_line = next(
            line
            for line in help_text.splitlines()
            if line.lstrip().startswith("--no-query-augmentation")
        )
        self.assertTrue(help_text.startswith("usage: rag.bat ask"))
        self.assertIn("Retrieve only with the original question.", option_line)


if __name__ == "__main__":
    unittest.main()
