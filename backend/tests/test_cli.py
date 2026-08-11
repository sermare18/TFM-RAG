from __future__ import annotations

import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from rag_cliente.cli import main
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
            def stream_answer(self, question, top_k=None, tag=None, enable_reasoning=False):
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
                }

        with tempfile.TemporaryDirectory() as tmp_dir:
            settings = Settings(lancedb_uri=str(Path(tmp_dir) / "lancedb"))
            with (
                patch("rag_cliente.cli.get_settings", return_value=settings),
                patch("rag_cliente.cli.RagPipeline", return_value=FakePipeline()),
                patch.object(sys, "argv", ["rag-cli", "ask", "pregunta", "--stream"]),
                redirect_stdout(io.StringIO()) as output,
            ):
                main()

        self.assertEqual(resolved_answers, ["Respuesta completa"])
        self.assertIn("guia.pdf", output.getvalue())


if __name__ == "__main__":
    unittest.main()
