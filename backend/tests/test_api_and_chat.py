from __future__ import annotations

import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from rag_cliente.api import create_app
from rag_cliente.config import Settings
from rag_cliente.llm_client import LlamaCppClient
from rag_cliente.pipeline import RagPipeline, UNGROUNDED_ANSWER, grounded_answer


class FakePipeline:
    def __init__(self) -> None:
        self.ask_calls: list[dict] = []
        self.index_calls: list[dict] = []

    def ask(self, question, top_k=None, messages=None, tag=None, enable_reasoning=False):
        self.ask_calls.append(
            {
                "question": question,
                "top_k": top_k,
                "messages": messages,
                "tag": tag,
                "enable_reasoning": enable_reasoning,
            }
        )
        return {"answer": "Ana", "reasoning": "", "citations": [], "matches": []}

    def index_documents(self, doc_dir, tag=None, refresh_bedrock=False):
        self.index_calls.append(
            {
                "doc_dir": doc_dir,
                "tag": tag,
                "refresh_bedrock": refresh_bedrock,
            }
        )
        return 7

    def stream_answer(self, question, top_k=None, messages=None, tag=None, enable_reasoning=False):
        return {
            "answer_stream": iter([{"type": "answer", "delta": "Ana"}]),
            "fallback_stream": lambda: iter(()),
            "resolve_citations": lambda _answer: [],
            "matches": [],
        }


class ApiTests(unittest.TestCase):
    def build_app(self, root: Path):
        settings = Settings(
            documents_dir=str(root / "documents"),
            lancedb_uri=str(root / "lance"),
            bm25_index_dir=str(root / "bm25"),
            model_supervision_enabled=False,
        )
        with patch("rag_cliente.api.get_settings", return_value=settings):
            app = create_app()
        fake = FakePipeline()
        app.state.pipeline = fake
        return app, fake

    def test_files_endpoint_exposes_only_pdf_and_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            app, _fake = self.build_app(root)
            documents = root / "documents"
            (documents / "manual.pdf").write_bytes(b"%PDF")
            (documents / "notes.md").write_text("hola", encoding="utf-8")
            (documents / "ignored.txt").write_text("no", encoding="utf-8")
            with TestClient(app) as client:
                payload = client.get("/files").json()
            self.assertEqual([item["name"] for item in payload["files"]], ["manual.pdf", "notes.md"])

    def test_upload_accepts_markdown_and_rejects_txt(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            app, _fake = self.build_app(root)
            with TestClient(app) as client:
                accepted = client.post(
                    "/files/upload",
                    files={"file": ("notes.md", b"# Notes", "text/markdown")},
                )
                rejected = client.post(
                    "/files/upload",
                    files={"file": ("notes.txt", b"Notes", "text/plain")},
                )
            self.assertEqual(accepted.status_code, 201)
            self.assertEqual(rejected.status_code, 400)
            self.assertIn("Tipo de archivo no compatible", rejected.json()["detail"])

    def test_index_forwards_bedrock_refresh_without_calling_aws(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            app, fake = self.build_app(root)
            with TestClient(app) as client:
                response = client.post(
                    "/index",
                    json={
                        "doc_dir": str(root / "documents"),
                        "tag": "guias",
                        "refresh_bedrock": True,
                    },
                )
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["indexed_chunks"], 7)
            self.assertTrue(fake.index_calls[0]["refresh_bedrock"])

    def test_chat_session_persists_history(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            app, fake = self.build_app(Path(temp))
            with TestClient(app) as client:
                first = client.post("/ask", json={"question": "Me llamo Ana"})
                session_id = first.json()["session_id"]
                second = client.post(
                    "/ask",
                    json={"question": "Como me llamo?", "session_id": session_id},
                )
            self.assertEqual(second.status_code, 200)
            self.assertEqual(
                fake.ask_calls[1]["messages"],
                [
                    {"role": "user", "content": "Me llamo Ana"},
                    {"role": "assistant", "content": "Ana"},
                ],
            )

    def test_stream_emits_session_answer_and_done(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            app, _fake = self.build_app(Path(temp))
            with TestClient(app) as client:
                response = client.post("/ask/stream", json={"question": "Nombre?"})
            events = [json.loads(line) for line in response.text.splitlines()]
            self.assertEqual(events[0]["type"], "session")
            self.assertIn({"type": "answer", "delta": UNGROUNDED_ANSWER}, events)
            self.assertNotIn({"type": "answer", "delta": "Ana"}, events)
            self.assertEqual(events[-1]["type"], "done")


class LocalChatClientTests(unittest.TestCase):
    def test_query_embeddings_include_instruction_but_documents_do_not(self) -> None:
        client = LlamaCppClient(Settings(model_supervision_enabled=False))
        client.embedding_client = Mock()
        client.embedding_client.embeddings.create.return_value = types.SimpleNamespace(
            data=[types.SimpleNamespace(embedding=[0.1, 0.2])]
        )

        client.embed_texts(["normativa de defensa"], query_mode=True)
        query_input = client.embedding_client.embeddings.create.call_args.kwargs["input"]
        self.assertTrue(query_input[0].startswith("Instruct: "))
        self.assertTrue(query_input[0].endswith("Query: normativa de defensa"))

        client.embed_texts(["contenido del documento"])
        document_input = client.embedding_client.embeddings.create.call_args.kwargs["input"]
        self.assertEqual(document_input, ["contenido del documento"])

    def test_prompt_keeps_history_and_page_context(self) -> None:
        messages = LlamaCppClient._build_messages(
            "Como me llamo?",
            ["[S1] notes.md p.1\nAna"],
            messages=[
                {"role": "user", "content": "Me llamo Ana"},
                {"role": "assistant", "content": "Hola Ana"},
            ],
        )
        self.assertIn("[S1] notes.md p.1", messages[-1]["content"])
        self.assertEqual(messages[1]["content"], "Me llamo Ana")
        self.assertIn("strictly document-grounded", messages[0]["content"])
        self.assertIn("Never use prior knowledge", messages[0]["content"])
        self.assertIn("No consta en los documentos recuperados", messages[-1]["content"])

    def test_answer_without_verified_citations_is_rejected(self) -> None:
        self.assertEqual(grounded_answer("codigo de tres en raya", []), UNGROUNDED_ANSWER)
        self.assertEqual(
            grounded_answer("dato respaldado", [{"source_id": "S1"}]),
            "dato respaldado",
        )

    def test_reasoning_stream_falls_back_to_non_reasoning_answer(self) -> None:
        client = LlamaCppClient(Settings(model_supervision_enabled=False))
        client.chat_client = Mock()
        reasoning = types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    delta=types.SimpleNamespace(content=None, reasoning_content="analisis"),
                    text=None,
                )
            ]
        )
        answer = types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    delta=types.SimpleNamespace(content="respuesta", reasoning_content=None),
                    text=None,
                )
            ]
        )
        client.chat_client.chat.completions.create.side_effect = [iter([reasoning]), iter([answer])]
        events = list(client.stream_answer("pregunta", ["contexto"], enable_reasoning=True))
        self.assertEqual(events[-1], {"type": "answer", "delta": "respuesta"})


class RetrievalModeTests(unittest.TestCase):
    def make_pipeline(self, root: Path, mode: str) -> RagPipeline:
        settings = Settings(
            retrieval_mode=mode,
            lancedb_uri=str(root / "lance"),
            bm25_index_dir=str(root / "bm25"),
            model_supervision_enabled=False,
        )
        return RagPipeline(settings)

    def test_bm25_mode_does_not_request_query_embeddings(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            pipeline = self.make_pipeline(Path(temp), "bm25")
            pipeline.client = Mock()
            pipeline.client.embed_texts.side_effect = AssertionError("embeddings not expected")
            pipeline.bm25_store = Mock()
            pipeline.bm25_store.search.return_value = []
            self.assertEqual(pipeline._retrieve(["consulta"], top_k=3, tag=None), [])
            pipeline.client.embed_texts.assert_not_called()

    def test_vector_mode_does_not_request_bm25(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            pipeline = self.make_pipeline(Path(temp), "vector")
            pipeline.client = Mock()
            pipeline.client.embed_texts.return_value = [[0.1, 0.2]]
            pipeline.store = Mock()
            pipeline.store.search.return_value = []
            pipeline.bm25_store = Mock()
            self.assertEqual(pipeline._retrieve(["consulta"], top_k=3, tag="guias"), [])
            pipeline.client.embed_texts.assert_called_once_with(
                ["consulta"],
                query_mode=True,
            )
            pipeline.bm25_store.search.assert_not_called()
            pipeline.store.search.assert_called_once_with([0.1, 0.2], top_k=40, tag="guias")

    def test_generation_inputs_add_two_augmented_queries_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            pipeline = self.make_pipeline(Path(temp), "hybrid")
            pipeline.client = Mock()
            pipeline.client.rewrite_question_for_retrieval.return_value = "pregunta"
            pipeline.generate_query_variants = Mock(
                return_value={"pregunta": ["variante uno", "variante dos"]}
            )
            pipeline._retrieve = Mock(return_value=[])

            inputs = pipeline._prepare_generation_inputs(
                "pregunta",
                top_k=5,
                query_augmentation=True,
            )

            self.assertEqual(
                inputs["retrieval_queries"],
                ["pregunta", "variante uno", "variante dos"],
            )
            self.assertEqual(inputs["query_variants"], ["variante uno", "variante dos"])
            self.assertIsNone(inputs["query_augmentation_error"])
            pipeline._retrieve.assert_called_once_with(
                ["pregunta", "variante uno", "variante dos"],
                top_k=5,
                tag=None,
                retrieval_mode=None,
                distance_type=None,
                use_query_instruction=None,
            )

    def test_generation_inputs_fall_back_when_augmentation_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            pipeline = self.make_pipeline(Path(temp), "hybrid")
            pipeline.client = Mock()
            pipeline.client.rewrite_question_for_retrieval.return_value = "pregunta"
            pipeline.generate_query_variants = Mock(side_effect=RuntimeError("salida invalida"))
            pipeline._retrieve = Mock(return_value=[])

            inputs = pipeline._prepare_generation_inputs(
                "pregunta",
                top_k=5,
                query_augmentation=True,
            )

            self.assertEqual(inputs["retrieval_queries"], ["pregunta"])
            self.assertEqual(inputs["query_variants"], [])
            self.assertEqual(inputs["query_augmentation_error"], "salida invalida")
            pipeline._retrieve.assert_called_once_with(
                ["pregunta"],
                top_k=5,
                tag=None,
                retrieval_mode=None,
                distance_type=None,
                use_query_instruction=None,
            )

    def test_generation_inputs_forward_explicit_retrieval_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            pipeline = self.make_pipeline(Path(temp), "hybrid")
            pipeline.client = Mock()
            pipeline.client.rewrite_question_for_retrieval.return_value = "pregunta"
            pipeline._retrieve = Mock(return_value=[])

            pipeline._prepare_generation_inputs(
                "pregunta",
                top_k=9,
                retrieval_mode="vector",
                distance_type="cosine",
                use_query_instruction=True,
            )

            pipeline._retrieve.assert_called_once_with(
                ["pregunta"],
                top_k=9,
                tag=None,
                retrieval_mode="vector",
                distance_type="cosine",
                use_query_instruction=True,
            )


if __name__ == "__main__":
    unittest.main()
