import json
import shutil
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from uuid import uuid4

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

fake_indexer = types.ModuleType("rag_cliente.indexer")
fake_pdf_loader = types.ModuleType("rag_cliente.pdf_loader")
fake_vector_store = types.ModuleType("rag_cliente.vector_store")
fake_rank_bm25 = types.ModuleType("rank_bm25")
fake_openai = types.ModuleType("openai")


class _FakePdfChunker:
    def __init__(self, settings) -> None:
        self.settings = settings

    def chunk_pages(self, pages, tag=None):
        return []


class _FakeLanceDBStore:
    def __init__(self, path, table) -> None:
        self.path = path
        self.table = table

    def replace_chunks(self, chunks, embeddings):
        return None

    def search(self, query_vector, top_k=2, tag=None):
        return []


class _FakeBM25Okapi:
    def __init__(self, tokenized_corpus) -> None:
        self.tokenized_corpus = tokenized_corpus

    def get_scores(self, query_tokens):
        return [0.0 for _ in self.tokenized_corpus]


fake_indexer.PdfChunker = _FakePdfChunker
fake_pdf_loader.IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
fake_pdf_loader.SUPPORTED_DOCUMENT_SUFFIXES = {
    ".pdf", ".docx", ".pptx", ".xlsx", ".epub", ".html", ".txt",
    *fake_pdf_loader.IMAGE_SUFFIXES,
}
fake_pdf_loader.load_documents_from_directory = lambda doc_dir: []
fake_vector_store.LanceDBStore = _FakeLanceDBStore
fake_rank_bm25.BM25Okapi = _FakeBM25Okapi
fake_openai.OpenAI = Mock
fake_openai.APITimeoutError = type("APITimeoutError", (Exception,), {})
fake_openai.RateLimitError = type("RateLimitError", (Exception,), {})
fake_openai.APIConnectionError = type("APIConnectionError", (Exception,), {})
fake_openai.BadRequestError = type("BadRequestError", (Exception,), {})

sys.modules.setdefault("rag_cliente.indexer", fake_indexer)
sys.modules.setdefault("rag_cliente.pdf_loader", fake_pdf_loader)
sys.modules.setdefault("rag_cliente.vector_store", fake_vector_store)
sys.modules.setdefault("rank_bm25", fake_rank_bm25)
sys.modules.setdefault("openai", fake_openai)

from rag_cliente.api import create_app
from rag_cliente.config import Settings
from rag_cliente.llm_client import LlamaCppClient
from rag_cliente.pipeline import RagPipeline


class ApiFileEndpointsTests(unittest.TestCase):
    def _temporary_documents_dir(self):
        base_dir = Path(__file__).resolve().parents[1] / "data"
        base_dir.mkdir(parents=True, exist_ok=True)
        documents_dir = base_dir / f"test-docs-{uuid4().hex}"
        documents_dir.mkdir()

        class TemporaryDocumentsDir:
            def __enter__(self):
                return str(documents_dir)

            def __exit__(self, exc_type, exc, traceback):
                shutil.rmtree(documents_dir, ignore_errors=True)

        return TemporaryDocumentsDir()

    def test_list_files_returns_supported_documents(self) -> None:
        with self._temporary_documents_dir() as tmp_dir:
            documents_dir = Path(tmp_dir)
            (documents_dir / "manual.pdf").write_bytes(b"%PDF-1.4")
            (documents_dir / "notes.txt").write_text("hola", encoding="utf-8")
            (documents_dir / "ignore.bin").write_bytes(b"\x00\x01")

            settings = Settings(documents_dir=str(documents_dir))
            with patch("rag_cliente.api.get_settings", return_value=settings):
                app = create_app()

            with TestClient(app) as client:
                response = client.get("/files")

                self.assertEqual(response.status_code, 200)
                payload = response.json()
                self.assertEqual(payload["directory"], str(documents_dir.resolve()))
                self.assertEqual([item["name"] for item in payload["files"]], ["manual.pdf", "notes.txt"])
                self.assertEqual(payload["files"][0]["download_url"], "/files/manual.pdf")

    def test_upload_and_download_file_from_frontend(self) -> None:
        with self._temporary_documents_dir() as tmp_dir:
            documents_dir = Path(tmp_dir)
            settings = Settings(documents_dir=str(documents_dir))
            with patch("rag_cliente.api.get_settings", return_value=settings):
                app = create_app()

            with TestClient(app) as client:
                upload_response = client.post(
                    "/files/upload",
                    files={"file": ("chat.txt", b"hola frontend", "text/plain")},
                )

                self.assertEqual(upload_response.status_code, 201)
                self.assertTrue((documents_dir / "chat.txt").exists())
                self.assertEqual((documents_dir / "chat.txt").read_text(encoding="utf-8"), "hola frontend")

                download_response = client.get("/files/chat.txt")
                self.assertEqual(download_response.status_code, 200)
                self.assertEqual(download_response.text, "hola frontend")

    def test_upload_with_tag_stores_file_in_tagged_subdirectory(self) -> None:
        with self._temporary_documents_dir() as tmp_dir:
            documents_dir = Path(tmp_dir)
            settings = Settings(documents_dir=str(documents_dir))
            with patch("rag_cliente.api.get_settings", return_value=settings):
                app = create_app()

            with TestClient(app) as client:
                upload_response = client.post(
                    "/files/upload",
                    data={"tag": "confidencial"},
                    files={"file": ("contrato.pdf", b"%PDF-1.4", "application/pdf")},
                )

                self.assertEqual(upload_response.status_code, 201)
                self.assertEqual(upload_response.json()["tag"], "confidencial")
                self.assertTrue((documents_dir / "confidencial" / "contrato.pdf").exists())

                list_response = client.get("/files")
                self.assertEqual(list_response.status_code, 200)
                self.assertEqual(list_response.json()["files"][0]["relative_path"], "confidencial/contrato.pdf")
                self.assertEqual(list_response.json()["files"][0]["tag"], "confidencial")
                self.assertEqual(list_response.json()["files"][0]["download_url"], "/files/confidencial/contrato.pdf")

                download_response = client.get("/files/confidencial/contrato.pdf")
                self.assertEqual(download_response.status_code, 200)
                self.assertEqual(download_response.content, b"%PDF-1.4")

    def test_upload_rejects_conflicts_and_unsupported_extensions(self) -> None:
        with self._temporary_documents_dir() as tmp_dir:
            documents_dir = Path(tmp_dir)
            (documents_dir / "chat.txt").write_text("anterior", encoding="utf-8")
            settings = Settings(documents_dir=str(documents_dir))
            with patch("rag_cliente.api.get_settings", return_value=settings):
                app = create_app()

            with TestClient(app) as client:
                conflict_response = client.post(
                    "/files/upload",
                    files={"file": ("chat.txt", b"nuevo", "text/plain")},
                )
                self.assertEqual(conflict_response.status_code, 409)

                unsupported_response = client.post(
                    "/files/upload",
                    files={"file": ("malware.exe", b"boom", "application/octet-stream")},
                )
                self.assertEqual(unsupported_response.status_code, 400)


class ApiChatHistoryTests(unittest.TestCase):
    def test_ask_endpoint_creates_session_and_persists_history(self) -> None:
        app = create_app()

        class FakePipeline:
            def __init__(self) -> None:
                self.calls = []

            def ask(self, question, top_k=None, messages=None, tag=None, enable_reasoning=False):
                self.calls.append(
                    {
                        "question": question,
                        "top_k": top_k,
                        "messages": messages,
                        "tag": tag,
                    }
                )
                return {
                    "answer": "Ana",
                    "reasoning": "",
                    "citations": [],
                    "matches": [],
                }

        fake_pipeline = FakePipeline()
        app.state.pipeline = fake_pipeline
        client = TestClient(app)

        first_response = client.post(
            "/ask",
            json={
                "question": "Me llamo Ana",
                "top_k": 3,
            },
        )
        self.assertEqual(first_response.status_code, 200)
        session_id = first_response.json()["session_id"]
        self.assertTrue(session_id)
        self.assertEqual(
            fake_pipeline.calls[0],
            {
                "question": "Me llamo Ana",
                "top_k": 3,
                "messages": [],
                "tag": None,
            },
        )

        second_response = client.post(
            "/ask",
            json={
                "question": "Como me llamo?",
                "top_k": 3,
                "session_id": session_id,
            },
        )

        self.assertEqual(second_response.status_code, 200)
        self.assertEqual(second_response.json()["session_id"], session_id)
        self.assertEqual(
            fake_pipeline.calls[1],
            {
                "question": "Como me llamo?",
                "top_k": 3,
                "messages": [
                    {"role": "user", "content": "Me llamo Ana"},
                    {"role": "assistant", "content": "Ana"},
                ],
                "tag": None,
            },
        )

    def test_ask_endpoint_forwards_messages_to_pipeline(self) -> None:
        app = create_app()

        class FakePipeline:
            def __init__(self) -> None:
                self.called_with = None

            def ask(self, question, top_k=None, messages=None, tag=None, enable_reasoning=False):
                self.called_with = {
                    "question": question,
                    "top_k": top_k,
                    "messages": messages,
                    "tag": tag,
                }
                return {
                    "answer": "Ana",
                    "reasoning": "",
                    "citations": [],
                    "matches": [],
                }

        fake_pipeline = FakePipeline()
        app.state.pipeline = fake_pipeline
        client = TestClient(app)

        payload = {
            "question": "Como me llamo?",
            "top_k": 3,
            "messages": [
                {"role": "user", "content": "Me llamo Ana"},
                {"role": "assistant", "content": "Hola Ana"},
                {"role": "user", "content": "Como me llamo?"},
            ],
        }
        response = client.post("/ask", json=payload)

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["session_id"])
        self.assertEqual(
            fake_pipeline.called_with,
            {
                "question": "Como me llamo?",
                "top_k": 3,
                "messages": payload["messages"],
                "tag": None,
            },
        )

    def test_ask_endpoint_forwards_tag_to_pipeline(self) -> None:
        app = create_app()

        class FakePipeline:
            def __init__(self) -> None:
                self.called_with = None

            def ask(self, question, top_k=None, messages=None, tag=None, enable_reasoning=False):
                self.called_with = {
                    "question": question,
                    "top_k": top_k,
                    "messages": messages,
                    "tag": tag,
                }
                return {
                    "answer": "Resumen",
                    "reasoning": "",
                    "citations": [],
                    "matches": [],
                }

        fake_pipeline = FakePipeline()
        app.state.pipeline = fake_pipeline
        client = TestClient(app)

        response = client.post(
            "/ask",
            json={
                "question": "Resume el contrato",
                "top_k": 3,
                "tag": "confidencial",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            fake_pipeline.called_with,
            {
                "question": "Resume el contrato",
                "top_k": 3,
                "messages": [],
                "tag": "confidencial",
            },
        )

    def test_ask_endpoint_accepts_tags_array_for_single_tag_filter(self) -> None:
        app = create_app()

        class FakePipeline:
            def __init__(self) -> None:
                self.called_with = None

            def ask(self, question, top_k=None, messages=None, tag=None, enable_reasoning=False):
                self.called_with = tag
                return {
                    "answer": "Resumen",
                    "reasoning": "",
                    "citations": [],
                    "matches": [],
                }

        fake_pipeline = FakePipeline()
        app.state.pipeline = fake_pipeline
        client = TestClient(app)

        response = client.post(
            "/ask",
            json={
                "question": "Resume el contrato",
                "tags": ["", "legal"],
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(fake_pipeline.called_with, "legal")

    def test_stream_endpoint_forwards_messages_to_pipeline_and_emits_session(self) -> None:
        app = create_app()

        class FakePipeline:
            def __init__(self) -> None:
                self.called_with = None

            def stream_answer(
                self,
                question,
                top_k=None,
                messages=None,
                tag=None,
                enable_reasoning=False,
            ):
                self.called_with = {
                    "question": question,
                    "top_k": top_k,
                    "messages": messages,
                    "tag": tag,
                }
                return {
                    "answer_stream": iter(()),
                    "fallback_stream": lambda: iter(
                        [{"type": "answer", "delta": "Ana"}]
                    ),
                    "resolve_citations": lambda answer: [],
                    "matches": [],
                }

        fake_pipeline = FakePipeline()
        app.state.pipeline = fake_pipeline
        client = TestClient(app)

        payload = {
            "question": "Como me llamo?",
            "top_k": 3,
            "messages": [
                {"role": "user", "content": "Me llamo Ana"},
                {"role": "assistant", "content": "Hola Ana"},
                {"role": "user", "content": "Como me llamo?"},
            ],
        }
        response = client.post("/ask/stream", json=payload)

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.headers["x-session-id"])
        self.assertEqual(
            fake_pipeline.called_with,
            {
                "question": "Como me llamo?",
                "top_k": 3,
                "messages": payload["messages"],
                "tag": None,
            },
        )
        events = [json.loads(line) for line in response.text.strip().splitlines()]
        self.assertEqual(events[0], {"type": "session", "session_id": response.headers["x-session-id"]})
        self.assertEqual(
            events[1],
            {"type": "fallback", "reason": "reasoning_finished_without_answer"},
        )
        self.assertEqual(events[2], {"type": "answer", "delta": "Ana"})

    def test_create_and_delete_session_endpoints(self) -> None:
        app = create_app()
        client = TestClient(app)

        create_response = client.post("/sessions")
        self.assertEqual(create_response.status_code, 200)
        session_id = create_response.json()["session_id"]
        self.assertTrue(session_id)

        delete_response = client.delete(f"/sessions/{session_id}")
        self.assertEqual(delete_response.status_code, 204)

        missing_response = client.delete(f"/sessions/{session_id}")
        self.assertEqual(missing_response.status_code, 404)

    def test_ask_returns_404_for_unknown_session(self) -> None:
        app = create_app()
        client = TestClient(app)

        response = client.post(
            "/ask",
            json={
                "question": "Como me llamo?",
                "session_id": "missing-session",
            },
        )

        self.assertEqual(response.status_code, 404)


class PromptConstructionTests(unittest.TestCase):
    def test_reasoning_stream_continues_with_streamed_final_answer(self) -> None:
        client = LlamaCppClient(Settings())
        client.chat_client = Mock()

        reasoning_chunk = types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    delta=types.SimpleNamespace(content=None, reasoning_content="analisis"),
                    text=None,
                )
            ]
        )
        answer_chunk = types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    delta=types.SimpleNamespace(content="respuesta", reasoning_content=None),
                    text=None,
                )
            ]
        )
        client.chat_client.chat.completions.create.side_effect = [
            iter([reasoning_chunk]),
            iter([answer_chunk]),
        ]

        events = list(
            client.stream_answer(
                "pregunta",
                ["contexto"],
                enable_reasoning=True,
            )
        )

        self.assertEqual(
            events,
            [
                {"type": "reasoning", "delta": "analisis"},
                {"type": "answer", "delta": "respuesta"},
            ],
        )
        calls = client.chat_client.chat.completions.create.call_args_list
        self.assertEqual(calls[0].kwargs["extra_body"]["chat_template_kwargs"]["enable_thinking"], True)
        self.assertEqual(calls[1].kwargs["extra_body"]["chat_template_kwargs"]["enable_thinking"], False)

    def test_build_messages_includes_history_and_retrieved_context(self) -> None:
        question = "Como me llamo?"
        history = [
            {"role": "user", "content": "Me llamo Ana"},
            {"role": "assistant", "content": "Hola Ana"},
            {"role": "user", "content": question},
        ]

        messages = LlamaCppClient._build_messages(
            question,
            ["[chat.txt p.1-1]\nAna dijo que se llama Ana."],
            messages=history,
        )

        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1], history[0])
        self.assertEqual(messages[2], history[1])
        self.assertEqual(messages[-1]["role"], "user")
        self.assertIn("Latest question:\nComo me llamo?", messages[-1]["content"])
        self.assertIn("Retrieved context:\n[chat.txt p.1-1]", messages[-1]["content"])
        self.assertEqual(sum(msg["content"] == question for msg in messages), 0)

    def test_build_rewrite_messages_includes_history(self) -> None:
        history = [
            {"role": "user", "content": "Que nomenclatura usan los ordenadores windows al asignarlos?"},
            {"role": "assistant", "content": "Usan LTxxxx o PCxxxx."},
        ]

        messages = LlamaCppClient._build_rewrite_messages("y en mac?", history)

        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1], history[0])
        self.assertEqual(messages[2], history[1])
        self.assertEqual(messages[-1]["role"], "user")
        self.assertIn("y en mac?", messages[-1]["content"])


class PipelineChatHistoryTests(unittest.TestCase):
    def test_build_retrieval_queries_adds_contextual_variant(self) -> None:
        queries = RagPipeline._build_retrieval_queries(
            "y en mac?",
            "Que nomenclatura usan los ordenadores mac al asignarlos?",
            messages=[
                {"role": "user", "content": "Que nomenclatura usan los ordenadores windows al asignarlos?"},
                {"role": "assistant", "content": "Usan LTXXXX o PCXXXX."},
            ],
        )

        self.assertEqual(queries[0], "Que nomenclatura usan los ordenadores mac al asignarlos?")
        self.assertEqual(queries[1], "y en mac?")
        self.assertIn("Previous user question:", queries[2])
        self.assertIn("Latest follow-up question: y en mac?", queries[2])

    def test_merge_matches_keeps_best_distance_per_chunk(self) -> None:
        merged = RagPipeline._merge_matches(
            [
                [
                    {"document_id": "doc-1", "source_path": "/tmp/a", "chunk_index": 0, "_distance": 0.9},
                    {"document_id": "doc-2", "source_path": "/tmp/b", "chunk_index": 1, "_distance": 0.7},
                ],
                [
                    {"document_id": "doc-1", "source_path": "/tmp/a", "chunk_index": 0, "_distance": 0.4},
                    {"document_id": "doc-3", "source_path": "/tmp/c", "chunk_index": 2, "_distance": 0.8},
                ],
            ],
            top_k=3,
        )

        self.assertEqual(len(merged), 3)
        self.assertEqual(merged[0]["document_id"], "doc-1")
        self.assertEqual(merged[0]["_distance"], 0.4)

    def test_ask_and_stream_share_generation_inputs(self) -> None:
        settings = Settings(
            hybrid_search_enabled=False,
            model_supervision_enabled=False,
        )
        pipeline = RagPipeline(settings)

        history = [
            {"role": "user", "content": "Me llamo Ana"},
            {"role": "assistant", "content": "Hola Ana"},
        ]
        match = {
            "document_id": "doc-1",
            "source": "chat.txt",
            "source_path": "/tmp/chat.txt",
            "source_type": "txt",
            "page_start": 1,
            "page_end": 1,
            "chunk_index": 0,
            "text": "Ana dijo que se llama Ana.",
        }

        pipeline.client = Mock()
        pipeline.store = Mock()
        pipeline.client.rewrite_question_for_retrieval.return_value = "Como me llamo?"
        pipeline.client.embed_texts.return_value = [[0.1, 0.2], [0.2, 0.3], [0.3, 0.4]]
        pipeline.store.search.return_value = [match]
        pipeline.client.generate_answer.return_value = {"answer": "Ana", "reasoning": ""}
        pipeline.client.stream_answer.return_value = iter([{"type": "answer", "delta": "Ana"}])
        pipeline.client.select_used_source_ids.return_value = ["S1"]

        ask_result = pipeline.ask("Como me llamo?", top_k=3, messages=history)
        stream_result = pipeline.stream_answer("Como me llamo?", top_k=3, messages=history)

        expected_context = ["[S1] chat.txt p.1\nAna dijo que se llama Ana."]

        pipeline.client.rewrite_question_for_retrieval.assert_any_call(
            "Como me llamo?",
            messages=history,
        )
        pipeline.client.embed_texts.assert_any_call(
            [
                "Como me llamo?",
                "Previous user question: Me llamo Ana\nLatest follow-up question: Como me llamo?",
            ]
        )
        pipeline.client.generate_answer.assert_any_call(
            "Como me llamo?",
            expected_context,
            messages=history,
            enable_reasoning=False,
        )
        pipeline.client.stream_answer.assert_called_once_with(
            "Como me llamo?",
            expected_context,
            messages=history,
            enable_reasoning=False,
        )
        self.assertIn("fallback_stream", stream_result)
        self.assertEqual(ask_result["citations"][0]["document_id"], "doc-1")
        self.assertEqual(stream_result["resolve_citations"]("Ana")[0]["document_id"], "doc-1")

    def test_prepare_generation_inputs_rewrites_follow_up_for_retrieval(self) -> None:
        settings = Settings(
            hybrid_search_enabled=False,
            model_supervision_enabled=False,
        )
        pipeline = RagPipeline(settings)

        history = [
            {"role": "user", "content": "Que nomenclatura usan los ordenadores windows al asignarlos?"},
            {"role": "assistant", "content": "Usan LTxxxx o PCxxxx."},
        ]

        pipeline.client = Mock()
        pipeline.store = Mock()
        pipeline.client.rewrite_question_for_retrieval.return_value = (
            "Que nomenclatura usan los ordenadores mac al asignarlos?"
        )
        pipeline.client.embed_texts.return_value = [[0.3, 0.4], [0.4, 0.5], [0.5, 0.6]]
        pipeline.store.search.return_value = []

        pipeline._prepare_generation_inputs("y en mac?", top_k=3, messages=history)

        pipeline.client.rewrite_question_for_retrieval.assert_called_once_with(
            "y en mac?",
            messages=history,
        )
        pipeline.client.embed_texts.assert_called_once_with(
            [
                "Que nomenclatura usan los ordenadores mac al asignarlos?",
                "y en mac?",
                (
                    "Previous user question: Que nomenclatura usan los ordenadores windows al asignarlos?\n"
                    "Latest follow-up question: y en mac?"
                ),
            ]
        )

    def test_prepare_generation_inputs_passes_tag_to_vector_search(self) -> None:
        settings = Settings(model_supervision_enabled=False)
        pipeline = RagPipeline(settings)

        pipeline.client = Mock()
        pipeline.store = Mock()
        pipeline.bm25_store = Mock()
        pipeline.client.rewrite_question_for_retrieval.return_value = "Resume el contrato"
        pipeline.client.embed_texts.return_value = [[0.1, 0.2]]
        pipeline.store.search.return_value = []
        pipeline.bm25_store.search.return_value = []

        pipeline._prepare_generation_inputs("Resume el contrato", top_k=3, tag="confidencial")

        pipeline.store.search.assert_called_once_with(
            [0.1, 0.2],
            top_k=40,
            tag="confidencial",
        )
        pipeline.bm25_store.search.assert_called_once_with(
            "Resume el contrato",
            top_k=40,
            tag="confidencial",
        )


if __name__ == "__main__":
    unittest.main()
