from __future__ import annotations

import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

from rag_cliente.bedrock_parser import MarkdownDocument, MarkdownPage
from rag_cliente.bm25_store import BM25Store
from rag_cliente.config import Settings
from rag_cliente.indexer import ChunkRecord
from rag_cliente.model_manifest import check_models, get_role, roles_for_profile
from rag_cliente.model_supervisor import ModelServerSpec, ModelSupervisor, build_server_specs
from rag_cliente.pipeline import RagPipeline
from rag_cliente.resource_coordinator import ResourceCoordinator
from rag_cliente.vector_store import LanceDBStore


def make_chunk(index: int, text: str, page: int | None = None) -> ChunkRecord:
    page = page or index + 1
    return ChunkRecord(
        id=f"chunk-{index}",
        chunk_id=f"chunk-{index}",
        document_id="doc-1",
        text=text,
        source="doc.md",
        source_path="C:/docs/doc.md",
        source_type="md",
        page_start=page,
        page_end=page,
        source_pages=[page],
        chunk_index=index,
        page_chunk_index=0,
        token_count=len(text.split()),
        oversize=False,
        parser_model="direct-markdown",
        prompt_version="none",
        source_sha256="hash",
        schema_version=3,
        metadata={},
    )


class ModelRuntimeTests(unittest.TestCase):
    def test_model_profiles_contain_only_embeddings_and_chat(self) -> None:
        self.assertEqual(
            [role.key for role in roles_for_profile("cpu")],
            ["embeddings_cpu", "chat_cpu"],
        )
        self.assertEqual(
            [role.key for role in roles_for_profile("gpu")],
            ["embeddings_gpu", "chat_gpu"],
        )
        gpu_embeddings = get_role("embeddings_gpu")
        self.assertEqual(gpu_embeddings.directory, "qwen3-embedding-8b")
        self.assertEqual(gpu_embeddings.quantization, "Q8_0")
        gpu_chat = get_role("chat_gpu")
        self.assertEqual(gpu_chat.directory, "qwen3-14b")
        self.assertEqual(gpu_chat.quantization, "Q5_K_M")
        self.assertEqual(
            gpu_chat.patterns[0],
            "Qwen3-14B-Q5_K_M.gguf",
        )

    def test_model_check_accepts_gguf_headers_without_loading(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            embedding = root / "embedding.gguf"
            chat = root / "chat.gguf"
            embedding.write_bytes(b"GGUF" + b"0" * 32)
            chat.write_bytes(b"GGUF" + b"0" * 32)
            settings = Settings(
                local_model_profile="gpu",
                embeddings_gpu_gguf_path=str(embedding),
                chat_gpu_gguf_path=str(chat),
            )
            self.assertTrue(all(item["valid"] for item in check_models(settings, "gpu")))

    def test_server_specs_have_no_document_parser_roles(self) -> None:
        settings = Settings(local_model_profile="cpu")
        specs = build_server_specs(settings)
        self.assertEqual(set(specs), {"embeddings", "chat"})
        self.assertTrue(specs["embeddings"].embeddings)

    def test_llama_command_contains_only_local_model_arguments(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            binary = root / "llama-server.exe"
            model = root / "model.gguf"
            binary.write_bytes(b"binary")
            model.write_bytes(b"GGUF" + b"0" * 32)
            settings = Settings(llama_cpp_binary=str(binary))
            spec = ModelServerSpec(
                role="embeddings",
                mode="managed",
                endpoint="http://127.0.0.1:8099/v1",
                model_path=model,
                alias="default",
                embeddings=True,
            )
            command = ModelSupervisor(settings, specs={"embeddings": spec}).build_command(spec)
            self.assertIn("--embedding", command)
            self.assertIn("--pooling", command)
            self.assertIn("last", command)
            self.assertNotIn("--mmproj", command)
            self.assertNotIn("--hf-repo", command)


class CoordinatorTests(unittest.TestCase):
    def test_index_job_can_acquire_embeddings_and_releases_cleanly(self) -> None:
        coordinator = ResourceCoordinator()
        with coordinator.acquire_indexing(timeout=0.1):
            with coordinator.acquire("embeddings", workload="index", timeout=0.1):
                self.assertEqual(coordinator.snapshot()["active_resource"], "embeddings")
        self.assertFalse(coordinator.snapshot()["indexing_active"])

    def test_removed_parser_resource_is_rejected(self) -> None:
        coordinator = ResourceCoordinator()
        with self.assertRaises(ValueError):
            coordinator.acquire("parser_bundle", timeout=0.01)  # type: ignore[arg-type]


class StorageAndRetrievalTests(unittest.TestCase):
    def test_lancedb_roundtrip_uses_page_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            store = LanceDBStore(Path(temp) / "lance", "chunks")
            chunk = make_chunk(0, "contenido")
            store.replace_chunks([chunk], [[0.1, 0.2, 0.3]])
            rows = store.list_chunks()
            self.assertEqual(rows[0]["page_start"], 1)
            self.assertEqual(rows[0]["page_chunk_index"], 0)
            self.assertEqual(rows[0]["schema_version"], 3)

    def test_lancedb_search_accepts_cosine_and_l2_distances(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            store = LanceDBStore(Path(temp) / "lance", "chunks")
            store.replace_chunks(
                [make_chunk(0, "primero"), make_chunk(1, "segundo")],
                [[1.0, 0.0], [0.0, 1.0]],
            )
            cosine = store.search([1.0, 0.0], 2, distance_type="cosine")
            l2 = store.search([1.0, 0.0], 2, distance_type="l2")
            self.assertEqual(cosine[0]["chunk_id"], "chunk-0")
            self.assertEqual(l2[0]["chunk_id"], "chunk-0")

    def test_bm25_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            store = BM25Store(Path(temp) / "bm25.json")
            store.replace_chunks(
                [
                    make_chunk(0, "ornitorrinco exclusivo"),
                    make_chunk(1, "texto comun"),
                    make_chunk(2, "otro documento"),
                ]
            )
            matches = BM25Store(Path(temp) / "bm25.json").search("ornitorrinco", 2)
            self.assertEqual(matches[0]["chunk_id"], "chunk-0")

    def test_bm25_exact_identifier_survives_similar_small_corpus(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            store = BM25Store(Path(temp) / "bm25.json")
            store.replace_chunks(
                [
                    make_chunk(0, "INC-001 05/01/2026 Zaragoza Operaciones Validada"),
                    make_chunk(1, "INC-021 05/02/2026 Zaragoza Operaciones Validada"),
                    make_chunk(2, "INC-060 05/03/2026 Zaragoza Operaciones Validada"),
                ]
            )

            matches = store.search(
                "INC-060 05/03/2026 Zaragoza Operaciones Validada",
                3,
            )

            self.assertEqual(matches[0]["chunk_id"], "chunk-2")
            self.assertGreater(matches[0]["_bm25_score"], 0.0)

    def test_hybrid_rrf_and_page_collapse_are_deterministic(self) -> None:
        vector = [[
            {**asdict(make_chunk(0, "a", page=4)), "_distance": 0.1},
            {**asdict(make_chunk(1, "b", page=4)), "_distance": 0.2},
        ]]
        lexical = [[
            {**asdict(make_chunk(2, "c", page=7)), "_bm25_score": 3.0},
            {**asdict(make_chunk(0, "a", page=4)), "_bm25_score": 2.0},
        ]]
        ranked = RagPipeline._merge_hybrid_matches(vector, lexical, 10, rrf_k=60)
        pages = RagPipeline._collapse_to_pages(ranked, 2)
        self.assertEqual(len(pages), 2)
        self.assertEqual(len({(item["document_id"], item["page_start"]) for item in pages}), 2)


class FakeDocumentParser:
    def __init__(self, document: MarkdownDocument) -> None:
        self.document = document
        self.calls: list[dict] = []

    def load_directory(self, _path, **kwargs):
        self.calls.append(kwargs)
        return [self.document]


class FakeEmbeddingClient:
    def embed_texts(self, texts, progress_callback=None):
        return [[float(index), 1.0] for index, _text in enumerate(texts)]


class PipelineIndexTests(unittest.TestCase):
    def test_index_uses_document_parser_then_local_embeddings(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            document = MarkdownDocument(
                document_id="doc",
                source="doc.md",
                source_path=str(root / "doc.md"),
                source_type="md",
                pages=[MarkdownPage(1, "# Titulo\n\nContenido")],
                source_sha256="hash",
                parser_model="direct-markdown",
                prompt_version="none",
            )
            parser = FakeDocumentParser(document)
            settings = Settings(
                model_supervision_enabled=False,
                lancedb_uri=str(root / "lance"),
                lancedb_table="chunks",
                bm25_index_dir=str(root / "bm25"),
            )
            pipeline = RagPipeline(settings, document_parser=parser)  # type: ignore[arg-type]
            pipeline.client = FakeEmbeddingClient()  # type: ignore[assignment]

            count = pipeline.index_documents(root, refresh_bedrock=False)

            self.assertEqual(count, 1)
            self.assertEqual(parser.calls[0]["refresh"], False)
            self.assertEqual(pipeline.store.list_chunks()[0]["page_start"], 1)
            self.assertTrue(settings.bm25_index_path.exists())


if __name__ == "__main__":
    unittest.main()
