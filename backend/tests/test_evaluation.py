from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, call

from evaluation_app import evaluation_table, results_csv
from rag_cliente.config import Settings
from rag_cliente.evaluation_store import EvaluationStore, RelevantPage
from rag_cliente.evaluator import EvaluationConfig, EvaluationRunner, score_retrieval
from rag_cliente.llm_client import LlamaCppClient
from rag_cliente.pipeline import RagPipeline
from streamlit_lancedb_viewer import build_table_rows


def relevant(document_id: str = "doc-1", page: int = 2) -> RelevantPage:
    return RelevantPage(
        document_id=document_id,
        source=f"{document_id}.pdf",
        source_path=f"asignatura/{document_id}.pdf",
        page=page,
        reference_text="fragmento correcto",
    )


def match(document_id: str, page: int, text: str = "texto") -> dict:
    return {
        "document_id": document_id,
        "source": f"{document_id}.pdf",
        "source_path": f"asignatura/{document_id}.pdf",
        "page_start": page,
        "page_chunk_index": 0,
        "text": text,
        "tag": "asignatura",
        "_distance": 0.2,
    }


class EvaluationStoreTests(unittest.TestCase):
    def test_dataset_is_persistent_and_hash_changes_with_ground_truth(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "evaluation.sqlite"
            store = EvaluationStore(path)
            question_id = store.save_question("Pregunta", [relevant()])
            questions, first_hash = EvaluationStore(path).active_dataset()
            self.assertEqual(questions[0].id, question_id)
            self.assertEqual(questions[0].relevant_pages[0].page, 2)

            store.save_question("Pregunta editada", [relevant(page=3)], question_id=question_id)
            _questions, second_hash = store.active_dataset()
            self.assertNotEqual(first_hash, second_hash)

    def test_query_variant_cache_uses_question_and_prompt_version(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            store = EvaluationStore(Path(temp) / "evaluation.sqlite")
            self.assertIsNone(store.get_query_variants("pregunta", "v1"))
            store.save_query_variants("pregunta", "v1", ["variante a", "variante b"])
            self.assertEqual(
                store.get_query_variants("pregunta", "v1"),
                ["variante a", "variante b"],
            )
            self.assertIsNone(store.get_query_variants("pregunta", "v2"))


class MetricTests(unittest.TestCase):
    def test_page_metrics_support_multiple_relevant_pages(self) -> None:
        metrics = score_retrieval(
            [relevant(page=2), relevant(page=4)],
            [match("doc-1", 2), match("other", 8), match("doc-1", 4)],
            top_k=3,
        )
        self.assertEqual(metrics["hit_at_1"], 1.0)
        self.assertEqual(metrics["recall_at_1"], 0.5)
        self.assertEqual(metrics["recall_at_k"], 1.0)
        self.assertEqual(metrics["precision_at_k"], 2 / 3)
        self.assertEqual(metrics["mrr_at_k"], 1.0)
        self.assertEqual(metrics["false_positives"], 1)
        self.assertFalse(metrics["failure"])


class SpanishPresentationTests(unittest.TestCase):
    def test_evaluation_history_and_csv_use_spanish_labels(self) -> None:
        evaluation = {
            "id": 1,
            "name": "prueba",
            "status": "completed",
            "dataset_hash": "1234567890abcdef",
            "question_count": 1,
            "config": {
                "retrieval_mode": "hybrid",
                "distance_type": "cosine",
                "use_query_instruction": True,
                "use_query_augmentation": True,
                "top_k": 5,
            },
            "metrics": {"hit_at_k": 1.0, "recall_at_k": 1.0, "mrr_at_k": 1.0},
            "results": [
                {
                    "question_id": 1,
                    "question_text": "pregunta",
                    "expected": [{"source": "doc.pdf", "page": 2}],
                    "retrieved": [{"source": "doc.pdf", "page": 2}],
                    "metrics": {},
                    "latency_ms": 10.0,
                }
            ],
        }

        row = evaluation_table([evaluation])[0]
        self.assertEqual(row["estado"], "Completada")
        self.assertEqual(row["modo"], "Híbrido")
        self.assertEqual(row["distancia"], "Coseno")
        csv_text = results_csv(evaluation).decode("utf-8-sig")
        self.assertIn("páginas_esperadas", csv_text.splitlines()[0])
        self.assertIn("posición_primer_relevante", csv_text.splitlines()[0])

    def test_viewer_table_uses_spanish_column_names(self) -> None:
        row = build_table_rows(
            [
                {
                    "source": "doc.pdf",
                    "source_type": "pdf",
                    "page_start": 2,
                    "page_chunk_index": 0,
                    "text": "contenido",
                }
            ],
            show_full_text=True,
        )[0]

        self.assertIn("documento", row)
        self.assertIn("página", row)
        self.assertIn("ruta de origen", row)
        self.assertIn("texto", row)


class FakeRetriever:
    def __init__(self) -> None:
        self.retrieval_calls: list[dict] = []
        self.augmentation_calls = 0

    def retrieve(self, question: str, **kwargs):
        self.retrieval_calls.append({"question": question, **kwargs})
        if question == "acierto":
            return [match("doc-1", 2), match("other", 9)]
        return [match("other", 7), match("other", 8)]

    def generate_query_variants(self, questions, progress_callback=None):
        self.augmentation_calls += 1
        if progress_callback:
            progress_callback("generando")
        return {
            question: [f"{question} variante 1", f"{question} variante 2"]
            for question in questions
        }


class EvaluationRunnerTests(unittest.TestCase):
    def test_run_persists_metrics_rankings_and_survives_question_deletion(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            store = EvaluationStore(Path(temp) / "evaluation.sqlite")
            first_id = store.save_question("acierto", [relevant()])
            store.save_question("fallo", [relevant(document_id="doc-2", page=3)])
            retriever = FakeRetriever()
            evaluation = EvaluationRunner(store, retriever).run(
                EvaluationConfig(
                    name="base",
                    retrieval_mode="hybrid",
                    top_k=2,
                    distance_type="cosine",
                )
            )
            self.assertEqual(evaluation["metrics"]["hit_at_1"], 0.5)
            self.assertEqual(evaluation["metrics"]["failures"], 1)
            self.assertEqual(evaluation["metrics"]["false_positives"], 3)
            self.assertEqual(evaluation["results"][0]["retrieved"][0]["page"], 2)

            store.delete_question(first_id)
            historical = store.get_evaluation(evaluation["id"])
            self.assertEqual(len(historical["results"]), 2)
            self.assertEqual(historical["results"][0]["question_text"], "acierto")

    def test_augmentation_is_generated_once_and_reused_from_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            store = EvaluationStore(Path(temp) / "evaluation.sqlite")
            store.save_question("acierto", [relevant()])
            retriever = FakeRetriever()
            runner = EvaluationRunner(store, retriever)
            config = EvaluationConfig(
                name="augmentation",
                retrieval_mode="bm25",
                top_k=2,
                use_query_augmentation=True,
            )
            runner.run(config)
            runner.run(config)
            self.assertEqual(retriever.augmentation_calls, 1)
            self.assertEqual(
                retriever.retrieval_calls[-1]["query_variants"],
                ["acierto variante 1", "acierto variante 2"],
            )


class ConfigurableRetrievalTests(unittest.TestCase):
    def test_vector_evaluation_forwards_variants_distance_and_instruction_toggle(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pipeline = RagPipeline(
                Settings(
                    model_supervision_enabled=False,
                    lancedb_uri=str(root / "lance"),
                    bm25_index_dir=str(root / "bm25"),
                )
            )
            pipeline.client = Mock()
            pipeline.client.embed_texts.return_value = [[0.1], [0.2], [0.3]]
            pipeline.store = Mock()
            pipeline.store.search.return_value = []
            pipeline.bm25_store = Mock()

            pipeline.retrieve(
                "original",
                top_k=5,
                retrieval_mode="vector",
                distance_type="cosine",
                use_query_instruction=False,
                query_variants=["variante 1", "variante 2"],
            )

            pipeline.client.embed_texts.assert_called_once_with(
                ["original", "variante 1", "variante 2"],
                query_mode=True,
                use_query_instruction=False,
            )
            self.assertEqual(
                pipeline.store.search.call_args_list,
                [
                    call([0.1], top_k=40, tag=None, distance_type="cosine"),
                    call([0.2], top_k=40, tag=None, distance_type="cosine"),
                    call([0.3], top_k=40, tag=None, distance_type="cosine"),
                ],
            )
            pipeline.bm25_store.search.assert_not_called()

    def test_batch_evaluation_keeps_embeddings_loaded_for_all_questions(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pipeline = RagPipeline(
                Settings(
                    model_supervision_enabled=False,
                    lancedb_uri=str(root / "lance"),
                    bm25_index_dir=str(root / "bm25"),
                )
            )
            pipeline.supervisor = Mock()
            pipeline.client = Mock()
            pipeline.client.embed_texts.side_effect = [[[0.1]], [[0.2]]]
            pipeline.store = Mock()
            pipeline.store.search.return_value = []
            pipeline.bm25_store = Mock()

            results, latencies = pipeline.retrieve_many(
                [("pregunta 1", []), ("pregunta 2", [])],
                top_k=5,
                retrieval_mode="vector",
                distance_type="cosine",
                use_query_instruction=True,
            )

            self.assertEqual(results, [[], []])
            self.assertEqual(len(latencies), 2)
            pipeline.supervisor.ensure_started.assert_called_once_with("embeddings")
            pipeline.supervisor.stop_bundle.assert_called_once_with(("embeddings",))
            self.assertEqual(pipeline.client.embed_texts.call_count, 2)

    def test_query_variant_parser_accepts_json_and_rejects_original(self) -> None:
        variants = LlamaCppClient._parse_query_variants(
            "pregunta original",
            '{"queries":["pregunta original","reformulacion uno","reformulacion dos"]}',
        )
        self.assertEqual(variants, ["reformulacion uno", "reformulacion dos"])


if __name__ == "__main__":
    unittest.main()
