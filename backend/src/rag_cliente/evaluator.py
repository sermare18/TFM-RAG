"""Ejecucion reproducible de evaluaciones de recuperacion por pagina."""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Literal, Protocol

from rag_cliente.evaluation_store import EvaluationStore, QuestionRecord, RelevantPage
from rag_cliente.llm_client import QUERY_AUGMENTATION_PROMPT_VERSION

RetrievalMode = Literal["bm25", "vector", "hybrid"]
DistanceType = Literal["l2", "cosine"]
EvaluationProgress = Callable[[int, int, str], None]


class EvaluationRetriever(Protocol):
    def retrieve(self, question: str, **kwargs: Any) -> list[dict[str, Any]]: ...

    def generate_query_variants(
        self,
        questions: list[str],
        progress_callback: Callable[[str], None] | None = None,
    ) -> dict[str, list[str]]: ...


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    name: str
    retrieval_mode: RetrievalMode
    top_k: int
    distance_type: DistanceType = "cosine"
    use_query_instruction: bool = True
    use_query_augmentation: bool = False
    tag: str | None = None

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("La evaluación necesita un nombre.")
        if self.retrieval_mode not in {"bm25", "vector", "hybrid"}:
            raise ValueError("El modo de recuperación no es válido.")
        if not 1 <= self.top_k <= 20:
            raise ValueError("Top K debe estar entre 1 y 20.")
        if self.distance_type not in {"l2", "cosine"}:
            raise ValueError("La distancia vectorial no es válida.")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["name"] = self.name.strip()
        payload["tag"] = (self.tag or "").strip() or None
        if self.retrieval_mode == "bm25":
            payload["distance_type"] = None
            payload["use_query_instruction"] = False
        return payload


def _page_key(document_id: str, page: int) -> tuple[str, int]:
    return document_id.strip(), int(page)


def score_retrieval(
    expected_pages: list[RelevantPage],
    retrieved: list[dict[str, Any]],
    top_k: int,
) -> dict[str, Any]:
    """Calcula metricas macro por pregunta sobre un ranking de paginas."""
    expected = {_page_key(page.document_id, page.page) for page in expected_pages}
    ranked = [
        _page_key(str(match.get("document_id", "")), int(match.get("page_start", 0)))
        for match in retrieved[:top_k]
    ]
    relevant_ranks = [
        rank for rank, candidate in enumerate(ranked, start=1) if candidate in expected
    ]
    relevant_at_1 = int(bool(ranked and ranked[0] in expected))
    relevant_at_k = len({candidate for candidate in ranked if candidate in expected})
    expected_count = len(expected)
    first_rank = relevant_ranks[0] if relevant_ranks else None
    return {
        "expected_count": expected_count,
        "retrieved_count": len(ranked),
        "relevant_at_1": relevant_at_1,
        "relevant_at_k": relevant_at_k,
        "hit_at_1": float(relevant_at_1 > 0),
        "hit_at_k": float(relevant_at_k > 0),
        "recall_at_1": relevant_at_1 / expected_count,
        "recall_at_k": relevant_at_k / expected_count,
        "precision_at_k": relevant_at_k / top_k,
        "mrr_at_k": 1.0 / first_rank if first_rank is not None else 0.0,
        "first_relevant_rank": first_rank,
        "false_positives": sum(candidate not in expected for candidate in ranked),
        "failure": first_rank is None,
    }


def aggregate_metrics(results: list[dict[str, Any]], top_k: int) -> dict[str, Any]:
    if not results:
        raise ValueError("No hay resultados para agregar.")
    metric_names = (
        "hit_at_1",
        "hit_at_k",
        "recall_at_1",
        "recall_at_k",
        "precision_at_k",
        "mrr_at_k",
    )
    count = len(results)
    latencies = sorted(float(result["latency_ms"]) for result in results)
    p95_index = max(0, math.ceil(0.95 * len(latencies)) - 1)
    aggregate = {
        name: sum(float(result["metrics"][name]) for result in results) / count
        for name in metric_names
    }
    aggregate.update(
        {
            "top_k": top_k,
            "question_count": count,
            "false_positives": sum(
                int(result["metrics"]["false_positives"]) for result in results
            ),
            "failures": sum(bool(result["metrics"]["failure"]) for result in results),
            "error_count": sum(bool(result.get("error")) for result in results),
            "latency_mean_ms": sum(latencies) / len(latencies),
            "latency_p95_ms": latencies[p95_index],
        }
    )
    return aggregate


def _serialized_match(match: dict[str, Any], rank: int) -> dict[str, Any]:
    def numeric(name: str) -> float | None:
        value = match.get(name)
        return float(value) if isinstance(value, (int, float)) else None

    return {
        "rank": rank,
        "document_id": str(match.get("document_id", "")),
        "source": str(match.get("source", "")),
        "source_path": str(match.get("source_path", "")),
        "page": int(match.get("page_start", 0)),
        "page_chunk_index": int(match.get("page_chunk_index", 0)),
        "tag": str(match.get("tag") or ""),
        "text": str(match.get("text", "")),
        "distance": numeric("_distance"),
        "vector_score": numeric("_vector_score"),
        "bm25_score": numeric("_bm25_score") or numeric("_bm25_raw_score"),
        "rrf_score": numeric("_rrf_score"),
        "retrieval_sources": [
            str(item) for item in match.get("_retrieval_sources", [])
        ],
    }


class EvaluationRunner:
    def __init__(self, store: EvaluationStore, retriever: EvaluationRetriever) -> None:
        self.store = store
        self.retriever = retriever

    def _load_variants(
        self,
        questions: list[QuestionRecord],
        progress_callback: EvaluationProgress | None,
    ) -> dict[int, list[str]]:
        variants_by_id: dict[int, list[str]] = {}
        missing: dict[str, list[int]] = {}
        for question in questions:
            cached = self.store.get_query_variants(
                question.question,
                QUERY_AUGMENTATION_PROMPT_VERSION,
            )
            if cached is not None:
                variants_by_id[question.id] = cached
            else:
                missing.setdefault(question.question, []).append(question.id)

        if missing:
            if progress_callback is not None:
                progress_callback(0, len(questions), "Generando reformulaciones pendientes")
            generated = self.retriever.generate_query_variants(
                list(missing),
                progress_callback=(
                    (lambda message: progress_callback(0, len(questions), message))
                    if progress_callback is not None
                    else None
                ),
            )
            for question_text, question_ids in missing.items():
                variants = generated[question_text]
                self.store.save_query_variants(
                    question_text,
                    QUERY_AUGMENTATION_PROMPT_VERSION,
                    variants,
                )
                for question_id in question_ids:
                    variants_by_id[question_id] = variants
        return variants_by_id

    def run(
        self,
        config: EvaluationConfig,
        progress_callback: EvaluationProgress | None = None,
    ) -> dict[str, Any]:
        config.validate()
        questions, dataset_hash = self.store.active_dataset()
        if not questions:
            raise ValueError("El dataset no contiene preguntas activas.")
        config_payload = config.to_dict()
        evaluation_id = self.store.start_evaluation(
            config.name,
            config_payload,
            dataset_hash,
            len(questions),
        )
        try:
            variants_by_id = (
                self._load_variants(questions, progress_callback)
                if config.use_query_augmentation
                else {}
            )
            batch_matches: list[list[dict[str, Any]]] | None = None
            batch_latencies_ms: list[float] | None = None
            retrieve_many = getattr(self.retriever, "retrieve_many", None)
            if callable(retrieve_many):
                batch_output = retrieve_many(
                    [
                        (question.question, variants_by_id.get(question.id, []))
                        for question in questions
                    ],
                    top_k=config.top_k,
                    retrieval_mode=config.retrieval_mode,
                    distance_type=config.distance_type,
                    use_query_instruction=config.use_query_instruction,
                    tag=(config.tag or "").strip() or None,
                    progress_callback=(
                        (lambda message: progress_callback(0, len(questions), message))
                        if progress_callback is not None
                        else None
                    ),
                )
                batch_matches, batch_latencies_ms = batch_output
            completed_results: list[dict[str, Any]] = []
            for position, question in enumerate(questions, start=1):
                if progress_callback is not None:
                    progress_callback(
                        position - 1,
                        len(questions),
                        f"Evaluando pregunta {position}/{len(questions)}",
                    )
                variants = variants_by_id.get(question.id, [])
                error: str | None = None
                if batch_matches is not None:
                    matches = batch_matches[position - 1]
                    latencies = batch_latencies_ms or [0.0] * len(questions)
                    latency_ms = float(latencies[position - 1])
                else:
                    started = time.perf_counter()
                    matches = []
                    try:
                        matches = self.retriever.retrieve(
                            question.question,
                            top_k=config.top_k,
                            retrieval_mode=config.retrieval_mode,
                            distance_type=config.distance_type,
                            use_query_instruction=config.use_query_instruction,
                            query_variants=variants,
                            tag=(config.tag or "").strip() or None,
                        )
                    except Exception as exc:  # conserva el resto del experimento
                        error = str(exc)
                    latency_ms = (time.perf_counter() - started) * 1000.0
                serialized = [
                    _serialized_match(match, rank)
                    for rank, match in enumerate(matches, start=1)
                ]
                metrics = score_retrieval(
                    list(question.relevant_pages),
                    matches,
                    config.top_k,
                )
                expected = [asdict(page) for page in question.relevant_pages]
                self.store.add_evaluation_result(
                    evaluation_id,
                    question_id=question.id,
                    question_text=question.question,
                    expected=expected,
                    retrieved=serialized,
                    query_variants=variants,
                    metrics=metrics,
                    latency_ms=latency_ms,
                    error=error,
                )
                completed_results.append(
                    {"metrics": metrics, "latency_ms": latency_ms, "error": error}
                )

            aggregate = aggregate_metrics(completed_results, config.top_k)
            self.store.finish_evaluation(evaluation_id, aggregate)
            if progress_callback is not None:
                progress_callback(len(questions), len(questions), "Evaluación completada")
            return self.store.get_evaluation(evaluation_id)
        except BaseException as exc:
            self.store.fail_evaluation(evaluation_id, str(exc))
            raise
