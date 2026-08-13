"""Orchestration for Bedrock ingestion and local RAG retrieval/generation."""

from __future__ import annotations

import gc
import threading
import time
from pathlib import Path
from typing import Any, Callable, Literal

from rag_cliente.bedrock_parser import BedrockMarkdownParser
from rag_cliente.bm25_store import BM25Store
from rag_cliente.config import Settings
from rag_cliente.indexer import MarkdownChunker
from rag_cliente.llm_client import LlamaCppClient
from rag_cliente.model_supervisor import ModelSupervisor
from rag_cliente.resource_coordinator import get_resource_coordinator
from rag_cliente.vector_store import LanceDBStore

ProgressCallback = Callable[[str], None]
UNGROUNDED_ANSWER = "No consta en los documentos recuperados."


def grounded_answer(answer: str, citations: list[dict[str, Any]]) -> str:
    """Impide entregar como respuesta una generación sin respaldo documental."""
    normalized = answer.strip()
    return normalized if normalized and citations else UNGROUNDED_ANSWER


class RagPipeline:
    """Small facade for indexing, retrieval and answer generation."""

    def __init__(
        self,
        settings: Settings,
        *,
        document_parser: BedrockMarkdownParser | None = None,
    ) -> None:
        self.settings = settings
        self.client = LlamaCppClient(settings)
        self.document_parser = document_parser or BedrockMarkdownParser(settings)
        self.chunker = MarkdownChunker(settings)
        self.store = LanceDBStore(settings.lancedb_path, settings.lancedb_table)
        self.bm25_store = BM25Store(settings.bm25_index_path)
        self.coordinator = get_resource_coordinator()
        self.supervisor = (
            ModelSupervisor(settings) if settings.model_supervision_enabled else None
        )

    def _ensure_models(self, roles: tuple[str, ...]) -> None:
        if self.supervisor is None:
            return
        started: list[str] = []
        try:
            for role in roles:
                self.supervisor.ensure_started(role)
                started.append(role)
        except BaseException:
            self.supervisor.stop_bundle(started)
            raise

    def _stop_models(self, roles: tuple[str, ...]) -> None:
        if self.supervisor is not None:
            self.supervisor.stop_bundle(roles)

    @staticmethod
    def _release_model_memory() -> None:
        gc.collect()

    @staticmethod
    def _emit(callback: ProgressCallback | None, message: str) -> None:
        if callback is not None:
            callback(message)

    def index_documents(
        self,
        doc_dir: Path,
        tag: str | None = None,
        progress_callback: ProgressCallback | None = None,
        *,
        refresh_bedrock: bool = False,
    ) -> int:
        """Index PDF/Markdown documents, reusing the paid Markdown cache."""
        self._emit(progress_callback, f"Iniciando indexado en: {doc_dir}")
        with self.coordinator.acquire_indexing(timeout=self.settings.index_job_timeout):
            documents = self.document_parser.load_directory(
                doc_dir,
                refresh=refresh_bedrock,
                progress_callback=progress_callback,
            )
            page_count = sum(len(document.pages) for document in documents)
            self._emit(
                progress_callback,
                f"Documentos: {len(documents)}; paginas Markdown: {page_count}",
            )

            chunks = self.chunker.chunk_documents(documents, tag=tag)
            self._emit(progress_callback, f"Chunks por pagina: {len(chunks)}")
            if not chunks:
                self._emit(progress_callback, "No hay contenido que indexar.")
                return 0

            with self.coordinator.acquire(
                "embeddings",
                workload="index",
                timeout=self.settings.index_job_timeout,
            ):
                self._ensure_models(("embeddings",))
                try:
                    embeddings = self.client.embed_texts(
                        [chunk.text for chunk in chunks],
                        progress_callback=progress_callback,
                    )
                finally:
                    self._stop_models(("embeddings",))
                    self._release_model_memory()

            self._emit(progress_callback, "Guardando LanceDB y BM25...")
            self.store.replace_chunks(chunks, embeddings)
            # Always persist both indexes so evaluation can switch retrieval mode
            # without paying for or repeating document parsing.
            self.bm25_store.replace_chunks(chunks)
            self._emit(
                progress_callback,
                f"Indexado completado en '{self.settings.lancedb_table}'.",
            )
            return len(chunks)

    @staticmethod
    def _is_contextual_retrieval_query(query: str) -> bool:
        return (
            "Previous user question:" in query
            or "Latest follow-up question:" in query
        )

    @staticmethod
    def _build_retrieval_queries(
        question: str,
        rewritten_question: str,
        messages: list[dict[str, str]] | None = None,
        query_variants: list[str] | None = None,
    ) -> list[str]:
        queries: list[str] = []
        for candidate in (question, *(query_variants or []), rewritten_question):
            normalized = candidate.strip()
            if normalized and normalized not in queries:
                queries.append(normalized)

        history = [
            {
                "role": str(message.get("role", "")).strip(),
                "content": str(message.get("content", "")).strip(),
            }
            for message in (messages or [])
            if str(message.get("role", "")).strip()
            and str(message.get("content", "")).strip()
        ]
        prior_users = [item["content"] for item in history if item["role"] == "user"]
        if prior_users:
            contextual = (
                f"Previous user question: {prior_users[-1]}\n"
                f"Latest follow-up question: {question.strip()}"
            )
            if contextual not in queries:
                queries.append(contextual)
        return queries

    @staticmethod
    def _match_key(match: dict[str, Any]) -> tuple[Any, ...]:
        stable_id = match.get("chunk_id") or match.get("id")
        if stable_id:
            return (stable_id,)
        return (
            match.get("document_id"),
            match.get("source_path"),
            match.get("chunk_index"),
        )

    @staticmethod
    def _vector_score(match: dict[str, Any]) -> float:
        distance = match.get("_distance")
        if not isinstance(distance, (int, float)):
            return 0.0
        return 1.0 / (1.0 + max(float(distance), 0.0))

    @classmethod
    def _merge_matches(
        cls,
        match_groups: list[list[dict[str, Any]]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        merged: dict[tuple[Any, ...], dict[str, Any]] = {}
        for matches in match_groups:
            for rank, match in enumerate(matches):
                key = cls._match_key(match)
                distance = match.get("_distance")
                normalized_distance = (
                    float(distance)
                    if isinstance(distance, (int, float))
                    else float("inf")
                )
                candidate = {
                    **match,
                    "_merge_rank": rank,
                    "_normalized_distance": normalized_distance,
                }
                current = merged.get(key)
                if current is None or (
                    normalized_distance,
                    rank,
                ) < (
                    current["_normalized_distance"],
                    current["_merge_rank"],
                ):
                    merged[key] = candidate

        ordered = sorted(
            merged.values(),
            key=lambda item: (item["_normalized_distance"], item["_merge_rank"]),
        )
        return [
            {
                key: value
                for key, value in match.items()
                if key not in {"_merge_rank", "_normalized_distance"}
            }
            for match in ordered[:top_k]
        ]

    @classmethod
    def _merge_hybrid_matches(
        cls,
        vector_match_groups: list[list[dict[str, Any]]],
        bm25_match_groups: list[list[dict[str, Any]]],
        top_k: int,
        rrf_k: int = 60,
        **_legacy_options: Any,
    ) -> list[dict[str, Any]]:
        merged: dict[tuple[Any, ...], dict[str, Any]] = {}

        def best_ranks(groups: list[list[dict[str, Any]]]):
            best: dict[tuple[Any, ...], tuple[int, dict[str, Any]]] = {}
            for matches in groups:
                for rank, match in enumerate(matches, start=1):
                    key = cls._match_key(match)
                    if key not in best or rank < best[key][0]:
                        best[key] = (rank, match)
            return best

        for source, groups in (
            ("vector", vector_match_groups),
            ("bm25", bm25_match_groups),
        ):
            for key, (rank, match) in best_ranks(groups).items():
                item = merged.setdefault(key, dict(match))
                item.setdefault("_retrieval_sources", [])
                for field, value in match.items():
                    item.setdefault(field, value)
                item[f"_{source}_rank"] = rank
                if source == "vector":
                    item["_vector_score"] = cls._vector_score(match)
                else:
                    item["_bm25_raw_score"] = float(match.get("_bm25_score", 0.0))
                if source not in item["_retrieval_sources"]:
                    item["_retrieval_sources"].append(source)

        for item in merged.values():
            item["_rrf_score"] = sum(
                1.0 / (rrf_k + int(item[field]))
                for field in ("_vector_rank", "_bm25_rank")
                if field in item
            )
        return sorted(
            merged.values(),
            key=lambda item: (
                -float(item.get("_rrf_score", 0.0)),
                int(item.get("_vector_rank", 10**9)),
                int(item.get("_bm25_rank", 10**9)),
                str(item.get("chunk_id") or item.get("id") or ""),
            ),
        )[:top_k]

    @staticmethod
    def _collapse_to_pages(matches: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        """Return at most one ranked hit per source page for page-level evals."""
        selected: list[dict[str, Any]] = []
        seen: set[tuple[str, int]] = set()
        for match in matches:
            key = (str(match.get("document_id", "")), int(match.get("page_start", 0)))
            if key in seen:
                continue
            seen.add(key)
            selected.append(match)
            if len(selected) >= top_k:
                break
        return selected

    def _retrieve(
        self,
        queries: list[str],
        *,
        top_k: int,
        tag: str | None,
        retrieval_mode: Literal["vector", "bm25", "hybrid"] | None = None,
        distance_type: Literal["l2", "cosine", "dot"] | None = None,
        use_query_instruction: bool | None = None,
        manage_embeddings: bool = True,
    ) -> list[dict[str, Any]]:
        mode = retrieval_mode or self.settings.retrieval_mode
        vector_groups: list[list[dict[str, Any]]] = []
        bm25_groups: list[list[dict[str, Any]]] = []

        if mode in {"vector", "hybrid"}:
            def vector_search() -> list[list[dict[str, Any]]]:
                if use_query_instruction is None:
                    query_vectors = self.client.embed_texts(queries, query_mode=True)
                else:
                    query_vectors = self.client.embed_texts(
                        queries,
                        query_mode=True,
                        use_query_instruction=use_query_instruction,
                    )
                return [
                    (
                        self.store.search(
                            vector,
                            top_k=self.settings.vector_candidates,
                            tag=tag,
                        )
                        if distance_type is None
                        else self.store.search(
                            vector,
                            top_k=self.settings.vector_candidates,
                            tag=tag,
                            distance_type=distance_type,
                        )
                    )
                    for vector in query_vectors
                ]

            if manage_embeddings:
                with self.coordinator.acquire(
                    "embeddings",
                    workload="query",
                    timeout=self.settings.model_request_timeout,
                ):
                    self._ensure_models(("embeddings",))
                    try:
                        vector_groups = vector_search()
                    finally:
                        self._stop_models(("embeddings",))
                        self._release_model_memory()
            else:
                vector_groups = vector_search()

        if mode in {"bm25", "hybrid"}:
            lexical_queries = [
                query for query in queries if not self._is_contextual_retrieval_query(query)
            ]
            bm25_groups = [
                self.bm25_store.search(
                    query,
                    top_k=self.settings.bm25_candidates,
                    tag=tag,
                )
                for query in lexical_queries
            ]

        candidate_limit = max(
            top_k * 4,
            self.settings.vector_candidates,
            self.settings.bm25_candidates,
        )
        if mode == "vector":
            ranked = self._merge_matches(vector_groups, candidate_limit)
        elif mode == "bm25":
            ranked = self._merge_hybrid_matches(
                [], bm25_groups, candidate_limit, rrf_k=self.settings.rrf_k
            )
        else:
            ranked = self._merge_hybrid_matches(
                vector_groups,
                bm25_groups,
                candidate_limit,
                rrf_k=self.settings.rrf_k,
            )
        return self._collapse_to_pages(ranked, top_k)

    def retrieve(
        self,
        question: str,
        *,
        top_k: int,
        retrieval_mode: Literal["vector", "bm25", "hybrid"],
        distance_type: Literal["l2", "cosine"] = "cosine",
        use_query_instruction: bool = True,
        query_variants: list[str] | None = None,
        tag: str | None = None,
    ) -> list[dict[str, Any]]:
        """Recupera paginas para evaluacion sin cargar el modelo de chat."""
        queries: list[str] = []
        for candidate in [question, *(query_variants or [])]:
            normalized = candidate.strip()
            if normalized and normalized not in queries:
                queries.append(normalized)
        if not queries:
            raise ValueError("La pregunta de evaluacion no puede estar vacia.")
        return self._retrieve(
            queries,
            top_k=top_k,
            tag=(tag or "").strip() or None,
            retrieval_mode=retrieval_mode,
            distance_type=distance_type if retrieval_mode != "bm25" else None,
            use_query_instruction=(
                use_query_instruction if retrieval_mode != "bm25" else None
            ),
        )

    def retrieve_many(
        self,
        questions: list[tuple[str, list[str]]],
        *,
        top_k: int,
        retrieval_mode: Literal["vector", "bm25", "hybrid"],
        distance_type: Literal["l2", "cosine"] = "cosine",
        use_query_instruction: bool = True,
        tag: str | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> tuple[list[list[dict[str, Any]]], list[float]]:
        """Evalua varias preguntas manteniendo embeddings cargado una sola vez."""
        if not questions:
            return [], []

        def run_all(
            *, manage_embeddings: bool
        ) -> tuple[list[list[dict[str, Any]]], list[float]]:
            results: list[list[dict[str, Any]]] = []
            latencies_ms: list[float] = []
            total = len(questions)
            for index, (question, variants) in enumerate(questions, start=1):
                self._emit(progress_callback, f"Recuperando pregunta {index}/{total}")
                queries: list[str] = []
                for candidate in [question, *variants]:
                    normalized = candidate.strip()
                    if normalized and normalized not in queries:
                        queries.append(normalized)
                if not queries:
                    raise ValueError("La pregunta de evaluacion no puede estar vacia.")
                started = time.perf_counter()
                results.append(
                    self._retrieve(
                        queries,
                        top_k=top_k,
                        tag=(tag or "").strip() or None,
                        retrieval_mode=retrieval_mode,
                        distance_type=(
                            distance_type if retrieval_mode != "bm25" else None
                        ),
                        use_query_instruction=(
                            use_query_instruction if retrieval_mode != "bm25" else None
                        ),
                        manage_embeddings=manage_embeddings,
                    )
                )
                latencies_ms.append((time.perf_counter() - started) * 1000.0)
            return results, latencies_ms

        if retrieval_mode == "bm25":
            return run_all(manage_embeddings=False)
        with self.coordinator.acquire(
            "embeddings",
            workload="query",
            timeout=self.settings.model_request_timeout,
        ):
            self._ensure_models(("embeddings",))
            try:
                return run_all(manage_embeddings=False)
            finally:
                self._stop_models(("embeddings",))
                self._release_model_memory()

    def generate_query_variants(
        self,
        questions: list[str],
        progress_callback: ProgressCallback | None = None,
    ) -> dict[str, list[str]]:
        """Genera variantes en un bloque y libera chat antes de usar embeddings."""
        if not questions:
            return {}
        variants: dict[str, list[str]] = {}
        with self.coordinator.acquire(
            "chat",
            workload="query",
            timeout=self.settings.model_request_timeout,
        ):
            self._ensure_models(("chat",))
            try:
                total = len(questions)
                for index, question in enumerate(questions, start=1):
                    self._emit(
                        progress_callback,
                        f"Reformulando consulta {index}/{total}",
                    )
                    variants[question] = self.client.generate_query_variants(question)
            finally:
                self._stop_models(("chat",))
                self._release_model_memory()
        return variants

    @staticmethod
    def _citation_from_match(match: dict[str, Any], source_id: str) -> dict[str, Any]:
        return {
            "source_id": source_id,
            "chunk_id": match.get("chunk_id") or match.get("id"),
            "document_id": match["document_id"],
            "source": match["source"],
            "source_path": match["source_path"],
            "source_type": match["source_type"],
            "page_start": match["page_start"],
            "page_end": match["page_end"],
            "source_pages": list(match.get("source_pages") or []),
            "chunk_index": match["chunk_index"],
            "page_chunk_index": match.get("page_chunk_index", 0),
            "tag": match.get("tag") or None,
        }

    @staticmethod
    def _source_option_from_match(match: dict[str, Any], source_id: str) -> dict[str, Any]:
        return {
            **RagPipeline._citation_from_match(match, source_id),
            "text": match.get("text", ""),
        }

    @staticmethod
    def _filter_citations_by_source_ids(
        citations: list[dict[str, Any]],
        used_source_ids: list[str],
    ) -> list[dict[str, Any]]:
        allowed = set(used_source_ids)
        return [item for item in citations if item.get("source_id") in allowed]

    def _select_used_citations(
        self,
        question: str,
        answer: str,
        generation_inputs: dict[str, Any],
        messages: list[dict[str, str]] | None = None,
    ) -> list[dict[str, Any]]:
        used = self.client.select_used_source_ids(
            question=question,
            answer=answer,
            source_options=generation_inputs["source_options"],
            messages=messages,
        )
        return self._filter_citations_by_source_ids(generation_inputs["citations"], used)

    def _prepare_generation_inputs(
        self,
        question: str,
        top_k: int | None = None,
        messages: list[dict[str, str]] | None = None,
        tag: str | None = None,
        query_augmentation: bool = False,
    ) -> dict[str, Any]:
        effective_top_k = top_k or self.settings.effective_retrieval_top_k
        query_variants: list[str] = []
        augmentation_error: str | None = None
        if query_augmentation:
            try:
                query_variants = self.generate_query_variants([question])[question]
            except Exception as exc:
                # Query augmentation improves recall but must not make a normal
                # question unusable if the local model returns malformed output.
                augmentation_error = str(exc)
        rewritten = self.client.rewrite_question_for_retrieval(question, messages=messages)
        queries = self._build_retrieval_queries(
            question,
            rewritten,
            messages=messages,
            query_variants=query_variants,
        )
        matches = self._retrieve(
            queries,
            top_k=effective_top_k,
            tag=(tag or "").strip() or None,
        )

        context_blocks: list[str] = []
        citations: list[dict[str, Any]] = []
        source_options: list[dict[str, Any]] = []
        for index, match in enumerate(matches, start=1):
            source_id = f"S{index}"
            page = int(match["page_start"])
            context_blocks.append(
                f"[{source_id}] {match['source']} p.{page}\n{match['text']}"
            )
            citations.append(self._citation_from_match(match, source_id))
            source_options.append(self._source_option_from_match(match, source_id))
        return {
            "context_blocks": context_blocks,
            "citations": citations,
            "source_options": source_options,
            "matches": matches,
            "retrieval_queries": queries,
            "query_variants": query_variants,
            "query_augmentation_error": augmentation_error,
        }

    def ask(
        self,
        question: str,
        top_k: int | None = None,
        messages: list[dict[str, str]] | None = None,
        tag: str | None = None,
        enable_reasoning: bool = False,
        query_augmentation: bool = False,
    ) -> dict[str, Any]:
        inputs = self._prepare_generation_inputs(
            question,
            top_k,
            messages,
            tag,
            query_augmentation=query_augmentation,
        )
        with self.coordinator.acquire(
            "chat",
            workload="query",
            timeout=self.settings.model_request_timeout,
        ):
            self._ensure_models(("chat",))
            try:
                generation = self.client.generate_answer(
                    question,
                    inputs["context_blocks"],
                    messages=messages,
                    enable_reasoning=enable_reasoning,
                )
                citations = self._select_used_citations(
                    question, generation["answer"], inputs, messages
                )
            finally:
                if self.supervisor is not None:
                    self.supervisor.schedule_idle_stop("chat")
        return {
            "answer": grounded_answer(generation["answer"], citations),
            "reasoning": generation["reasoning"],
            "citations": citations,
            "matches": inputs["matches"],
            "retrieval_queries": inputs["retrieval_queries"],
            "query_variants": inputs["query_variants"],
            "query_augmentation_error": inputs["query_augmentation_error"],
        }

    def stream_answer(
        self,
        question: str,
        top_k: int | None = None,
        messages: list[dict[str, str]] | None = None,
        tag: str | None = None,
        enable_reasoning: bool = False,
        query_augmentation: bool = False,
    ) -> dict[str, Any]:
        inputs = self._prepare_generation_inputs(
            question,
            top_k,
            messages,
            tag,
            query_augmentation=query_augmentation,
        )
        lease = self.coordinator.acquire(
            "chat",
            workload="query",
            timeout=self.settings.model_request_timeout,
        )
        try:
            self._ensure_models(("chat",))
            primary = self.client.stream_answer(
                question,
                inputs["context_blocks"],
                messages=messages,
                enable_reasoning=enable_reasoning,
            )
        except BaseException:
            lease.release()
            raise

        lock = threading.Lock()
        cleaned = False

        def cleanup() -> None:
            nonlocal cleaned
            with lock:
                if cleaned:
                    return
                cleaned = True
                try:
                    if self.supervisor is not None:
                        self.supervisor.schedule_idle_stop("chat")
                finally:
                    lease.release()

        def guarded(stream):
            completed = False
            try:
                yield from stream
                completed = True
            finally:
                if not completed:
                    cleanup()

        def fallback_stream():
            return guarded(
                self.client.stream_answer(
                    question,
                    inputs["context_blocks"],
                    messages=messages,
                    enable_reasoning=False,
                )
            )

        def resolve_citations(answer: str) -> list[dict[str, Any]]:
            try:
                return self._select_used_citations(question, answer, inputs, messages)
            finally:
                cleanup()

        return {
            "answer_stream": guarded(primary),
            "fallback_stream": fallback_stream,
            "resolve_citations": resolve_citations,
            "close": cleanup,
            "matches": inputs["matches"],
            "retrieval_queries": inputs["retrieval_queries"],
            "query_variants": inputs["query_variants"],
            "query_augmentation_error": inputs["query_augmentation_error"],
        }
