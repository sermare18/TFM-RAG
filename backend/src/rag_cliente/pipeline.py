"""Orquestación del flujo completo RAG.

Conecta carga de documentos, chunking, embeddings, almacenamiento vectorial,
recuperación y generación.
"""
from __future__ import annotations

import gc
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

from rag_cliente.bm25_store import BM25Store
from rag_cliente.config import Settings, resolve_marker_profile
from rag_cliente.indexer import PdfChunker
from rag_cliente.llm_client import LlamaCppClient
from rag_cliente.model_supervisor import ModelSupervisor
from rag_cliente.pdf_loader import load_documents_from_directory
from rag_cliente.resource_coordinator import get_resource_coordinator
from rag_cliente.vector_store import LanceDBStore

ProgressCallback = Callable[[str], None]


class RagPipeline:
    """Facade principal del sistema RAG."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = LlamaCppClient(settings)
        self.chunker = PdfChunker(settings)
        self.store = LanceDBStore(settings.lancedb_path, settings.lancedb_table)
        self.bm25_store = BM25Store(settings.bm25_index_path)
        self.coordinator = get_resource_coordinator()
        self.supervisor = (
            ModelSupervisor(settings)
            if settings.model_supervision_enabled
            else None
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
        """Libera referencias Python/CUDA sin importar frameworks nuevos."""
        gc.collect()
        torch_module = sys.modules.get("torch")
        cuda = getattr(torch_module, "cuda", None)
        if cuda is not None:
            try:
                if cuda.is_available():
                    cuda.empty_cache()
            except (AttributeError, RuntimeError):
                pass

    @staticmethod
    def _emit(progress_callback: ProgressCallback | None, message: str) -> None:
        if progress_callback is not None:
            progress_callback(message)

    def index_documents(
        self,
        doc_dir: Path,
        tag: str | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> int:
        """Indexa todos los documentos soportados de un directorio.

        Flujo:
        1. cargar documentos
        2. trocear texto
        3. generar embeddings
        4. persistir en LanceDB
        """
        self._emit(progress_callback, f"Iniciando indexado en: {doc_dir}")
        with self.coordinator.acquire_indexing(timeout=self.settings.parser_job_timeout):
            parser_started_at = time.monotonic()
            with self.coordinator.acquire(
                "parser_bundle",
                workload="index",
                timeout=self.settings.parser_job_timeout,
            ):
                profile = resolve_marker_profile(self.settings)
                parser_roles = ("surya", "vlm") if profile.use_llm else ()
                self._ensure_models(parser_roles)
                try:
                    pages = load_documents_from_directory(
                        doc_dir,
                        settings=self.settings,
                        progress_callback=progress_callback,
                    )
                    parser_elapsed = time.monotonic() - parser_started_at
                    if parser_elapsed > self.settings.parser_job_timeout:
                        raise TimeoutError(
                            f"PARSER_JOB_TIMEOUT excedido: {parser_elapsed:.1f}s"
                        )
                finally:
                    self._stop_models(parser_roles)
                    self._release_model_memory()

            self._emit(progress_callback, f"Total de páginas/bloques extraídos: {len(pages)}")

            normalized_tag = (tag or "").strip()
            if normalized_tag:
                self._emit(progress_callback, f"Aplicando tag a los chunks: {normalized_tag}")

            self._emit(progress_callback, "Chunking de documentos...")
            chunks = self.chunker.chunk_pages(pages, tag=normalized_tag or None)
            self._emit(progress_callback, f"Chunks generados: {len(chunks)}")
            if not chunks:
                self._emit(progress_callback, "Indexado completado: no hay chunks para guardar.")
                return 0

            self._emit(
                progress_callback,
                f"Generando embeddings en lotes de {max(1, self.settings.embedding_batch_size)}...",
            )
            with self.coordinator.acquire(
                "embeddings",
                workload="index",
                timeout=self.settings.parser_job_timeout,
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

            self._emit(progress_callback, f"Guardando {len(chunks)} chunks en LanceDB...")
            self.store.replace_chunks(chunks, embeddings)

            if self.settings.hybrid_search_enabled:
                self._emit(progress_callback, "Guardando índice BM25 para búsqueda híbrida...")
                self.bm25_store.replace_chunks(chunks)

            self._emit(
                progress_callback,
                f"Indexado completado en tabla '{self.settings.lancedb_table}'.",
            )
            return len(chunks)
    
    @staticmethod
    def _is_contextual_retrieval_query(query: str) -> bool:
        """Detecta queries contextuales generadas para embeddings, no para BM25."""
        return (
            "Previous user question:" in query
            or "Latest follow-up question:" in query
        )

    @staticmethod
    def _build_retrieval_queries(
        question: str,
        rewritten_question: str,
        messages: list[dict[str, str]] | None = None,
    ) -> list[str]:
        """Construye variantes de consulta para recuperar mejor follow-ups cortas."""
        queries: list[str] = []

        for candidate in (rewritten_question, question):
            normalized_candidate = candidate.strip()
            if normalized_candidate and normalized_candidate not in queries:
                queries.append(normalized_candidate)

        normalized_messages = [
            {
                "role": str(message.get("role", "")).strip(),
                "content": str(message.get("content", "")).strip(),
            }
            for message in (messages or [])
            if str(message.get("role", "")).strip() and str(message.get("content", "")).strip()
        ]
        user_messages = [message["content"] for message in normalized_messages if message["role"] == "user"]
        previous_user_message = user_messages[-1] if user_messages else ""

        if previous_user_message:
            contextual_query = (
                f"Previous user question: {previous_user_message}\n"
                f"Latest follow-up question: {question.strip()}"
            )
            if contextual_query not in queries:
                queries.append(contextual_query)

        return queries

    @staticmethod
    def _match_key(match: dict[str, Any]) -> tuple[Any, ...]:
        """Identificador estable de un chunk para fusionar resultados."""
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
        """Convierte distancia vectorial en score donde mayor es mejor."""
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
        """Fusiona resultados vectoriales quedándose con la mejor distancia."""
        merged: dict[tuple[Any, ...], dict[str, Any]] = {}

        for matches in match_groups:
            for index, match in enumerate(matches):
                key = cls._match_key(match)
                distance = match.get("_distance")
                normalized_distance = (
                    float(distance)
                    if isinstance(distance, (int, float))
                    else float("inf")
                )

                candidate = {
                    **match,
                    "_rank": index,
                    "_normalized_distance": normalized_distance,
                }

                current = merged.get(key)
                if current is None:
                    merged[key] = candidate
                    continue

                current_distance = current["_normalized_distance"]
                current_rank = current["_rank"]

                if (normalized_distance, index) < (current_distance, current_rank):
                    merged[key] = candidate

        ordered_matches = sorted(
            merged.values(),
            key=lambda item: (item["_normalized_distance"], item["_rank"]),
        )

        return [
            {
                key: value
                for key, value in match.items()
                if key not in {"_rank", "_normalized_distance"}
            }
            for match in ordered_matches[:top_k]
        ]
    
    @staticmethod
    def _citation_from_match(match: dict[str, Any], source_id: str) -> dict[str, Any]:
        """Crea la cita de un chunk recuperado con un ID auditable por el LLM."""
        return {
            "source_id": source_id,
            "chunk_id": match.get("chunk_id") or match.get("id"),
            "document_id": match["document_id"],
            "kind": match.get("kind"),
            "table_id": match.get("table_id"),
            "source": match["source"],
            "source_path": match["source_path"],
            "source_type": match["source_type"],
            "page_start": match["page_start"],
            "page_end": match["page_end"],
            "source_pages": list(match.get("source_pages") or []),
            "chunk_index": match["chunk_index"],
            "ocr_used": bool(match.get("ocr_used", False)),
            "tag": match.get("tag") or None,
        }

    @staticmethod
    def _page_label(page_start: int, page_end: int) -> str:
        return f"p.{page_start}" if page_start == page_end else f"pp.{page_start}-{page_end}"

    @staticmethod
    def _source_option_from_match(match: dict[str, Any], source_id: str) -> dict[str, Any]:
        """Prepara un candidato de fuente para la auditoría posterior a la respuesta."""
        return {
            "source_id": source_id,
            "chunk_id": match.get("chunk_id") or match.get("id"),
            "document_id": match.get("document_id"),
            "kind": match.get("kind"),
            "table_id": match.get("table_id"),
            "source": match.get("source"),
            "source_path": match.get("source_path"),
            "source_type": match.get("source_type"),
            "page_start": match.get("page_start"),
            "page_end": match.get("page_end"),
            "source_pages": list(match.get("source_pages") or []),
            "chunk_index": match.get("chunk_index"),
            "tag": match.get("tag") or None,
            "text": match.get("text", ""),
        }

    @staticmethod
    def _filter_citations_by_source_ids(
        citations: list[dict[str, Any]],
        used_source_ids: list[str],
    ) -> list[dict[str, Any]]:
        """Filtra las citas recuperadas dejando solo las usadas por la respuesta."""
        allowed_ids = set(used_source_ids)
        if not allowed_ids:
            return []

        return [
            citation
            for citation in citations
            if str(citation.get("source_id", "")).strip() in allowed_ids
        ]

    def _select_used_citations(
        self,
        question: str,
        answer: str,
        generation_inputs: dict[str, Any],
        messages: list[dict[str, str]] | None = None,
    ) -> list[dict[str, Any]]:
        """Audita la respuesta final y devuelve solo las citas realmente usadas."""
        used_source_ids = self.client.select_used_source_ids(
            question=question,
            answer=answer,
            source_options=generation_inputs["source_options"],
            messages=messages,
        )
        return self._filter_citations_by_source_ids(
            generation_inputs["citations"],
            used_source_ids,
        )

    @classmethod
    def _merge_hybrid_matches(
        cls,
        vector_match_groups: list[list[dict[str, Any]]],
        bm25_match_groups: list[list[dict[str, Any]]],
        top_k: int,
        rrf_k: int = 60,
        **_legacy_options: Any,
    ) -> list[dict[str, Any]]:
        """Fusiona rankings con RRF sin mezclar escalas vectoriales y BM25."""
        merged: dict[tuple[Any, ...], dict[str, Any]] = {}

        def best_ranks(
            groups: list[list[dict[str, Any]]],
        ) -> dict[tuple[Any, ...], tuple[int, dict[str, Any]]]:
            best: dict[tuple[Any, ...], tuple[int, dict[str, Any]]] = {}
            for matches in groups:
                for rank, match in enumerate(matches, start=1):
                    key = cls._match_key(match)
                    current = best.get(key)
                    if current is None or rank < current[0]:
                        best[key] = (rank, match)
            return best

        for source, groups in (
            ("vector", vector_match_groups),
            ("bm25", bm25_match_groups),
        ):
            for key, (rank, match) in best_ranks(groups).items():
                item = merged.get(key)
                if item is None:
                    item = dict(match)
                    item["_retrieval_sources"] = []
                    merged[key] = item
                else:
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
                1.0 / (rrf_k + int(item[rank_field]))
                for rank_field in ("_vector_rank", "_bm25_rank")
                if rank_field in item
            )

        ordered_matches = sorted(
            merged.values(),
            key=lambda item: (
                -float(item.get("_rrf_score", 0.0)),
                int(item.get("_vector_rank", 10**9)),
                int(item.get("_bm25_rank", 10**9)),
                str(item.get("chunk_id") or item.get("id") or ""),
            ),
        )

        return ordered_matches[:top_k]

    def _prepare_generation_inputs(
        self,
        question: str,
        top_k: int | None = None,
        messages: list[dict[str, str]] | None = None,
        tag: str | None = None,
    ) -> dict[str, Any]:
        """Recupera el contexto RAG y genera metadatos compartidos."""
        top_k = top_k or self.settings.effective_retrieval_top_k
        retrieval_query = self.client.rewrite_question_for_retrieval(question, messages=messages)
        retrieval_queries = self._build_retrieval_queries(question, retrieval_query, messages=messages)


        # Debug de consultas de retrieval, ver en consola en la que lanzamos la API

        print("\n" + "=" * 80, flush=True)
        print("[RAG DEBUG] Nueva interacción", flush=True)
        print(f"[RAG DEBUG] Pregunta original: {question}", flush=True)
        print(f"[RAG DEBUG] retrieval_query: {retrieval_query}", flush=True)
        print("[RAG DEBUG] retrieval_queries:", flush=True)

        for index, query in enumerate(retrieval_queries, start=1):
            print(f"  {index}. {query}", flush=True)

            print("=" * 80 + "\n", flush=True)
        
        # FIN debug

        normalized_tag = (tag or "").strip()
        with self.coordinator.acquire(
            "embeddings",
            workload="query",
            timeout=self.settings.model_request_timeout,
        ):
            self._ensure_models(("embeddings",))
            try:
                query_vectors = self.client.embed_texts(retrieval_queries)
                vector_match_groups = [
                    self.store.search(
                        query_vector,
                        top_k=self.settings.vector_candidates,
                        tag=normalized_tag or None,
                    )
                    for query_vector in query_vectors
                ]
            finally:
                # Embeddings se descarga antes de solicitar el lease de chat.
                self._stop_models(("embeddings",))
                self._release_model_memory()

        if self.settings.hybrid_search_enabled:
            bm25_queries = [
                query
                for query in retrieval_queries
                if not self._is_contextual_retrieval_query(query)
            ]

            bm25_match_groups = [
                self.bm25_store.search(
                    query,
                    top_k=self.settings.bm25_candidates,
                    tag=normalized_tag or None,
                )
                for query in bm25_queries
            ]

            matches = self._merge_hybrid_matches(
                vector_match_groups=vector_match_groups,
                bm25_match_groups=bm25_match_groups,
                top_k=top_k,
                rrf_k=self.settings.rrf_k,
            )
        else:
            matches = self._merge_matches(
                match_groups=vector_match_groups,
                top_k=top_k,
            )

        # Debug de matches seleccionados, ver en consola en la que lanzamos la API

        print("\n" + "=" * 80, flush=True)
        
        print("[HYBRID DEBUG] retrieval_queries:", flush=True)
        for index, query in enumerate(retrieval_queries, start=1):
            print(f"  {index}. {query!r}", flush=True)

        if self.settings.hybrid_search_enabled:
            print("[HYBRID DEBUG] bm25_queries:", flush=True)
            for index, query in enumerate(bm25_queries, start=1):
                print(f"  {index}. {query!r}", flush=True)

        print("[HYBRID DEBUG] selected matches:", flush=True)
        for index, match in enumerate(matches, start=1):
            print(
                {
                    "rank": index,
                    "source": match.get("source"),
                    "page": f"{match.get('page_start')}-{match.get('page_end')}",
                    "chunk_index": match.get("chunk_index"),
                    "_distance": match.get("_distance"),
                    "_vector_score": match.get("_vector_score"),
                    "_bm25_raw_score": match.get("_bm25_raw_score"),
                    "_bm25_score": match.get("_bm25_score"),
                    "_vector_rank": match.get("_vector_rank"),
                    "_bm25_rank": match.get("_bm25_rank"),
                    "_rrf_score": match.get("_rrf_score"),
                    "_retrieval_sources": match.get("_retrieval_sources"),
                },
                flush=True,
            )
        print("=" * 80 + "\n", flush=True)

        # Fin debug

        context_blocks = []
        citations = []
        source_options = []
        for index, match in enumerate(matches, start=1):
            source_id = f"S{index}"
            source = match["source"]
            page_start = match["page_start"]
            page_end = match["page_end"]
            page_label = self._page_label(page_start, page_end)
            context_blocks.append(f"[{source_id}] {source} {page_label}\n{match['text']}")
            citations.append(self._citation_from_match(match, source_id))
            source_options.append(self._source_option_from_match(match, source_id))

        return {
            "context_blocks": context_blocks,
            "citations": citations,
            "source_options": source_options,
            "matches": matches,
        }

    def ask(
        self,
        question: str,
        top_k: int | None = None,
        messages: list[dict[str, str]] | None = None,
        tag: str | None = None,
        enable_reasoning: bool = False,
    ) -> dict[str, Any]:
        """Responde una pregunta usando retrieval + generación no streaming."""
        generation_inputs = self._prepare_generation_inputs(question, top_k=top_k, messages=messages, tag=tag)
        with self.coordinator.acquire(
            "chat",
            workload="query",
            timeout=self.settings.model_request_timeout,
        ):
            self._ensure_models(("chat",))
            try:
                generation = self.client.generate_answer(
                    question,
                    generation_inputs["context_blocks"],
                    messages=messages,
                    enable_reasoning=enable_reasoning,
                )
                # La auditoría comparte exactamente la misma carga de chat.
                citations = self._select_used_citations(
                    question=question,
                    answer=generation["answer"],
                    generation_inputs=generation_inputs,
                    messages=messages,
                )
            finally:
                if self.supervisor is not None:
                    self.supervisor.schedule_idle_stop("chat")
        return {
            "answer": generation["answer"],
            "reasoning": generation["reasoning"],
            "citations": citations,
            "matches": generation_inputs["matches"],
        }

    def stream_answer(
        self,
        question: str,
        top_k: int | None = None,
        messages: list[dict[str, str]] | None = None,
        tag: str | None = None,
        enable_reasoning: bool = False,
    ) -> dict[str, Any]:
        """Responde una pregunta en streaming."""
        generation_inputs = self._prepare_generation_inputs(question, top_k=top_k, messages=messages, tag=tag)
        chat_lease = self.coordinator.acquire(
            "chat",
            workload="query",
            timeout=self.settings.model_request_timeout,
        )
        try:
            self._ensure_models(("chat",))
            primary_stream = self.client.stream_answer(
                question,
                generation_inputs["context_blocks"],
                messages=messages,
                enable_reasoning=enable_reasoning,
            )
        except BaseException:
            chat_lease.release()
            raise

        cleanup_lock = threading.Lock()
        cleaned = False

        def cleanup() -> None:
            nonlocal cleaned
            with cleanup_lock:
                if cleaned:
                    return
                cleaned = True
                try:
                    if self.supervisor is not None:
                        self.supervisor.schedule_idle_stop("chat")
                finally:
                    chat_lease.release()

        def guarded_stream(stream):
            completed = False
            try:
                for event in stream:
                    yield event
                completed = True
            finally:
                # En flujo normal se conserva el lease hasta auditar las citas.
                # Un cierre/una excepción prematuros lo libera inmediatamente.
                if not completed:
                    cleanup()

        def fallback_stream():
            stream = self.client.stream_answer(
                question,
                generation_inputs["context_blocks"],
                messages=messages,
                enable_reasoning=False,
            )
            return guarded_stream(stream)

        def resolve_citations(answer: str) -> list[dict[str, Any]]:
            try:
                return self._select_used_citations(
                    question=question,
                    answer=answer,
                    generation_inputs=generation_inputs,
                    messages=messages,
                )
            finally:
                cleanup()

        return {
            "answer_stream": guarded_stream(primary_stream),
            "fallback_stream": fallback_stream,
            "resolve_citations": resolve_citations,
            "close": cleanup,
            "matches": generation_inputs["matches"],
        }
