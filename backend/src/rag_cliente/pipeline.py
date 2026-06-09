"""Orquestación del flujo completo RAG.

Conecta carga de documentos, chunking, embeddings, almacenamiento vectorial,
recuperación y generación.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from rag_cliente.bm25_store import BM25Store
from rag_cliente.config import Settings
from rag_cliente.indexer import PdfChunker
from rag_cliente.llm_client import LlamaCppClient
from rag_cliente.pdf_loader import load_documents_from_directory
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
        pages = load_documents_from_directory(
            doc_dir,
            settings=self.settings,
            progress_callback=progress_callback,
        )
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
        embeddings = self.client.embed_texts(
            [chunk.text for chunk in chunks],
            progress_callback=progress_callback,
        )

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
            "document_id": match["document_id"],
            "source": match["source"],
            "source_path": match["source_path"],
            "source_type": match["source_type"],
            "page_start": match["page_start"],
            "page_end": match["page_end"],
            "chunk_index": match["chunk_index"],
            "ocr_used": bool(match.get("ocr_used", False)),
            "tag": match.get("tag") or None,
        }

    @staticmethod
    def _source_option_from_match(match: dict[str, Any], source_id: str) -> dict[str, Any]:
        """Prepara un candidato de fuente para la auditoría posterior a la respuesta."""
        return {
            "source_id": source_id,
            "document_id": match.get("document_id"),
            "source": match.get("source"),
            "source_path": match.get("source_path"),
            "source_type": match.get("source_type"),
            "page_start": match.get("page_start"),
            "page_end": match.get("page_end"),
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
        vector_weight: float,
        bm25_weight: float,
        bm25_min_raw_score: float,
    ) -> list[dict[str, Any]]:
        """Fusiona resultados vectoriales y BM25 en un ranking híbrido.

        - Vector: LanceDB devuelve `_distance`; se transforma a score con
          `1 / (1 + distance)`.
        - BM25: el score de cada grupo se normaliza dividiendo por el máximo
          score BM25 de ese grupo.
        - Si un chunk aparece varias veces, se conserva su mejor score por vía.
        """
        merged: dict[tuple[Any, ...], dict[str, Any]] = {}

        def ensure_item(match: dict[str, Any], rank: int) -> dict[str, Any]:
            key = cls._match_key(match)
            item = merged.get(key)

            if item is None:
                item = dict(match)
                item["_vector_score"] = 0.0
                item["_bm25_score"] = 0.0
                item["_hybrid_rank"] = rank
                item["_retrieval_sources"] = []
                merged[key] = item
            else:
                item["_hybrid_rank"] = min(
                    int(item.get("_hybrid_rank", rank)),
                    rank,
                )

            return item

        for matches in vector_match_groups:
            for rank, match in enumerate(matches):
                item = ensure_item(match, rank)
                score = cls._vector_score(match)

                if score > float(item.get("_vector_score", 0.0)):
                    previous_sources = item.get("_retrieval_sources", [])
                    item.update(match)
                    item["_retrieval_sources"] = previous_sources
                    item["_vector_score"] = score

                if "vector" not in item["_retrieval_sources"]:
                    item["_retrieval_sources"].append("vector")
        
        for matches in bm25_match_groups:
            raw_scores = [
                float(match.get("_bm25_score", 0.0))
                for match in matches
            ]

            max_score = max(raw_scores, default=0.0)

            # Si incluso el mejor BM25 bruto es muy bajo, ignoramos este grupo.
            # Esto evita que la normalización convierta resultados malos en 1.0.
            if max_score < bm25_min_raw_score:
                continue

            for rank, match in enumerate(matches):
                raw_score = float(match.get("_bm25_score", 0.0))

                # Filtra candidatos BM25 débiles antes de normalizar
                if raw_score < bm25_min_raw_score:
                    continue

                item = ensure_item(match, rank)
                normalized_score = raw_score / max_score if max_score > 0 else 0.0
                
                item["_bm25_score"] = max(
                    float(item.get("_bm25_score", 0.0)),
                    normalized_score,
                )
                item["_bm25_raw_score"] = max(
                    float(item.get("_bm25_raw_score", 0.0)),
                    raw_score,
                )
                if "bm25" not in item["_retrieval_sources"]:
                    item["_retrieval_sources"].append("bm25")

        for item in merged.values():
            item["_hybrid_score"] = (
                vector_weight * float(item.get("_vector_score", 0.0))
                + bm25_weight * float(item.get("_bm25_score", 0.0))
            )

        ordered_matches = sorted(
            merged.values(),
            key=lambda item: (
                -float(item.get("_hybrid_score", 0.0)),
                int(item.get("_hybrid_rank", 0)),
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
        top_k = top_k or self.settings.top_k
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

        query_vectors = self.client.embed_texts(retrieval_queries)
        normalized_tag = (tag or "").strip()
        
        vector_match_groups = [
            self.store.search(query_vector, top_k=top_k, tag=normalized_tag or None)
            for query_vector in query_vectors
        ]

        if self.settings.hybrid_search_enabled:
            bm25_top_k = top_k * self.settings.bm25_top_k_multiplier

            bm25_queries = [
                query
                for query in retrieval_queries
                if not self._is_contextual_retrieval_query(query)
            ]

            bm25_match_groups = [
                self.bm25_store.search(query, top_k=bm25_top_k, tag=normalized_tag or None)
                for query in bm25_queries
            ]

            matches = self._merge_hybrid_matches(
                vector_match_groups=vector_match_groups,
                bm25_match_groups=bm25_match_groups,
                top_k=top_k,
                vector_weight=self.settings.vector_weight,
                bm25_weight=self.settings.bm25_weight,
                bm25_min_raw_score=self.settings.bm25_min_raw_score,
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

            print(
                f"[HYBRID DEBUG] bm25_min_raw_score: {self.settings.bm25_min_raw_score}",
                flush=True,
            )

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
                    "_hybrid_score": match.get("_hybrid_score"),
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
            context_blocks.append(f"[{source_id}] {source} p.{page_start}-{page_end}\n{match['text']}")
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
    ) -> dict[str, Any]:
        """Responde una pregunta usando retrieval + generación no streaming."""
        generation_inputs = self._prepare_generation_inputs(question, top_k=top_k, messages=messages, tag=tag)
        generation = self.client.generate_answer(
            question,
            generation_inputs["context_blocks"],
            messages=messages,
        )
        citations = self._select_used_citations(
            question=question,
            answer=generation["answer"],
            generation_inputs=generation_inputs,
            messages=messages,
        )
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
    ) -> dict[str, Any]:
        """Responde una pregunta en streaming."""
        generation_inputs = self._prepare_generation_inputs(question, top_k=top_k, messages=messages, tag=tag)

        return {
            "answer_stream": self.client.stream_answer(
                question,
                generation_inputs["context_blocks"],
                messages=messages,
            ),
            "fallback_response": lambda: self.client.generate_answer(
                question,
                generation_inputs["context_blocks"],
                messages=messages,
            ),
            "resolve_citations": lambda answer: self._select_used_citations(
                question=question,
                answer=answer,
                generation_inputs=generation_inputs,
                messages=messages,
            ),
            "matches": generation_inputs["matches"],
        }
