"""Índice léxico BM25 para búsqueda híbrida.

Este módulo guarda un corpus ligero de chunks en disco y construye un índice
BM25 en memoria cuando se necesita buscar. El corpus se reemplaza completo en
cada indexado, igual que ocurre con LanceDB.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rank_bm25 import BM25Okapi

from rag_cliente.index_schema import INDEX_SCHEMA_VERSION, incompatible_index_message

if TYPE_CHECKING:
    from rag_cliente.indexer import ChunkRecord

_TOKEN_PATTERN = re.compile(r"[\wáéíóúüñÁÉÍÓÚÜÑ]+", flags=re.UNICODE)


def tokenize_for_bm25(text: str) -> list[str]:
    """Tokeniza texto para BM25 de forma simple y estable.

    - minúsculas
    - conserva letras acentuadas y números
    - elimina puntuación
    """
    return [token.lower() for token in _TOKEN_PATTERN.findall(text or "")]


class BM25Store:
    """Índice BM25 persistido como JSON y reconstruido en memoria al buscar."""

    def __init__(self, index_path: Path) -> None:
        self.index_path = index_path
        self._rows: list[dict[str, Any]] = []
        self._bm25: BM25Okapi | None = None
        self._loaded = False

    def replace_chunks(self, chunks: list[ChunkRecord]) -> None:
        """Reemplaza el corpus BM25 por los chunks recibidos."""
        self.index_path.parent.mkdir(parents=True, exist_ok=True)

        rows = [asdict(chunk) for chunk in chunks]
        payload = {
            "schema_version": INDEX_SCHEMA_VERSION,
            "rows": rows,
        }

        self.index_path.write_text(
            json.dumps(payload, ensure_ascii=False),
            encoding="utf-8",
        )

        self._rows = rows
        self._bm25 = self._build_index(rows)
        self._loaded = True

    def search(
        self,
        query: str,
        top_k: int,
        tag: str | None = None,
        document_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Busca chunks por coincidencia léxica BM25.

        Devuelve una lista de diccionarios compatibles con los matches de
        LanceDB, añadiendo `_bm25_score` y `_bm25_rank`.
        """
        if top_k <= 0:
            return []

        self._ensure_loaded()

        if self._bm25 is None or not self._rows:
            return []

        query_tokens = tokenize_for_bm25(query)
        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)
        normalized_tag = (tag or "").strip()
        normalized_document_id = (document_id or "").strip()

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: float(scores[index]),
            reverse=True,
        )

        matches: list[dict[str, Any]] = []

        for rank, row_index in enumerate(ranked_indices):
            score = float(scores[row_index])
            row = self._rows[row_index]

            if score <= 0.0:
                continue

            if normalized_tag and str(row.get("tag", "")).strip() != normalized_tag:
                continue
            if (
                normalized_document_id
                and str(row.get("document_id", "")).strip() != normalized_document_id
            ):
                continue

            match = dict(row)
            match["_bm25_score"] = score
            match["_bm25_rank"] = rank + 1

            matches.append(match)

            if len(matches) >= top_k:
                break

        return matches

    def _ensure_loaded(self) -> None:
        """Carga el corpus BM25 desde disco si todavía no está en memoria."""
        if self._loaded:
            return

        if not self.index_path.exists():
            self._rows = []
            self._bm25 = None
            self._loaded = True
            return

        payload = json.loads(self.index_path.read_text(encoding="utf-8"))
        found_version = payload.get("schema_version")
        if found_version != INDEX_SCHEMA_VERSION:
            raise RuntimeError(incompatible_index_message(found_version))
        rows = payload.get("rows", [])

        self._rows = [dict(row) for row in rows if isinstance(row, dict)]
        self._bm25 = self._build_index(self._rows)
        self._loaded = True

    @staticmethod
    def _build_index(rows: list[dict[str, Any]]) -> BM25Okapi | None:
        """Construye el índice BM25 en memoria."""
        tokenized_corpus = [
            tokenize_for_bm25(str(row.get("text", "")))
            for row in rows
        ]

        if not any(tokenized_corpus):
            return None

        return BM25Okapi(tokenized_corpus)
