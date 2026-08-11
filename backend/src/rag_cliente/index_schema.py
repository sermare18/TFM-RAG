"""Contrato versionado compartido por LanceDB y el corpus BM25."""

from __future__ import annotations

INDEX_SCHEMA_VERSION = 3


def incompatible_index_message(found: object = "desconocida") -> str:
    return (
        "El índice existente usa un esquema incompatible "
        f"(encontrado={found}, requerido={INDEX_SCHEMA_VERSION}). "
        "Ejecuta 'rag.bat index' para reindexar todos los documentos."
    )
