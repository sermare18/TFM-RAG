"""Capa de almacenamiento vectorial sobre LanceDB.

Este módulo encapsula dos responsabilidades:
- definir el esquema de la tabla vectorial
- ofrecer operaciones de escritura y búsqueda sobre LanceDB

Decisiones de diseño:
- Se almacena una tabla denormalizada con texto, metadatos y vector. (Significa que cada fila ya guarda todo junto: el texto del chunk, sus metadatos, su embedding)
- El indexado actual reemplaza por completo la tabla existente.
- Esta capa devuelve candidatos vectoriales; `pipeline.py` los fusiona con BM25
  mediante RRF.

Limitaciones:
- No hay inserción incremental ni upsert. (p.ej., si existe X documento, actualízalo; si no existe, insértalo)
- Existe filtro simple por `tag`; otros filtros de metadatos quedan fuera.
- No se aplica umbral mínimo de similitud; siempre se devuelven resultados. (p.ej., solo devuelve chunks si su similitud es mayor que 0.8)
- La fusión híbrida no se hace aquí para no mezclar responsabilidades ni
  escalas de score incompatibles.
"""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import TYPE_CHECKING

import lancedb
import pyarrow as pa

from rag_cliente.index_schema import INDEX_SCHEMA_VERSION, incompatible_index_message

if TYPE_CHECKING:
    # Importo ChunkRecord solo para tipos y evito cargar el stack de chunking en el viewer.
    from rag_cliente.indexer import ChunkRecord


def build_schema(vector_dim: int) -> pa.Schema:
    """Construye el esquema Arrow usado por la tabla de LanceDB.

    `vector_dim` se deduce en tiempo de indexación a partir del embedding real
    devuelto por el endpoint configurado.
    """
    return pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field("chunk_id", pa.string()),
            pa.field("document_id", pa.string()),
            pa.field("kind", pa.string()),
            pa.field("text", pa.string()),
            pa.field("html", pa.string()),
            pa.field("source", pa.string()),
            pa.field("source_path", pa.string()),
            pa.field("source_type", pa.string()),
            pa.field("section_path", pa.list_(pa.string())),
            pa.field("page_start", pa.int32()),
            pa.field("page_end", pa.int32()),
            pa.field("source_pages", pa.list_(pa.int32())),
            pa.field("source_spans", pa.string()),
            pa.field("source_block_ids", pa.list_(pa.string())),
            pa.field("table_id", pa.string()),
            pa.field("token_count", pa.int32()),
            pa.field("oversize", pa.bool_()),
            pa.field("parser_profile", pa.string()),
            pa.field("marker_version", pa.string()),
            pa.field("provenance", pa.string()),
            pa.field("schema_version", pa.int32()),
            pa.field("chunk_index", pa.int32()),
            pa.field("ocr_used", pa.bool_()),
            pa.field("tag", pa.string()),
            pa.field("metadata", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), vector_dim)),
        ],
        metadata={b"rag_index_schema_version": str(INDEX_SCHEMA_VERSION).encode("ascii")},
    )


def _escape_lancedb_string_literal(value: str) -> str:
    """Escapa comillas simples para construir filtros SQL simples de LanceDB."""
    return value.replace("'", "''")


class LanceDBStore:
    """Pequeño wrapper de LanceDB usado por el pipeline RAG."""

    def __init__(self, uri: Path, table_name: str) -> None:
        self.db = lancedb.connect(str(uri))
        self.table_name = table_name

    def _table_names(self) -> list[str]:
        return list(self.db.list_tables().tables)

    @staticmethod
    def _deserialize_row(row: dict) -> dict:
        item = dict(row)
        for field_name, default in (("source_spans", []), ("metadata", {})):
            value = item.get(field_name)
            if isinstance(value, str):
                try:
                    item[field_name] = json.loads(value)
                except json.JSONDecodeError:
                    item[field_name] = default
        return item

    @staticmethod
    def _validate_table_schema(table) -> None:
        schema = table.schema
        names = set(getattr(schema, "names", []))
        required = {
            "chunk_id",
            "kind",
            "html",
            "source_pages",
            "source_spans",
            "source_block_ids",
            "schema_version",
        }
        missing = sorted(required - names)
        if missing:
            raise RuntimeError(incompatible_index_message(f"faltan campos {missing}"))

        schema_metadata = getattr(schema, "metadata", None) or {}
        declared_version = schema_metadata.get(b"rag_index_schema_version")
        if declared_version is None:
            raise RuntimeError(incompatible_index_message("sin versión declarada"))
        try:
            parsed_declared_version = int(declared_version.decode("ascii"))
        except (AttributeError, UnicodeDecodeError, ValueError):
            parsed_declared_version = declared_version
        if parsed_declared_version != INDEX_SCHEMA_VERSION:
            raise RuntimeError(incompatible_index_message(parsed_declared_version))

    def replace_chunks(self, chunks: list[ChunkRecord], embeddings: list[list[float]]) -> None:
        """Reemplaza el contenido completo de la tabla por los chunks recibidos.

        Esta operación es destructiva: si la tabla ya existe, se elimina y se
        vuelve a crear. Es una estrategia simple y fácil de entender, pero no
        adecuada para catálogos grandes o actualizaciones parciales.
        """
        if not chunks:
            return

        # Prepara una lista vacía donde va a meter las filas que se guardarán en la base vectorial
        # Cada fila será un diccionario con: texto del chunk, metadatos, vector embedding
        rows = []
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            row = asdict(chunk)
            row["source_spans"] = json.dumps(
                row.get("source_spans", []),
                ensure_ascii=False,
                sort_keys=True,
            )
            row["metadata"] = json.dumps(
                row.get("metadata") or {},
                ensure_ascii=False,
                sort_keys=True,
            )
            row["vector"] = [float(value) for value in embedding]
            rows.append(row)

        # Si la tabla ya existe, borrarla
        if self.table_name in self._table_names():
            self.db.drop_table(self.table_name)

        # Construir el esquema
        schema = build_schema(len(embeddings[0]))
        # Crear la tabla nueva
        self.db.create_table(self.table_name, data=rows, schema=schema)

    def search(
        self,
        query_vector: list[float],
        top_k: int,
        tag: str | None = None,
        document_id: str | None = None,
    ) -> list[dict]:
        """Busca los chunks más próximos al embedding de consulta."""
        if self.table_name not in self._table_names():
            raise RuntimeError(
                f"LanceDB table '{self.table_name}' does not exist yet. Run the index command first."
            )
        table = self.db.open_table(self.table_name)
        self._validate_table_schema(table)
        query = table.search(query_vector)
        filters: list[str] = []
        normalized_tag = (tag or "").strip()
        if normalized_tag:
            schema_names = set(getattr(table.schema, "names", []))
            if "tag" not in schema_names:
                raise RuntimeError(
                    "LanceDB table does not include the 'tag' column. Reindex documents with the updated indexer."
                )
            escaped_tag = _escape_lancedb_string_literal(normalized_tag)
            filters.append(f"tag = '{escaped_tag}'")

        normalized_document_id = (document_id or "").strip()
        if normalized_document_id:
            escaped_document_id = _escape_lancedb_string_literal(normalized_document_id)
            filters.append(f"document_id = '{escaped_document_id}'")

        if filters:
            query = query.where(" AND ".join(filters))
        query = query.limit(top_k)

        results = query.to_list()

        # Devuelve una lista de diccionarios, donde cada diccionario representa un chunk recuperado de LanceDB
        return [self._deserialize_row(dict(item)) for item in results]
    
    def list_tables(self) -> list[str]:
        return sorted(self._table_names())

    def table_exists(self) -> bool:
        return self.table_name in self._table_names()

    def list_chunks(self, include_vector: bool = False) -> list[dict]:
        if not self.table_exists():
            raise RuntimeError(
                f"LanceDB table '{self.table_name}' does not exist yet. Run the index command first."
            )

        table = self.db.open_table(self.table_name)
        self._validate_table_schema(table)
        rows = table.to_arrow().to_pylist()

        cleaned_rows: list[dict] = []
        for row in rows:
            item = self._deserialize_row(dict(row))
            if not include_vector:
                item.pop("vector", None)
            cleaned_rows.append(item)

        return cleaned_rows
