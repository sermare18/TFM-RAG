"""Capa de almacenamiento vectorial sobre LanceDB.

Este módulo encapsula dos responsabilidades:
- definir el esquema de la tabla vectorial
- ofrecer operaciones de escritura y búsqueda sobre LanceDB

Decisiones de diseño:
- Se almacena una tabla denormalizada con texto, metadatos y vector. (Significa que cada fila ya guarda todo junto: el texto del chunk, sus metadatos, su embedding)
- El indexado actual reemplaza por completo la tabla existente.
- La búsqueda es puramente vectorial y devuelve los `top_k` más cercanos.

Limitaciones:
- No hay inserción incremental ni upsert. (p.ej., si existe X documento, actualízalo; si no existe, insértalo)
- Existe filtro simple por `tag`; otros filtros de metadatos quedan fuera.
- No se aplica umbral mínimo de similitud; siempre se devuelven resultados. (p.ej., solo devuelve chunks si su similitud es mayor que 0.8)
- No hay búsqueda híbrida ni re-ranking. (Búsqueda híbrida: mezcla vectorial (similitud semántica) con léxica (palabras exactas); Re-ranking: Una segunda pasada para reordenar resultados)
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

import lancedb
import pyarrow as pa

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
            pa.field("document_id", pa.string()),
            pa.field("text", pa.string()),
            pa.field("source", pa.string()),
            pa.field("source_path", pa.string()),
            pa.field("source_type", pa.string()),
            pa.field("page_start", pa.int32()),
            pa.field("page_end", pa.int32()),
            pa.field("chunk_index", pa.int32()),
            pa.field("ocr_used", pa.bool_()),
            pa.field("tag", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), vector_dim)),
        ]
    )


def _escape_lancedb_string_literal(value: str) -> str:
    """Escapa comillas simples para construir filtros SQL simples de LanceDB."""
    return value.replace("'", "''")


class LanceDBStore:
    """Pequeño wrapper de LanceDB usado por el pipeline RAG."""

    def __init__(self, uri: Path, table_name: str) -> None:
        self.db = lancedb.connect(str(uri))
        self.table_name = table_name

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
            row["vector"] = [float(value) for value in embedding]
            rows.append(row)

        # Si la tabla ya existe, borrarla
        if self.table_name in self.db.table_names():
            self.db.drop_table(self.table_name)

        # Construir el esquema
        schema = build_schema(len(embeddings[0]))
        # Crear la tabla nueva
        self.db.create_table(self.table_name, data=rows, schema=schema)

    def search(self, query_vector: list[float], top_k: int, tag: str | None = None) -> list[dict]:
        """Busca los chunks más próximos al embedding de consulta."""
        if self.table_name not in self.db.table_names():
            raise RuntimeError(
                f"LanceDB table '{self.table_name}' does not exist yet. Run the index command first."
            )
        table = self.db.open_table(self.table_name)
        query = table.search(query_vector).limit(top_k)
        normalized_tag = (tag or "").strip()
        if normalized_tag:
            schema_names = set(getattr(table.schema, "names", []))
            if "tag" not in schema_names:
                raise RuntimeError(
                    "LanceDB table does not include the 'tag' column. Reindex documents with the updated indexer."
                )
            escaped_tag = _escape_lancedb_string_literal(normalized_tag)
            query = query.where(f"tag = '{escaped_tag}'")

        results = query.to_list()

        # Devuelve una lista de diccionarios, donde cada diccionario representa un chunk recuperado de LanceDB
        return [dict(item) for item in results]
    
    def list_tables(self) -> list[str]:
        return sorted(self.db.table_names())

    def table_exists(self) -> bool:
        return self.table_name in self.db.table_names()

    def list_chunks(self, include_vector: bool = False) -> list[dict]:
        if not self.table_exists():
            raise RuntimeError(
                f"LanceDB table '{self.table_name}' does not exist yet. Run the index command first."
            )

        table = self.db.open_table(self.table_name)
        rows = table.to_arrow().to_pylist()

        cleaned_rows: list[dict] = []
        for row in rows:
            item = dict(row)
            if not include_vector:
                item.pop("vector", None)
            cleaned_rows.append(item)

        return cleaned_rows
