"""Persistencia vectorial LanceDB para chunks Markdown por página."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

import lancedb
import pyarrow as pa

from rag_cliente.index_schema import INDEX_SCHEMA_VERSION, incompatible_index_message

if TYPE_CHECKING:
    from rag_cliente.indexer import ChunkRecord


def build_schema(vector_dim: int) -> pa.Schema:
    return pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field("chunk_id", pa.string()),
            pa.field("document_id", pa.string()),
            pa.field("text", pa.string()),
            pa.field("source", pa.string()),
            pa.field("source_path", pa.string()),
            pa.field("source_type", pa.string()),
            pa.field("page_start", pa.int32()),
            pa.field("page_end", pa.int32()),
            pa.field("source_pages", pa.list_(pa.int32())),
            pa.field("chunk_index", pa.int32()),
            pa.field("page_chunk_index", pa.int32()),
            pa.field("token_count", pa.int32()),
            pa.field("oversize", pa.bool_()),
            pa.field("parser_model", pa.string()),
            pa.field("prompt_version", pa.string()),
            pa.field("source_sha256", pa.string()),
            pa.field("schema_version", pa.int32()),
            pa.field("tag", pa.string()),
            pa.field("metadata", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), vector_dim)),
        ],
        metadata={b"rag_index_schema_version": str(INDEX_SCHEMA_VERSION).encode("ascii")},
    )


def _escape_literal(value: str) -> str:
    return value.replace("'", "''")


class LanceDBStore:
    def __init__(self, uri: Path, table_name: str) -> None:
        self.db = lancedb.connect(str(uri))
        self.table_name = table_name

    def _table_names(self) -> list[str]:
        return list(self.db.list_tables().tables)

    @staticmethod
    def _deserialize(row: dict) -> dict:
        item = dict(row)
        value = item.get("metadata")
        if isinstance(value, str):
            try:
                item["metadata"] = json.loads(value)
            except json.JSONDecodeError:
                item["metadata"] = {}
        return item

    @staticmethod
    def _validate_schema(table) -> None:
        required = {
            "chunk_id",
            "page_chunk_index",
            "parser_model",
            "prompt_version",
            "source_sha256",
            "schema_version",
        }
        missing = sorted(required - set(table.schema.names))
        if missing:
            raise RuntimeError(incompatible_index_message(f"faltan campos {missing}"))
        metadata = table.schema.metadata or {}
        declared = metadata.get(b"rag_index_schema_version")
        if declared is None:
            raise RuntimeError(incompatible_index_message("sin versión declarada"))
        try:
            version = int(declared.decode("ascii"))
        except (AttributeError, UnicodeDecodeError, ValueError):
            version = declared
        if version != INDEX_SCHEMA_VERSION:
            raise RuntimeError(incompatible_index_message(version))

    def replace_chunks(
        self,
        chunks: list[ChunkRecord],
        embeddings: list[list[float]],
    ) -> None:
        if not chunks:
            return
        rows = []
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            row = asdict(chunk)
            row["metadata"] = json.dumps(
                row.get("metadata") or {},
                ensure_ascii=False,
                sort_keys=True,
            )
            row["vector"] = [float(value) for value in embedding]
            rows.append(row)
        if self.table_name in self._table_names():
            self.db.drop_table(self.table_name)
        self.db.create_table(
            self.table_name,
            data=rows,
            schema=build_schema(len(embeddings[0])),
        )

    def search(
        self,
        query_vector: list[float],
        top_k: int,
        tag: str | None = None,
        document_id: str | None = None,
    ) -> list[dict]:
        if self.table_name not in self._table_names():
            raise RuntimeError(
                f"LanceDB table '{self.table_name}' does not exist yet. Run the index command first."
            )
        table = self.db.open_table(self.table_name)
        self._validate_schema(table)
        query = table.search(query_vector)
        filters: list[str] = []
        if (tag or "").strip():
            filters.append(f"tag = '{_escape_literal(tag.strip())}'")
        if (document_id or "").strip():
            filters.append(
                f"document_id = '{_escape_literal(document_id.strip())}'"
            )
        if filters:
            query = query.where(" AND ".join(filters))
        return [
            self._deserialize(dict(item))
            for item in query.limit(top_k).to_list()
        ]

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
        self._validate_schema(table)
        rows = []
        for row in table.to_arrow().to_pylist():
            item = self._deserialize(dict(row))
            if not include_vector:
                item.pop("vector", None)
            rows.append(item)
        return rows
