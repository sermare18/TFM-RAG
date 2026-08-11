"""Chunking Markdown sencillo, determinista y limitado por página."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

from rag_cliente.bedrock_parser import MarkdownDocument, MarkdownPage
from rag_cliente.config import Settings
from rag_cliente.index_schema import INDEX_SCHEMA_VERSION

_TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)


def estimate_token_count(text: str) -> int:
    """Estimación offline estable; evita cargar un tokenizador."""
    return len(_TOKEN_PATTERN.findall(text or ""))


@dataclass(slots=True)
class ChunkRecord:
    id: str
    chunk_id: str
    document_id: str
    text: str
    source: str
    source_path: str
    source_type: str
    page_start: int
    page_end: int
    source_pages: list[int]
    chunk_index: int
    page_chunk_index: int
    token_count: int
    oversize: bool
    parser_model: str
    prompt_version: str
    source_sha256: str
    schema_version: int
    tag: str = ""
    metadata: dict[str, Any] | None = None


def _overlap_tail(lines: list[str], max_tokens: int) -> list[str]:
    if max_tokens <= 0:
        return []
    selected: list[str] = []
    count = 0
    for line in reversed(lines):
        line_tokens = estimate_token_count(line)
        if selected and count + line_tokens > max_tokens:
            break
        if line_tokens > max_tokens:
            break
        selected.append(line)
        count += line_tokens
    return list(reversed(selected))


class MarkdownChunker:
    """Genera uno o varios chunks por página sin cruzar su frontera."""

    def __init__(self, settings: Settings) -> None:
        self.target_tokens = settings.chunk_target_tokens
        self.max_tokens = settings.chunk_max_tokens
        self.overlap_tokens = min(
            settings.chunk_overlap_tokens,
            max(0, settings.chunk_max_tokens - 1),
        )
        self.pages_per_batch = settings.bedrock_pages_per_batch

    @staticmethod
    def _chunk_id(
        document: MarkdownDocument,
        page_number: int,
        page_chunk_index: int,
        text: str,
    ) -> str:
        payload = {
            "document_id": document.document_id,
            "source_sha256": document.source_sha256,
            "page": page_number,
            "page_chunk_index": page_chunk_index,
            "text": text,
            "schema_version": INDEX_SCHEMA_VERSION,
        }
        canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _page_parts(self, page: MarkdownPage) -> list[tuple[str, bool]]:
        lines = page.markdown.splitlines()
        if not lines:
            return []

        parts: list[tuple[str, bool]] = []
        current: list[str] = []
        new_lines = 0

        def emit() -> list[str]:
            nonlocal current, new_lines
            previous = list(current)
            text = "\n".join(current).strip()
            # Do not emit a final chunk made only from overlap carried from the
            # previous chunk.
            if text and (new_lines > 0 or not parts):
                count = estimate_token_count(text)
                parts.append((text, count > self.max_tokens))
            current = []
            new_lines = 0
            return previous

        for line in lines:
            line_tokens = estimate_token_count(line)
            if line_tokens > self.max_tokens:
                emit()
                parts.append((line.strip(), True))
                continue

            if (
                current
                and line.lstrip().startswith("#")
                and estimate_token_count("\n".join(current)) >= self.target_tokens
            ):
                previous = emit()
                current = _overlap_tail(previous, self.overlap_tokens)

            candidate = "\n".join([*current, line])
            if current and estimate_token_count(candidate) > self.max_tokens:
                previous = emit()
                current = _overlap_tail(previous, self.overlap_tokens)
                candidate = "\n".join([*current, line])
                if current and estimate_token_count(candidate) > self.max_tokens:
                    current = []

            current.append(line)
            new_lines += 1
            if (
                not line.strip()
                and estimate_token_count("\n".join(current)) >= self.target_tokens
            ):
                previous = emit()
                current = _overlap_tail(previous, self.overlap_tokens)

        emit()
        return parts

    def chunk_documents(
        self,
        documents: list[MarkdownDocument],
        tag: str | None = None,
    ) -> list[ChunkRecord]:
        chunks: list[ChunkRecord] = []
        explicit_tag = (tag or "").strip() or None

        for document in documents:
            document_tag = explicit_tag or document.tag
            for page in sorted(document.pages, key=lambda item: item.page_number):
                for page_chunk_index, (text, oversize) in enumerate(
                    self._page_parts(page)
                ):
                    chunk_id = self._chunk_id(
                        document,
                        page.page_number,
                        page_chunk_index,
                        text,
                    )
                    chunks.append(
                        ChunkRecord(
                            id=chunk_id,
                            chunk_id=chunk_id,
                            document_id=document.document_id,
                            text=text,
                            source=document.source,
                            source_path=document.source_path,
                            source_type=document.source_type,
                            page_start=page.page_number,
                            page_end=page.page_number,
                            source_pages=[page.page_number],
                            chunk_index=len(chunks),
                            page_chunk_index=page_chunk_index,
                            token_count=estimate_token_count(text),
                            oversize=oversize,
                            parser_model=document.parser_model,
                            prompt_version=document.prompt_version,
                            source_sha256=document.source_sha256,
                            schema_version=INDEX_SCHEMA_VERSION,
                            tag=document_tag,
                            metadata={
                                "batch_index": (
                                    (page.page_number - 1) // self.pages_per_batch
                                ),
                                **document.metadata,
                            },
                        )
                    )
        return chunks
