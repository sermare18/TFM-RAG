"""Chunking documental estructurado y consciente de procedencia.

El chunker consume exclusivamente los tipos publicados por Marker. Solo trata
como tabla un elemento que Marker ya haya clasificado como ``Table`` o
``TableOfContents``; nunca reclasifica texto ni une ``Table + Text``.
"""

from __future__ import annotations

import hashlib
import html as html_lib
import json
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Any, Iterable

from rag_cliente.config import Settings
from rag_cliente.index_schema import INDEX_SCHEMA_VERSION
from rag_cliente.pdf_loader import (
    DocumentElement,
    PageDocument,
    ParsedDocument,
    parsed_documents_from_pages,
)

_TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)
_TABLE_KINDS = {"table", "tableofcontents"}
_CONTAINER_KINDS = {"document", "page"}
_HEADING_KINDS = {"sectionheader", "title", "heading"}


@dataclass(slots=True)
class ChunkRecord:
    """Chunk persistible con identidad determinista y procedencia completa."""

    id: str
    chunk_id: str
    document_id: str
    kind: str
    text: str
    html: str
    source: str
    source_path: str
    source_type: str
    section_path: list[str]
    page_start: int
    page_end: int
    source_pages: list[int]
    source_spans: list[dict[str, Any]]
    source_block_ids: list[str]
    table_id: str | None
    token_count: int
    oversize: bool
    parser_profile: str
    marker_version: str
    provenance: str
    schema_version: int
    chunk_index: int
    ocr_used: bool
    tag: str = ""
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class _TableCell:
    text: str
    is_header: bool


@dataclass(frozen=True, slots=True)
class _TableRow:
    cells: tuple[_TableCell, ...]
    html: str
    is_header: bool


class _MarkerTableHTMLParser(HTMLParser):
    """Lee filas/celdas del HTML de un bloque ya clasificado por Marker."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self.rows: list[_TableRow] = []
        self.table_open_tag = "<table>"
        self._table_depth = 0
        self._thead_depth = 0
        self._row_html: list[str] | None = None
        self._row_cells: list[_TableCell] = []
        self._row_in_thead = False
        self._cell_tag: str | None = None
        self._cell_text: list[str] = []

    def _append_raw(self, value: str) -> None:
        if self._row_html is not None:
            self._row_html.append(value)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        normalized = tag.lower()
        raw = self.get_starttag_text() or f"<{tag}>"

        if normalized == "table":
            if self._table_depth == 0:
                self.table_open_tag = raw
            else:
                self._append_raw(raw)
            self._table_depth += 1
            return
        if self._table_depth == 0:
            return

        if normalized == "thead":
            self._thead_depth += 1
            self._append_raw(raw)
            return
        if normalized == "tr" and self._row_html is None:
            self._row_html = [raw]
            self._row_cells = []
            self._row_in_thead = self._thead_depth > 0
            return

        self._append_raw(raw)
        if normalized in {"th", "td"} and self._row_html is not None:
            self._cell_tag = normalized
            self._cell_text = []
        elif normalized == "br" and self._cell_tag is not None:
            self._cell_text.append("\n")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        if tag.lower() not in {"br", "img", "hr", "meta", "link", "input"}:
            self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        normalized = tag.lower()
        if self._table_depth == 0:
            return

        if normalized in {"th", "td"} and self._cell_tag == normalized:
            self._append_raw(f"</{tag}>")
            text = html_lib.unescape("".join(self._cell_text))
            text = re.sub(r"[ \t\r\f\v]+", " ", text)
            text = re.sub(r"\n+", "<br>", text).strip()
            self._row_cells.append(
                _TableCell(
                    text=text,
                    is_header=normalized == "th" or self._row_in_thead,
                )
            )
            self._cell_tag = None
            self._cell_text = []
            return

        if normalized == "tr" and self._row_html is not None:
            self._append_raw(f"</{tag}>")
            if self._row_cells:
                self.rows.append(
                    _TableRow(
                        cells=tuple(self._row_cells),
                        html="".join(self._row_html),
                        is_header=(
                            self._row_in_thead
                            or any(cell.is_header for cell in self._row_cells)
                        ),
                    )
                )
            self._row_html = None
            self._row_cells = []
            self._row_in_thead = False
            return

        if normalized == "thead":
            self._append_raw(f"</{tag}>")
            self._thead_depth = max(0, self._thead_depth - 1)
            return
        if normalized == "table":
            if self._table_depth > 1:
                self._append_raw(f"</{tag}>")
            self._table_depth = max(0, self._table_depth - 1)
            return

        self._append_raw(f"</{tag}>")

    def handle_data(self, data: str) -> None:
        self._append_raw(data)
        if self._cell_tag is not None:
            self._cell_text.append(data)

    def handle_entityref(self, name: str) -> None:
        value = f"&{name};"
        self._append_raw(value)
        if self._cell_tag is not None:
            self._cell_text.append(value)

    def handle_charref(self, name: str) -> None:
        value = f"&#{name};"
        self._append_raw(value)
        if self._cell_tag is not None:
            self._cell_text.append(value)


def estimate_token_count(text: str) -> int:
    """Estimación offline determinista; no carga ningún tokenizador/modelo."""
    return len(_TOKEN_PATTERN.findall(text or ""))


def _unique_strings(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = str(value).strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result


def _unique_ints(values: Iterable[int]) -> list[int]:
    return sorted({int(value) for value in values})


def _unique_spans(values: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, dict):
            continue
        plain = dict(value)
        key = json.dumps(plain, ensure_ascii=False, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        result.append(plain)
    return result


def _iter_chunkable_elements(element: DocumentElement) -> Iterable[DocumentElement]:
    kind = element.kind.strip().lower()
    if kind in _CONTAINER_KINDS and element.children:
        for child in element.children:
            yield from _iter_chunkable_elements(child)
        return
    yield element


def _row_to_markdown(row: _TableRow) -> str:
    cells = [cell.text.replace("|", r"\|").strip() for cell in row.cells]
    return "| " + " | ".join(cells) + " |"


def _table_text(header_rows: list[_TableRow], body_rows: list[_TableRow]) -> str:
    lines = [_row_to_markdown(row) for row in header_rows]
    if header_rows:
        column_count = max(len(row.cells) for row in header_rows)
        lines.append("| " + " | ".join("---" for _ in range(column_count)) + " |")
    lines.extend(_row_to_markdown(row) for row in body_rows)
    return "\n".join(lines).strip()


def _table_html(
    opening_tag: str,
    header_rows: list[_TableRow],
    body_rows: list[_TableRow],
) -> str:
    rows = [*header_rows, *body_rows]
    return opening_tag + "".join(row.html for row in rows) + "</table>"


def _split_text_window(text: str, window: int, overlap: int) -> list[str]:
    matches = list(_TOKEN_PATTERN.finditer(text))
    if not matches:
        return []
    if len(matches) <= window:
        return [text.strip()]

    chunks: list[str] = []
    start = 0
    while start < len(matches):
        end = min(start + window, len(matches))
        char_start = matches[start].start()
        char_end = matches[end - 1].end()
        chunk = text[char_start:char_end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(matches):
            break
        start = max(start + 1, end - overlap)
    return chunks


class PdfChunker:
    """Nombre público histórico para el chunker documental estructurado."""

    def __init__(self, settings: Settings) -> None:
        self.target_tokens = settings.chunk_target_tokens
        self.max_tokens = settings.chunk_max_tokens
        self.overlap_tokens = settings.chunk_overlap_tokens
        self.table_max_tokens = settings.table_chunk_max_tokens

    @staticmethod
    def _can_merge_text(previous: DocumentElement, current: DocumentElement) -> bool:
        if current.kind.strip().lower() in _HEADING_KINDS:
            return False
        if previous.section_path != current.section_path:
            return False
        return previous.page_start <= current.page_start <= previous.page_end + 1

    @staticmethod
    def _ocr_used(document: ParsedDocument, source_pages: list[int]) -> bool:
        usage = document.metadata.get("ocr_used_by_page", {})
        if isinstance(usage, dict):
            return any(bool(usage.get(str(page), usage.get(page, False))) for page in source_pages)
        return False

    @staticmethod
    def _chunk_id(
        document: ParsedDocument,
        *,
        kind: str,
        text: str,
        source_pages: list[int],
        source_block_ids: list[str],
        table_id: str | None,
        document_ordinal: int,
    ) -> str:
        identity = {
            "document_id": document.id,
            "source_path": document.source_path,
            "kind": kind,
            "text": text,
            "source_pages": source_pages,
            "source_block_ids": source_block_ids,
            "table_id": table_id,
            "ordinal": document_ordinal,
        }
        canonical = json.dumps(identity, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _build_record(
        self,
        document: ParsedDocument,
        elements: list[DocumentElement],
        *,
        kind: str,
        text: str,
        html: str,
        table_id: str | None,
        oversize: bool,
        chunk_index: int,
        document_ordinal: int,
        tag: str,
    ) -> ChunkRecord:
        source_pages = _unique_ints(
            page
            for element in elements
            for page in (
                element.source_pages
                or range(element.page_start, element.page_end + 1)
            )
        )
        source_spans = _unique_spans(
            span for element in elements for span in element.source_spans
        )
        source_block_ids = _unique_strings(
            block_id
            for element in elements
            for block_id in ([element.id] if element.id else []) + element.source_block_ids
        )
        page_start = min(source_pages) if source_pages else min(element.page_start for element in elements)
        page_end = max(source_pages) if source_pages else max(element.page_end for element in elements)
        capabilities = document.metadata.get("capabilities", {})
        if not isinstance(capabilities, dict):
            capabilities = {}
        marker_version = str(capabilities.get("marker_version") or "2.0.0")
        parser_profile = str(document.metadata.get("parser_profile") or "unknown")
        provenance_values = _unique_strings(element.provenance for element in elements)
        provenance = "+".join(provenance_values) or "unknown"
        section_path = list(elements[0].section_path) if elements else []
        token_count = estimate_token_count(text)
        chunk_id = self._chunk_id(
            document,
            kind=kind,
            text=text,
            source_pages=source_pages,
            source_block_ids=source_block_ids,
            table_id=table_id,
            document_ordinal=document_ordinal,
        )
        return ChunkRecord(
            id=chunk_id,
            chunk_id=chunk_id,
            document_id=document.id,
            kind=kind,
            text=text,
            html=html,
            source=document.source,
            source_path=document.source_path,
            source_type=document.source_type,
            section_path=section_path,
            page_start=page_start,
            page_end=page_end,
            source_pages=source_pages,
            source_spans=source_spans,
            source_block_ids=source_block_ids,
            table_id=table_id,
            token_count=token_count,
            oversize=oversize,
            parser_profile=parser_profile,
            marker_version=marker_version,
            provenance=provenance,
            schema_version=INDEX_SCHEMA_VERSION,
            chunk_index=chunk_index,
            ocr_used=self._ocr_used(document, source_pages),
            tag=tag,
            metadata={
                "capabilities": dict(capabilities),
                "element_metadata": [dict(element.metadata) for element in elements],
                "token_count_method": "offline_regex_estimate",
            },
        )

    def _table_parts(
        self,
        element: DocumentElement,
    ) -> list[tuple[str, str, bool]]:
        parser = _MarkerTableHTMLParser()
        try:
            parser.feed(element.html)
            parser.close()
        except (ValueError, TypeError):
            parser.rows = []

        if not parser.rows:
            text = element.text.strip()
            if not text:
                return []
            return [
                (
                    text,
                    element.html,
                    estimate_token_count(text) > self.table_max_tokens,
                )
            ]

        header_rows: list[_TableRow] = []
        body_rows: list[_TableRow] = []
        in_header = True
        for row in parser.rows:
            if in_header and row.is_header:
                header_rows.append(row)
            else:
                in_header = False
                body_rows.append(row)

        if not body_rows:
            text = _table_text(header_rows, [])
            return [
                (
                    text,
                    _table_html(parser.table_open_tag, header_rows, []),
                    estimate_token_count(text) > self.table_max_tokens,
                )
            ] if text else []

        parts: list[tuple[str, str, bool]] = []
        current_rows: list[_TableRow] = []
        for row in body_rows:
            candidate_rows = [*current_rows, row]
            candidate_text = _table_text(header_rows, candidate_rows)
            if current_rows and estimate_token_count(candidate_text) > self.table_max_tokens:
                current_text = _table_text(header_rows, current_rows)
                parts.append(
                    (
                        current_text,
                        _table_html(parser.table_open_tag, header_rows, current_rows),
                        estimate_token_count(current_text) > self.table_max_tokens,
                    )
                )
                current_rows = [row]
                single_text = _table_text(header_rows, current_rows)
                if estimate_token_count(single_text) > self.table_max_tokens:
                    parts.append(
                        (
                            single_text,
                            _table_html(parser.table_open_tag, header_rows, current_rows),
                            True,
                        )
                    )
                    current_rows = []
            else:
                current_rows = candidate_rows

        if current_rows:
            text = _table_text(header_rows, current_rows)
            parts.append(
                (
                    text,
                    _table_html(parser.table_open_tag, header_rows, current_rows),
                    estimate_token_count(text) > self.table_max_tokens,
                )
            )
        return parts

    def chunk_documents(
        self,
        documents: list[ParsedDocument],
        tag: str | None = None,
    ) -> list[ChunkRecord]:
        chunks: list[ChunkRecord] = []
        global_index = 0
        explicit_tag = (tag or "").strip() or None

        for document in documents:
            document_ordinal = 0
            pending: list[DocumentElement] = []
            document_tag = explicit_tag or str(document.metadata.get("tag") or "").strip()

            def emit(
                elements: list[DocumentElement],
                *,
                kind: str,
                text: str,
                html: str = "",
                table_id: str | None = None,
                oversize: bool = False,
            ) -> None:
                nonlocal global_index, document_ordinal
                if not text.strip():
                    return
                chunks.append(
                    self._build_record(
                        document,
                        elements,
                        kind=kind,
                        text=text.strip(),
                        html=html.strip(),
                        table_id=table_id,
                        oversize=oversize,
                        chunk_index=global_index,
                        document_ordinal=document_ordinal,
                        tag=document_tag,
                    )
                )
                global_index += 1
                document_ordinal += 1

            def flush_pending() -> None:
                nonlocal pending
                if not pending:
                    return
                text = "\n\n".join(element.text.strip() for element in pending if element.text.strip())
                html = "\n".join(element.html.strip() for element in pending if element.html.strip())
                if estimate_token_count(text) <= self.max_tokens:
                    emit(pending, kind="text", text=text, html=html)
                else:
                    for part in _split_text_window(
                        text,
                        window=min(self.target_tokens, self.max_tokens),
                        overlap=min(self.overlap_tokens, self.max_tokens - 1),
                    ):
                        emit(
                            pending,
                            kind="text",
                            text=part,
                            html=html,
                            oversize=estimate_token_count(part) > self.max_tokens,
                        )
                pending = []

            logical_elements = [
                item
                for root in document.elements
                for item in _iter_chunkable_elements(root)
            ]
            for element in logical_elements:
                normalized_kind = element.kind.strip().lower()
                if normalized_kind in _TABLE_KINDS:
                    flush_pending()
                    table_id = element.table_id or element.id or None
                    for table_text, table_html, oversize in self._table_parts(element):
                        emit(
                            [element],
                            kind="table",
                            text=table_text,
                            html=table_html,
                            table_id=table_id,
                            oversize=oversize,
                        )
                    continue

                if not element.text.strip():
                    continue
                if pending and not self._can_merge_text(pending[-1], element):
                    flush_pending()
                if pending:
                    candidate = "\n\n".join(
                        [*(item.text.strip() for item in pending), element.text.strip()]
                    )
                    if estimate_token_count(candidate) > self.max_tokens:
                        flush_pending()
                pending.append(element)

            flush_pending()

        return chunks

    def chunk_pages(
        self,
        pages: list[PageDocument],
        tag: str | None = None,
    ) -> list[ChunkRecord]:
        """Compatibilidad pública: agrupa páginas y aplica chunking documental."""
        return self.chunk_documents(parsed_documents_from_pages(pages), tag=tag)
