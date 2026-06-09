"""Chunking de documentos para indexación vectorial.

Este módulo toma `PageDocument` ya cargados y los transforma en `ChunkRecord`,
que es la estructura que luego se embebe y se guarda en la base vectorial.

Decisiones principales:
- El chunking se hace por página lógica, no cruzando páginas.
- El texto normal se divide con `RecursiveCharacterTextSplitter` de LangChain.
- Las tablas Markdown se separan del texto normal y se trocean por filas
  completas, repitiendo la cabecera en cada fragmento de tabla.
- Las tablas Markdown se compactan antes de medir tamaño para evitar que el
  padding visual de Marker genere chunks artificialmente pequeños.
- Las filas vacías reales de una tabla se consumen como parte de la tabla, pero
  no generan chunks.
- Se conservan metadatos de trazabilidad: archivo, página, índice de chunk y uso de OCR.

Limitaciones:
- El nombre `PdfChunker` es más estrecho de lo que realmente hace, porque también
  procesa DOCX y TXT.
- Al trocear página por página se simplifican las citas, pero se puede romper el
  contexto cuando una idea continúa entre dos páginas.
- `chunk_index` es global dentro de una ejecución de chunking; no se reinicia por
  documento.
- Si una fila de tabla es más grande que `CHUNK_SIZE`, se conserva completa en un
  único chunk aunque supere ese tamaño, para no partir datos de una misma fila.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from uuid import uuid4

from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_cliente.config import Settings
from rag_cliente.pdf_loader import PageDocument

_TABLE_SEPARATOR_CELL_RE = re.compile(r"^:?-{3,}:?$")


@dataclass(slots=True)
class ChunkRecord:
    """Representa un chunk listo para ser indexado en el vector store."""

    id: str
    document_id: str
    text: str
    source: str
    source_path: str
    source_type: str
    page_start: int
    page_end: int
    chunk_index: int
    ocr_used: bool
    tag: str = ""


def _split_markdown_cells(line: str) -> list[str]:
    """Divide una fila Markdown en celdas, con o sin pipes externos."""
    stripped = line.strip()
    if "|" not in stripped:
        return []
    return [cell.strip() for cell in stripped.strip("|").split("|")]


def _is_markdown_table_separator(line: str) -> bool:
    """Detecta la línea separadora típica de una tabla Markdown."""
    cells = _split_markdown_cells(line)
    return len(cells) >= 2 and all(_TABLE_SEPARATOR_CELL_RE.match(cell) for cell in cells)


def _is_empty_markdown_table_row(line: str) -> bool:
    """Detecta filas de tabla Markdown completamente vacías.

    Ejemplo:

    `|     |     |     |`

    Estas filas pueden existir visualmente en el PDF, pero no aportan contenido
    semántico al RAG. Las consumimos dentro de la tabla para que no se conviertan
    en chunks basura.
    """
    cells = _split_markdown_cells(line)
    return len(cells) >= 2 and all(not cell.strip() for cell in cells)


def _is_markdown_table_row(line: str) -> bool:
    """Detecta filas Markdown con dos o más columnas y algo de contenido."""
    cells = _split_markdown_cells(line)
    return len(cells) >= 2 and any(cell.strip() for cell in cells)


def _is_markdown_table_continuation_line(line: str) -> bool:
    """Detecta líneas que deben seguir dentro de una tabla ya iniciada."""
    return _is_markdown_table_row(line) or _is_empty_markdown_table_row(line)


def _normalize_table_cell(cell: str) -> str:
    """Compacta una celda de tabla Markdown sin destruir contenido útil.

    Marker puede devolver tablas con padding visual para alinear columnas.
    Ese padding infla artificialmente el tamaño de los chunks. Compactamos
    espacios y tabuladores, pero preservamos contenido como `<br>`.
    """
    return re.sub(r"[ \t]+", " ", cell.strip())


def _build_compact_markdown_row(cells: list[str]) -> str:
    """Reconstruye una fila Markdown sin padding visual innecesario."""
    normalized_cells = [_normalize_table_cell(cell) for cell in cells]
    return "| " + " | ".join(normalized_cells) + " |"


def _build_compact_separator(column_count: int) -> str:
    """Construye un separador Markdown mínimo para N columnas."""
    return "| " + " | ".join("---" for _ in range(column_count)) + " |"


def _compact_markdown_table(table: str) -> str:
    """Elimina padding visual y filas vacías de tablas Markdown generadas por Marker.

    Marker puede devolver líneas con cientos de espacios para alinear columnas:

    `| Descripción                                                                                                                                 |`

    Si medimos esa tabla con `len()`, el chunker cree que ocupa mucho más de lo
    que realmente ocupa semánticamente.

    También puede devolver filas completamente vacías:

    `|     |     |     |`

    Esas filas no aportan contenido al índice, así que se descartan.
    """
    compact_lines: list[str] = []
    table_lines = [line for line in table.splitlines() if line.strip()]

    for line in table_lines:
        cells = _split_markdown_cells(line)
        if not cells:
            continue

        if _is_empty_markdown_table_row(line):
            continue

        if _is_markdown_table_separator(line):
            if compact_lines:
                compact_lines.append(_build_compact_separator(len(cells)))
            continue

        compact_lines.append(_build_compact_markdown_row(cells))

    return "\n".join(compact_lines).strip()


def _is_empty_table_noise(text: str) -> bool:
    """Detecta chunks formados solo por filas vacías de tabla."""
    lines = [line for line in text.splitlines() if line.strip()]
    return bool(lines) and all(_is_empty_markdown_table_row(line) for line in lines)


def _looks_like_markdown_table_start(lines: list[str], index: int) -> bool:
    """Comprueba si `lines[index]` inicia una tabla Markdown."""
    if index + 1 >= len(lines):
        return False
    return _is_markdown_table_row(lines[index]) and _is_markdown_table_separator(lines[index + 1])


def _split_text_and_markdown_tables(text: str) -> list[tuple[str, str]]:
    """Separa texto normal y tablas Markdown preservando el orden original.

    Devuelve una lista de pares `(kind, content)`, donde `kind` es `"text"` o
    `"table"`.

    Solo separa tablas Markdown bien formadas con cabecera y línea separadora.
    Las filas vacías que aparezcan dentro de la tabla se consumen como parte de
    esa tabla para evitar que salgan como chunks independientes.
    """
    lines = text.splitlines()
    segments: list[tuple[str, str]] = []
    text_buffer: list[str] = []
    index = 0

    def flush_text_buffer() -> None:
        nonlocal text_buffer
        content = "\n".join(text_buffer).strip()
        if content and not _is_empty_table_noise(content):
            segments.append(("text", content))
        text_buffer = []

    while index < len(lines):
        if _looks_like_markdown_table_start(lines, index):
            flush_text_buffer()

            table_lines = [lines[index].rstrip(), lines[index + 1].rstrip()]
            index += 2

            while index < len(lines):
                line = lines[index]

                if _is_markdown_table_continuation_line(line):
                    table_lines.append(line.rstrip())
                    index += 1
                    continue

                break

            compact_table = _compact_markdown_table("\n".join(table_lines))
            if compact_table:
                segments.append(("table", compact_table))
            continue

        text_buffer.append(lines[index].rstrip())
        index += 1

    flush_text_buffer()
    return segments


def _split_markdown_table_rows(table: str) -> tuple[str, str, list[str]] | None:
    """Extrae cabecera, separador y filas de una tabla Markdown."""
    lines = [line.rstrip() for line in table.splitlines() if line.strip()]
    if len(lines) < 2:
        return None

    header = lines[0]
    separator = lines[1]

    if not _is_markdown_table_row(header) or not _is_markdown_table_separator(separator):
        return None

    rows = [
        line
        for line in lines[2:]
        if _is_markdown_table_row(line) and not _is_empty_markdown_table_row(line)
    ]

    return header, separator, rows


class PdfChunker:
    """Divide texto cargado desde documentos en chunks pequeños y solapados.

    Aunque el nombre sugiere PDF, esta clase no procesa solo PDFs:
    recibe una lista de `PageDocument`, por lo que también puede trocear
    contenidos procedentes de DOCX y TXT.

    Estrategia:
    - Detecta tablas Markdown antes de aplicar el splitter genérico.
    - Compacta tablas Markdown para eliminar padding visual.
    - Descarta filas vacías de tablas.
    - En tablas, genera chunks independientes por filas completas.
    - Repite cabecera y separador de tabla en cada chunk.
    - En texto normal, usa el splitter textual existente.
    """

    def __init__(self, settings: Settings) -> None:
        """Inicializa el splitter con la configuración del proyecto."""
        self.chunk_size = settings.chunk_size
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def _split_regular_text(self, text: str) -> list[str]:
        """Aplica el splitter existente al texto que no es tabla."""
        chunks: list[str] = []

        for chunk in self.splitter.split_text(text):
            normalized = chunk.strip()
            if not normalized:
                continue

            if _is_empty_table_noise(normalized):
                continue

            chunks.append(normalized)

        return chunks

    def _split_markdown_table(self, table: str) -> list[str]:
        """Trocea una tabla Markdown sin partir filas.

        Cada chunk de tabla conserva la cabecera y la línea separadora, aunque
        contenga solo una parte de las filas. Así una fila recuperada por RAG no
        pierde el significado de sus columnas.
        """
        compact_table = _compact_markdown_table(table)

        parsed_table = _split_markdown_table_rows(compact_table)
        if parsed_table is None:
            return self._split_regular_text(compact_table)

        header, separator, rows = parsed_table
        table_prefix = [header, separator]

        if not rows:
            return []

        chunks: list[str] = []
        current_rows: list[str] = []

        def build_table_chunk(rows_to_include: list[str]) -> str:
            return "\n".join([*table_prefix, *rows_to_include]).strip()

        for row in rows:
            if not current_rows:
                current_rows.append(row)
                continue

            candidate = build_table_chunk([*current_rows, row])

            if len(candidate) <= self.chunk_size:
                current_rows.append(row)
                continue

            chunks.append(build_table_chunk(current_rows))
            current_rows = [row]

        if current_rows:
            chunks.append(build_table_chunk(current_rows))

        return chunks

    def _split_page_text(self, text: str) -> list[str]:
        """Divide una página alternando texto normal y tablas Markdown."""
        chunks: list[str] = []

        for kind, content in _split_text_and_markdown_tables(text):
            if kind == "table":
                chunks.extend(self._split_markdown_table(content))
            else:
                chunks.extend(self._split_regular_text(content))

        return chunks

    def _build_chunk_record(
        self,
        page: PageDocument,
        text: str,
        chunk_index: int,
        tag: str = "",
    ) -> ChunkRecord:
        """Crea un `ChunkRecord` preservando metadatos de origen."""
        return ChunkRecord(
            id=str(uuid4()),
            document_id=page.document_id,
            text=text,
            source=page.source,
            source_path=page.source_path,
            source_type=page.source_type,
            page_start=page.page_number,
            page_end=page.page_number,
            chunk_index=chunk_index,
            ocr_used=page.ocr_used,
            tag=tag,
        )

    def chunk_pages(self, pages: list[PageDocument], tag: str | None = None) -> list[ChunkRecord]:
        """Convierte páginas/unidades cargadas en una lista de `ChunkRecord`.

        Flujo:
        1. Recorre cada `PageDocument`.
        2. Separa tablas Markdown del texto normal.
        3. Divide texto normal con el splitter configurado.
        4. Divide tablas por filas completas, repitiendo cabecera.
        5. Descarta filas vacías de tabla.
        6. Descarta chunks vacíos.
        7. Crea un `ChunkRecord` por cada fragmento válido.
        """
        chunks: list[ChunkRecord] = []
        chunk_index = 0
        explicit_tag = (tag or "").strip() or None

        for page in pages:
            split_texts = self._split_page_text(page.text)

            for text in split_texts:
                normalized = text.strip()
                if not normalized:
                    continue

                if _is_empty_table_noise(normalized):
                    continue

                chunk_tag = explicit_tag if explicit_tag is not None else (page.tag or "").strip()
                chunks.append(self._build_chunk_record(page, normalized, chunk_index, tag=chunk_tag))
                chunk_index += 1

        return chunks
