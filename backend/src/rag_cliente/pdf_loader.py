"""Carga y extracción de documentos para el índice RAG.

Este módulo convierte documentos locales en una lista homogénea de
`PageDocument`, que es la estructura que consume el chunker del proyecto.

Formatos soportados:
- PDF (`.pdf`): parser principal con Marker. Si se desactiva Marker, queda un
  fallback nativo con PyMuPDF para PDFs digitales.
- Word (`.docx`): párrafos y tablas con python-docx.
- Texto plano (`.txt`): lectura UTF-8.
- Imágenes (`.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif`, `.tiff`, `.webp`): parser
  con Marker cuando está habilitado.

Decisión principal para PDFs:
Marker se usa como parser/OCR principal y se configura para devolver Markdown
paginado. Markdown conserva encabezados, listas, tablas y fórmulas en una forma
muy útil para embeddings y RAG sin obligar a reescribir el chunker actual. El
paginado permite seguir generando citas por página/unidad lógica.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import fitz
from docx import Document

from rag_cliente.config import Settings

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
ProgressCallback = Callable[[str], None]
_MARKER_PAGE_SEPARATOR_RE = re.compile(
    r"(?:^|\n+)(?:\{)?(\d+)(?:\})?-{20,}[ \t]*(?:\n+|$)"
)


@dataclass(slots=True)
class PageDocument:
    """Unidad de contenido ya extraída y lista para chunking.

    Atributos:
        document_id: Identificador lógico del documento.
        source: Nombre visible del archivo original.
        source_path: Ruta absoluta al archivo original.
        source_type: Tipo de documento: pdf, docx, txt o extensión de imagen.
        page_number: Página/unidad lógica asociada al texto.
        text: Contenido textual normalizado.
        ocr_used: Indica si el texto de esta página/unidad procede de OCR.
        tag: Etiqueta derivada de la primera carpeta relativa bajo el directorio indexado.
    """

    document_id: str
    source: str
    source_path: str
    source_type: str
    page_number: int
    text: str
    ocr_used: bool = False
    tag: str = ""


def _emit(progress_callback: ProgressCallback | None, message: str) -> None:
    if progress_callback is not None:
        progress_callback(message)


def _normalize_text(text: str) -> str:
    """Normaliza bloques de texto preservando separación por párrafos."""
    return "\n\n".join(block.strip() for block in text.split("\n\n") if block.strip()).strip()


def _extract_docx_text(document: Document) -> str:
    """Extrae párrafos y tablas de un documento Word (.docx)."""
    blocks: list[str] = []

    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        if text:
            blocks.append(text)

    for table in document.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if cells:
                blocks.append(" | ".join(cells))

    return _normalize_text("\n\n".join(blocks))


def _as_1_based_page_number(raw_page_id: Any) -> int | None:
    """Convierte IDs internos de Marker a numeración humana 1-based."""
    try:
        page_id = int(raw_page_id)
    except (TypeError, ValueError):
        return None
    return page_id + 1 if page_id >= 0 else None


def _collect_marker_page_extraction_details(page: Any, document: Any) -> dict[str, Any]:
    """Recoge señales internas de Marker sobre OCR/visión en una página.

    `page_stats.text_extraction_method` solo describe el método dominante de la
    página. Para no perder OCR parcial, también inspeccionamos bloques internos
    renderizados por Marker y `ocr_errors_detected`.
    """
    methods: set[str] = set()

    def add_method(value: Any) -> None:
        method = str(value or "").strip().lower()
        if method:
            methods.add(method)

    add_method(getattr(page, "text_extraction_method", None))

    try:
        blocks = list(page.contained_blocks(document))
    except Exception:
        blocks = []

    for block in blocks:
        add_method(getattr(block, "text_extraction_method", None))

    ocr_errors_detected = bool(getattr(page, "ocr_errors_detected", False))
    ocr_used = ocr_errors_detected or any(method != "pdftext" for method in methods)

    return {
        "text_extraction_methods": sorted(methods),
        "ocr_errors_detected": ocr_errors_detected,
        "ocr_used": ocr_used,
    }


class MarkerMarkdownRendererWithOcrMetadata:
    """Renderer Markdown de Marker enriquecido con metadatos OCR por página.

    Marker ya genera `page_stats`, pero esa salida puede quedarse corta para
    saber si hubo OCR parcial. Este wrapper delega el Markdown real en el
    renderer oficial y añade campos derivados inspeccionando el documento
    interno justo antes de devolver el resultado.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        from marker.renderers.markdown import MarkdownRenderer

        self._renderer = MarkdownRenderer(config=config)

    def __call__(self, document: Any) -> Any:
        rendered = self._renderer(document)
        metadata = dict(getattr(rendered, "metadata", {}) or {})
        page_stats_by_id: dict[int, dict[str, Any]] = {}

        for page_stat in metadata.get("page_stats", []) or []:
            if not isinstance(page_stat, dict):
                continue
            try:
                page_stats_by_id[int(page_stat.get("page_id"))] = dict(page_stat)
            except (TypeError, ValueError):
                continue

        enriched_page_stats: list[dict[str, Any]] = []
        for page in getattr(document, "pages", []) or []:
            raw_page_id = getattr(page, "page_id", None)
            try:
                page_id = int(raw_page_id)
            except (TypeError, ValueError):
                continue

            page_stat = page_stats_by_id.get(page_id, {"page_id": page_id})
            extraction_details = _collect_marker_page_extraction_details(page, document)
            page_stat.update(extraction_details)
            enriched_page_stats.append(page_stat)

        if enriched_page_stats:
            metadata["page_stats"] = enriched_page_stats

        if hasattr(rendered, "model_copy"):
            return rendered.model_copy(update={"metadata": metadata})

        rendered.metadata = metadata
        return rendered


def _load_native_pdf_pages(pdf_path: Path) -> list[PageDocument]:
    """Fallback rápido para PDFs digitales cuando Marker está deshabilitado."""
    pages: list[PageDocument] = []
    with fitz.open(pdf_path) as document:
        for page_index, page in enumerate(document):
            text = page.get_text("text").strip()
            if not text:
                continue
            pages.append(
                PageDocument(
                    document_id=pdf_path.stem,
                    source=pdf_path.name,
                    source_path=str(pdf_path.resolve()),
                    source_type="pdf",
                    page_number=page_index + 1,
                    text=_normalize_text(text),
                    ocr_used=False,
                )
            )
    return pages


def _build_marker_config(settings: Settings) -> dict[str, Any]:
    """Construye la configuración mínima para Marker.

    Se pide Markdown paginado porque conserva estructura útil para RAG y permite
    reconstruir `PageDocument` por página sin cambiar el resto del pipeline.
    """
    config: dict[str, Any] = {
        "output_format": "markdown",
        "paginate_output": True,
        "disable_image_extraction": settings.marker_disable_image_extraction,
    }

    if settings.marker_force_ocr:
        config["force_ocr"] = True

    if settings.marker_strip_existing_ocr:
        config["strip_existing_ocr"] = True

    if settings.marker_use_llm:
        config["use_llm"] = True

    if settings.marker_page_range.strip():
        config["page_range"] = settings.marker_page_range.strip()

    return config


def create_marker_converter(settings: Settings) -> Any:
    """Inicializa Marker una sola vez para reutilizar modelos entre archivos.

    Raises:
        RuntimeError: Si `marker-pdf` no está instalado o no puede inicializarse.
    """
    if settings.marker_torch_device.strip():
        # Marker documenta TORCH_DEVICE como mecanismo para forzar cpu/cuda/mps.
        # Se fija antes de importar Marker para que torch lo vea al inicializar.
        os.environ.setdefault("TORCH_DEVICE", settings.marker_torch_device.strip())

    try:
        from marker.config.parser import ConfigParser
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict
    except ImportError as exc:
        raise RuntimeError(
            "Marker no está instalado. Instala la dependencia con: "
            "python -m pip install marker-pdf"
        ) from exc

    try:
        config_parser = ConfigParser(_build_marker_config(settings))
        return PdfConverter(
            config=config_parser.generate_config_dict(),
            artifact_dict=create_model_dict(),
            processor_list=config_parser.get_processors(),
            renderer="rag_cliente.pdf_loader.MarkerMarkdownRendererWithOcrMetadata",
            llm_service=config_parser.get_llm_service(),
        )
    except Exception as exc:
        raise RuntimeError(f"No se pudo inicializar Marker: {exc}") from exc


def _render_with_marker(path: Path, marker_converter: Any) -> tuple[str, dict[str, Any]]:
    """Convierte un PDF/imagen a Markdown y devuelve metadatos de Marker.

    Marker incluye `metadata["page_stats"]` con `text_extraction_method` por
    página. Ese dato se usa para saber si una página acabó usando OCR (`surya`)
    o texto embebido del PDF (`pdftext`).
    """
    try:
        from marker.output import text_from_rendered
    except ImportError as exc:
        raise RuntimeError(
            "Marker está instalado parcialmente o cambió su API: no se pudo importar "
            "marker.output.text_from_rendered. Reinstala/actualiza marker-pdf."
        ) from exc

    rendered = marker_converter(str(path))
    text, _, _images = text_from_rendered(rendered)
    metadata = getattr(rendered, "metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    if not isinstance(text, str):
        return "", metadata
    return text, metadata


def _extract_marker_ocr_usage_by_page(metadata: dict[str, Any]) -> dict[int, bool]:
    """Extrae uso de OCR por página a partir de metadatos de Marker.

    Prioriza el campo enriquecido `ocr_used`. Si no existe, cae a
    `text_extraction_methods` o `text_extraction_method`. `pdftext` significa
    texto nativo del PDF; `surya`/`gemini`/otros valores implican OCR o visión.
    """
    page_stats = metadata.get("page_stats")
    if not isinstance(page_stats, list):
        return {}

    ocr_usage_by_page: dict[int, bool] = {}
    for page_stat in page_stats:
        if not isinstance(page_stat, dict):
            continue

        page_number = _as_1_based_page_number(page_stat.get("page_id"))
        if page_number is None:
            continue

        if isinstance(page_stat.get("ocr_used"), bool):
            ocr_usage_by_page[page_number] = page_stat["ocr_used"]
            continue

        methods: list[str] = []
        raw_methods = page_stat.get("text_extraction_methods")
        if isinstance(raw_methods, list):
            methods.extend(str(method).strip().lower() for method in raw_methods if str(method).strip())

        method = str(page_stat.get("text_extraction_method", "")).strip().lower()
        if method:
            methods.append(method)

        if methods:
            ocr_usage_by_page[page_number] = any(method != "pdftext" for method in methods)

    return ocr_usage_by_page


def _split_marker_markdown_by_page(markdown: str) -> list[tuple[int, str]]:
    """Divide el Markdown paginado de Marker en pares (page_number, text).

    Marker pagina con una línea de número de página y un separador de guiones.
    Si la salida no incluye separadores, se devuelve una única unidad lógica.
    """
    normalized_markdown = markdown.strip()
    if not normalized_markdown:
        return []

    matches = list(_MARKER_PAGE_SEPARATOR_RE.finditer(normalized_markdown))
    if not matches:
        return [(1, _normalize_text(normalized_markdown))]

    raw_page_numbers = [int(match.group(1)) for match in matches]
    page_offset = 1 if min(raw_page_numbers) == 0 else 0
    pages: list[tuple[int, str]] = []

    preamble = normalized_markdown[: matches[0].start()].strip()
    if preamble:
        first_page = raw_page_numbers[0] + page_offset
        pages.append((max(first_page, 1), _normalize_text(preamble)))

    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(normalized_markdown)
        text = _normalize_text(normalized_markdown[start:end])
        if not text:
            continue
        page_number = int(match.group(1)) + page_offset
        pages.append((max(page_number, 1), text))

    return pages


def _marker_document_pages(
    path: Path,
    source_type: str,
    marker_converter: Any,
    default_ocr_used: bool = False,
) -> list[PageDocument]:
    """Procesa un PDF o imagen con Marker y lo adapta a PageDocument."""
    try:
        markdown, metadata = _render_with_marker(path, marker_converter)
    except Exception as exc:
        raise RuntimeError(f"Marker falló procesando '{path.name}': {exc}") from exc

    page_texts = _split_marker_markdown_by_page(markdown)
    marker_ocr_usage = _extract_marker_ocr_usage_by_page(metadata)
    document_ocr_used = any(marker_ocr_usage.values()) if marker_ocr_usage else default_ocr_used

    pages: list[PageDocument] = []
    for page_number, text in page_texts:
        if not text:
            continue
        ocr_used = marker_ocr_usage.get(
            page_number,
            document_ocr_used if len(page_texts) == 1 else default_ocr_used,
        )
        pages.append(
            PageDocument(
                document_id=path.stem,
                source=path.name,
                source_path=str(path.resolve()),
                source_type=source_type,
                page_number=page_number,
                text=text,
                ocr_used=ocr_used,
            )
        )
    return pages


def load_pdf_pages(
    pdf_path: Path,
    settings: Settings | None = None,
    ocr_pipeline: Any | None = None,
    marker_converter: Any | None = None,
) -> list[PageDocument]:
    """Carga un PDF usando Marker como parser principal.

    `ocr_pipeline` se mantiene por compatibilidad con llamadas antiguas del
    backend; si se recibe, se trata como un converter de Marker ya inicializado.
    """
    if settings is None or not settings.marker_enabled:
        return _load_native_pdf_pages(pdf_path)

    converter = marker_converter or ocr_pipeline
    if converter is None:
        converter = create_marker_converter(settings)

    return _marker_document_pages(
        pdf_path,
        source_type="pdf",
        marker_converter=converter,
        default_ocr_used=settings.marker_force_ocr,
    )


def load_docx_pages(docx_path: Path) -> list[PageDocument]:
    """Carga y extrae texto de un documento Word (.docx)."""
    document = Document(docx_path)
    text = _extract_docx_text(document)
    if not text:
        return []

    return [
        PageDocument(
            document_id=docx_path.stem,
            source=docx_path.name,
            source_path=str(docx_path.resolve()),
            source_type="docx",
            page_number=1,
            text=text,
            ocr_used=False,
        )
    ]


def load_txt_pages(txt_path: Path) -> list[PageDocument]:
    """Carga y extrae texto de un archivo de texto plano UTF-8."""
    text = txt_path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    return [
        PageDocument(
            document_id=txt_path.stem,
            source=txt_path.name,
            source_path=str(txt_path.resolve()),
            source_type="txt",
            page_number=1,
            text=_normalize_text(text),
            ocr_used=False,
        )
    ]


def load_image_pages(
    image_path: Path,
    settings: Settings | None = None,
    ocr_pipeline: Any | None = None,
    marker_converter: Any | None = None,
) -> list[PageDocument]:
    """Carga una imagen mediante Marker cuando está habilitado.

    Mantiene compatibilidad funcional con el soporte previo de imágenes sin
    depender de PaddleOCR/PaddleX.
    """
    if settings is None or not settings.marker_enabled:
        return []

    converter = marker_converter or ocr_pipeline
    if converter is None:
        converter = create_marker_converter(settings)

    return _marker_document_pages(
        image_path,
        source_type=image_path.suffix.lower().lstrip("."),
        marker_converter=converter,
        default_ocr_used=True,
    )


def load_documents_from_directory(
    doc_dir: Path,
    settings: Settings | None = None,
    progress_callback: ProgressCallback | None = None,
) -> list[PageDocument]:
    """Carga todos los documentos soportados de un directorio de forma recursiva.

    Si un archivo falla, se emite un aviso y se continúa con el resto. No se
    tragan excepciones silenciosamente: el motivo queda visible en el progreso.
    """
    if not doc_dir.exists():
        raise FileNotFoundError(f"Document directory does not exist: {doc_dir}")

    base_dir = doc_dir.resolve()
    paths = [path for path in sorted(doc_dir.rglob("*")) if path.is_file()]
    supported_paths = [
        path
        for path in paths
        if path.suffix.lower() in {".pdf", ".docx", ".txt", *IMAGE_SUFFIXES}
    ]

    marker_converter: Any | None = None
    if settings is not None and settings.marker_enabled:
        needs_marker = any(
            path.suffix.lower() == ".pdf" or path.suffix.lower() in IMAGE_SUFFIXES
            for path in supported_paths
        )
        if needs_marker:
            _emit(progress_callback, "Inicializando Marker...")
            marker_converter = create_marker_converter(settings)

    all_pages: list[PageDocument] = []
    total_files = len(supported_paths)

    for file_index, path in enumerate(supported_paths, start=1):
        suffix = path.suffix.lower()
        try:
            relative_path = path.resolve().relative_to(base_dir)
        except ValueError:
            relative_path = Path(path.name)
        tag = relative_path.parts[0] if len(relative_path.parts) > 1 else ""
        display_path = str(relative_path)
        _emit(progress_callback, f"Procesando archivo {file_index}/{total_files}: {display_path}")

        try:
            if suffix == ".pdf":
                if settings is not None and settings.marker_enabled:
                    _emit(progress_callback, f"Parseando PDF con Marker: {display_path}")
                pages = load_pdf_pages(
                    path,
                    settings=settings,
                    marker_converter=marker_converter,
                )
            elif suffix == ".docx":
                pages = load_docx_pages(path)
            elif suffix == ".txt":
                pages = load_txt_pages(path)
            elif suffix in IMAGE_SUFFIXES:
                if settings is not None and settings.marker_enabled:
                    _emit(progress_callback, f"Parseando imagen con Marker: {display_path}")
                pages = load_image_pages(
                    path,
                    settings=settings,
                    marker_converter=marker_converter,
                )
            else:
                continue
        except Exception as exc:
            _emit(progress_callback, f"AVISO: no se pudo procesar {display_path}: {exc}")
            continue

        if pages:
            for page in pages:
                page.tag = tag
            all_pages.extend(pages)
            ocr_pages = sum(1 for page in pages if page.ocr_used)
            tag_label = f", tag: {tag}" if tag else ""
            _emit(
                progress_callback,
                f"Extraídas {len(pages)} páginas/bloques de {display_path} "
                f"(OCR usado en {ocr_pages}{tag_label})",
            )
        else:
            _emit(progress_callback, f"AVISO: {display_path} no produjo texto indexable")

    return all_pages
