"""Carga y extracción de documentos para el índice RAG.

Este módulo convierte documentos locales en una lista homogénea de
`PageDocument`, que es la estructura que consume el chunker del proyecto.

Formatos soportados:
- Con Marker 2 full: PDF, DOCX, PPTX, XLSX, EPUB, HTML e imágenes.
- Texto plano (`.txt`): lectura UTF-8.
- Si desactivo Marker, conservo fallbacks nativos para PDF digital, DOCX y TXT.

Decisión principal:
Uso Marker 2 en modo adaptativo para conservar texto digital fiable y activar
OCR/VLM solo en páginas, bloques o tablas que lo necesiten. Pido Markdown
paginado porque conserva estructura útil para RAG y mantiene las citas.
"""
from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Callable

import fitz
from docx import Document

from rag_cliente.config import Settings

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
MARKER_DOCUMENT_SUFFIXES = {
    ".pdf",
    ".docx",
    ".pptx",
    ".xlsx",
    ".epub",
    ".html",
    *IMAGE_SUFFIXES,
}
SUPPORTED_DOCUMENT_SUFFIXES = {".txt", *MARKER_DOCUMENT_SUFFIXES}
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
    """Recojo señales internas de Marker 2 sobre OCR/visión por página.

    No me basta con `page.text_extraction_method`: en modo fast Marker puede
    reparar un bloque aislado, y en ambos modos una tabla digital de baja
    confianza puede usar el fallback visual sin convertir toda la página.
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
    visual_methods = sorted(method for method in methods if method != "pdftext")
    ocr_used = ocr_errors_detected or bool(visual_methods)

    reasons: list[str] = []
    if ocr_errors_detected:
        reasons.append("embedded_text_rejected")
    if visual_methods:
        reasons.append("visual_block_or_table_fallback")

    return {
        "text_extraction_methods": sorted(methods),
        "ocr_errors_detected": ocr_errors_detected,
        "ocr_used": ocr_used,
        "ocr_reasons": reasons,
    }


class MarkerMarkdownRendererWithOcrMetadata:
    """Enriquezco el renderer Markdown de Marker con metadatos OCR por página.

    Delego el Markdown al renderer oficial para no duplicar su lógica y, antes
    de devolverlo, registro también reparaciones visuales parciales y tablas que
    hayan caído al OCR adaptativo de Marker 2.
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
    """Construyo la configuración adaptativa de Marker 2.

    Uso `balanced` por defecto en CUDA para que Marker decida automáticamente
    entre pdftext, OCR completo y fallback visual de tablas. Mantengo
    `force_ocr` como override manual, pero no lo necesito en el flujo normal.
    """
    ocr_mode = settings.marker_ocr_mode
    config: dict[str, Any] = {
        "output_format": "markdown",
        "paginate_output": True,
        "disable_image_extraction": settings.marker_disable_image_extraction,
        "mode": settings.marker_mode,
        "disable_ocr": ocr_mode == "disabled",
        "min_recon_score": settings.marker_table_min_recon_score,
        "force_ocr_complex_layout": settings.marker_full_page_ocr_complex_layout,
    }

    if ocr_mode == "force":
        config["force_ocr"] = True

    if settings.marker_strip_existing_ocr:
        config["strip_existing_ocr"] = True

    if settings.marker_use_llm:
        config["use_llm"] = True

    if settings.marker_page_range.strip():
        config["page_range"] = settings.marker_page_range.strip()

    return config


def _require_marker_2() -> str:
    """Compruebo que el entorno ejecuta Marker 2 antes de crear sus modelos."""
    try:
        installed_version = version("marker-pdf")
    except PackageNotFoundError as exc:
        raise RuntimeError(
            "Marker no está instalado. Ejecuta setup.ps1 para instalar marker-pdf 2.x."
        ) from exc

    try:
        major_version = int(installed_version.split(".", 1)[0])
    except ValueError as exc:
        raise RuntimeError(f"No pude interpretar la versión de Marker: {installed_version}") from exc

    if major_version != 2:
        raise RuntimeError(
            f"Este proyecto requiere marker-pdf 2.x y encontré {installed_version}. "
            "Ejecuta setup.ps1 para sincronizar el entorno."
        )
    return installed_version


def _install_surya_windows_cleanup_workaround() -> None:
    """Evito que el cierre de llama.cpp de Surya interrumpa mi propio proceso en Windows.

    Surya 0.22.1 registra un callback `atexit` que usa `os.kill()` y después
    sondea el PID. En Windows ese flujo puede generar un `KeyboardInterrupt`
    en el proceso que ejecuta el indexado. Sustituyo solo esa función interna
    por `taskkill /F`, dirigido al PID exacto que Surya acaba de crear.
    """
    if os.name != "nt":
        return

    from surya.inference.backends import spawn as surya_spawn

    def stop_process_without_console_signal(pid: int, name: str) -> None:
        try:
            completed = subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                check=False,
                capture_output=True,
                timeout=15,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            if completed.returncode == 0:
                surya_spawn.logger.info(f"Stopped {name} (pid {pid})")
                return

            # Considero inocuo que el servidor ya haya terminado antes del callback.
            output = (completed.stderr or completed.stdout or b"").decode(
                errors="replace"
            ).strip()
            if output:
                surya_spawn.logger.debug(
                    f"No tuve que detener {name} (pid {pid}): {output}"
                )
        except Exception as exc:
            # No dejo que un fallo de limpieza oculte un indexado ya completado.
            surya_spawn.logger.warning(
                f"No pude cerrar {name} (pid {pid}) durante la limpieza: {exc}"
            )

    surya_spawn._stop_process = stop_process_without_console_signal


def create_marker_converter(settings: Settings) -> Any:
    """Inicializo Marker 2 una vez y reutilizo sus modelos entre documentos.

    Raises:
        RuntimeError: Si `marker-pdf` no está instalado o no puede inicializarse.
    """
    _require_marker_2()
    marker_device = settings.marker_torch_device.strip().lower()
    if marker_device:
        # Fijo TORCH_DEVICE antes de importar Marker para que yo no inicialice
        # accidentalmente los modelos grandes en CPU.
        os.environ["TORCH_DEVICE"] = marker_device

    inference_backend = settings.marker_inference_backend.strip().lower()
    if inference_backend != "auto":
        # Propago el backend antes de importar Surya para que yo controle si el
        # servidor visual se ejecuta con vLLM/Docker o con llama.cpp local.
        os.environ["SURYA_INFERENCE_BACKEND"] = inference_backend

    llama_cpp_binary = settings.marker_llama_cpp_binary.strip()
    if llama_cpp_binary:
        resolved_binary = Path(llama_cpp_binary).expanduser().resolve()
        if not resolved_binary.is_file():
            raise RuntimeError(
                f"MARKER_LLAMA_CPP_BINARY no existe o no es un archivo: {resolved_binary}"
            )
        os.environ["LLAMA_CPP_BINARY"] = str(resolved_binary)

    if marker_device == "cuda":
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "PyTorch no esta instalado. Ejecuta setup.ps1 para instalar la version CUDA."
            ) from exc

        if not torch.cuda.is_available():
            raise RuntimeError(
                "MARKER_TORCH_DEVICE=cuda, pero PyTorch no detecta la GPU. "
                "Ejecuta setup.ps1 y comprueba el resultado con 'rag.bat gpu'."
            )

    try:
        from marker.builders.document import DocumentBuilder
        from marker.builders.layout import LayoutBuilder
        from marker.builders.line import LineBuilder
        from marker.builders.ocr import OcrBuilder
        from marker.builders.structure import StructureBuilder
        from marker.config.parser import ConfigParser
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict
        from marker.providers.registry import provider_from_filepath
        from marker.schema import BlockTypes
    except ImportError as exc:
        raise RuntimeError(
            "Marker 2 no está instalado con todos los proveedores. "
            "Ejecuta setup.ps1 para instalar marker-pdf[full]."
        ) from exc

    # Instalo el ajuste antes de que Surya registre su callback de salida.
    _install_surya_windows_cleanup_workaround()

    try:
        class LayoutAwareLineBuilder(LineBuilder):
            """Promuevo a OCR completo las páginas que yo detecto como complejas."""

            force_ocr_complex_layout: bool = True
            complex_layout_types = (
                BlockTypes.Table,
                BlockTypes.Form,
                BlockTypes.ComplexRegion,
            )

            def get_all_lines(self, document: Any, provider: Any) -> dict[int, list[Any]]:
                page_lines = super().get_all_lines(document, provider)
                if self.disable_ocr or not self.force_ocr_complex_layout:
                    return page_lines

                for page in document.pages:
                    # Decido después del layout y antes del OCR: si veo una
                    # estructura compleja, descarto solo las líneas digitales de
                    # esa página para que Surya la reconstruya con contexto global.
                    blocks = page.structure_blocks(document)
                    if any(block.block_type in self.complex_layout_types for block in blocks):
                        page.text_extraction_method = "surya"
                        page_lines[page.page_id] = []
                return page_lines

        class AdaptivePdfConverter(PdfConverter):
            """Uso mi selector de OCR por layout sin alterar los procesadores de Marker."""

            def build_document(self, filepath: str) -> Any:
                provider_cls = provider_from_filepath(filepath)
                layout_builder = self.resolve_dependencies(LayoutBuilder)
                line_builder = self.resolve_dependencies(LayoutAwareLineBuilder)
                ocr_builder = self.resolve_dependencies(OcrBuilder)
                provider = provider_cls(filepath, self.config)
                document = DocumentBuilder(self.config)(
                    provider,
                    layout_builder,
                    line_builder,
                    ocr_builder,
                )
                structure_builder = self.resolve_dependencies(StructureBuilder)
                structure_builder(document)
                for processor in self.processor_list:
                    processor(document)
                return document

        config_parser = ConfigParser(_build_marker_config(settings))
        return AdaptivePdfConverter(
            config=config_parser.generate_config_dict(),
            artifact_dict=create_model_dict(
                inference_backend=None if inference_backend == "auto" else inference_backend
            ),
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
    """Divido el Markdown de Marker 2 en pares con página humana 1-based.

    Marker 2 escribe siempre el `page_id` interno 0-based en el separador. Sumo
    uno incluso cuando yo proceso un rango que empieza después de la página 0;
    así evito que las citas queden desplazadas al usar `MARKER_PAGE_RANGE`.
    """
    normalized_markdown = markdown.strip()
    if not normalized_markdown:
        return []

    matches = list(_MARKER_PAGE_SEPARATOR_RE.finditer(normalized_markdown))
    if not matches:
        return [(1, _normalize_text(normalized_markdown))]

    raw_page_numbers = [int(match.group(1)) for match in matches]
    pages: list[tuple[int, str]] = []

    preamble = normalized_markdown[: matches[0].start()].strip()
    if preamble:
        first_page = raw_page_numbers[0] + 1
        pages.append((max(first_page, 1), _normalize_text(preamble)))

    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(normalized_markdown)
        text = _normalize_text(normalized_markdown[start:end])
        if not text:
            continue
        page_number = int(match.group(1)) + 1
        pages.append((max(page_number, 1), text))

    return pages


def _marker_document_pages(
    path: Path,
    source_type: str,
    marker_converter: Any,
    default_ocr_used: bool = False,
) -> list[PageDocument]:
    """Adapto cualquier salida paginada de Marker 2 a `PageDocument`."""
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


def load_marker_document_pages(
    document_path: Path,
    settings: Settings,
    marker_converter: Any | None = None,
) -> list[PageDocument]:
    """Proceso con Marker 2 cualquier formato admitido por su paquete full.

    Dejo que el modo de Marker decida la estrategia para cada página y bloque.
    Solo marco OCR por defecto en imágenes puras o cuando yo lo fuerzo de forma
    explícita; para el resto uso los metadatos reales del renderer enriquecido.
    """
    converter = marker_converter or create_marker_converter(settings)
    suffix = document_path.suffix.lower()
    return _marker_document_pages(
        document_path,
        source_type=suffix.lstrip("."),
        marker_converter=converter,
        default_ocr_used=suffix in IMAGE_SUFFIXES or settings.marker_ocr_mode == "force",
    )


def load_pdf_pages(
    pdf_path: Path,
    settings: Settings | None = None,
    ocr_pipeline: Any | None = None,
    marker_converter: Any | None = None,
) -> list[PageDocument]:
    """Cargo un PDF con Marker 2 o con mi fallback digital si lo desactivo.

    `ocr_pipeline` se mantiene por compatibilidad con llamadas antiguas del
    backend; si se recibe, se trata como un converter de Marker ya inicializado.
    """
    if settings is None or not settings.marker_enabled:
        return _load_native_pdf_pages(pdf_path)

    return load_marker_document_pages(
        pdf_path,
        settings=settings,
        marker_converter=marker_converter or ocr_pipeline,
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
    """Cargo una imagen mediante el OCR adaptativo de Marker 2.

    Mantiene compatibilidad funcional con el soporte previo de imágenes sin
    depender de PaddleOCR/PaddleX.
    """
    if settings is None or not settings.marker_enabled:
        return []

    return load_marker_document_pages(
        image_path,
        settings=settings,
        marker_converter=marker_converter or ocr_pipeline,
    )


def load_documents_from_directory(
    doc_dir: Path,
    settings: Settings | None = None,
    progress_callback: ProgressCallback | None = None,
) -> list[PageDocument]:
    """Cargo recursivamente todos los formatos soportados por Marker 2 full.

    Si un archivo falla, continúo con los demás y dejo el motivo visible en el
    progreso para que yo pueda diagnosticarlo sin perder toda la indexación.
    """
    if not doc_dir.exists():
        raise FileNotFoundError(f"Document directory does not exist: {doc_dir}")

    base_dir = doc_dir.resolve()
    paths = [path for path in sorted(doc_dir.rglob("*")) if path.is_file()]
    supported_paths = [
        path
        for path in paths
        if path.suffix.lower() in SUPPORTED_DOCUMENT_SUFFIXES
    ]

    marker_converter: Any | None = None
    if settings is not None and settings.marker_enabled:
        needs_marker = any(path.suffix.lower() in MARKER_DOCUMENT_SUFFIXES for path in supported_paths)
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
            if suffix == ".txt":
                pages = load_txt_pages(path)
            elif settings is not None and settings.marker_enabled and suffix in MARKER_DOCUMENT_SUFFIXES:
                _emit(
                    progress_callback,
                    f"Parseando {suffix.lstrip('.').upper()} con Marker 2 "
                    f"({settings.marker_mode}): {display_path}",
                )
                pages = load_marker_document_pages(
                    path,
                    settings=settings,
                    marker_converter=marker_converter,
                )
            elif suffix == ".pdf":
                pages = _load_native_pdf_pages(path)
            elif suffix == ".docx":
                pages = load_docx_pages(path)
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
