"""Carga y extracción de documentos para el índice RAG.

Este módulo convierte documentos locales en una lista homogénea de
`PageDocument`, que es la estructura que consume el chunker del proyecto.

Formatos soportados:
- Con Marker 2 full: PDF, DOCX, PPTX, XLSX, EPUB, HTML e imágenes.
- Texto plano (`.txt`): lectura UTF-8.
- Si desactivo Marker, conservo fallbacks nativos para PDF digital, DOCX y TXT.

Decisión principal:
Uso exclusivamente el converter y los builders oficiales de Marker 2. La
salida primaria es JSON estructurado; el renderer Markdown anterior queda
disponible solo mediante un flag temporal de compatibilidad.
"""
from __future__ import annotations

import html as html_lib
import os
import re
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Callable

import fitz
from docx import Document

from rag_cliente.config import ResolvedMarkerProfile, Settings, resolve_marker_profile

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
        block_type/id/html/page/polygon/children/section_hierarchy: Estructura
            oficial conservada desde la salida JSON de Marker.
        extraction_metadata: Metadatos de extracción del documento y la página.
    """

    document_id: str
    source: str
    source_path: str
    source_type: str
    page_number: int
    text: str
    ocr_used: bool = False
    tag: str = ""
    block_type: str = ""
    id: str = ""
    html: str = ""
    page: int | None = None
    polygon: list[list[float]] = field(default_factory=list)
    children: list[dict[str, Any]] = field(default_factory=list)
    section_hierarchy: dict[str, Any] = field(default_factory=dict)
    extraction_metadata: dict[str, Any] = field(default_factory=dict)


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


class MarkerJSONRendererWithOcrMetadata:
    """Delega la estructura al renderer JSON público y añade metadatos OCR."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        from marker.renderers.json import JSONRenderer

        self._renderer = JSONRenderer(config=config)

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
            try:
                page_id = int(getattr(page, "page_id", None))
            except (TypeError, ValueError):
                continue
            page_stat = page_stats_by_id.get(page_id, {"page_id": page_id})
            page_stat.update(_collect_marker_page_extraction_details(page, document))
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


def _build_marker_config(
    settings: Settings,
    profile: ResolvedMarkerProfile | None = None,
) -> dict[str, Any]:
    """Construye configuración exacta para uno de los perfiles soportados."""
    resolved_profile = profile or resolve_marker_profile(settings)
    config: dict[str, Any] = {
        "output_format": "markdown" if settings.marker_markdown_compatibility else "json",
        "paginate_output": settings.marker_markdown_compatibility,
        "disable_image_extraction": settings.marker_disable_image_extraction,
        "mode": resolved_profile.mode,
        "disable_ocr": resolved_profile.disable_ocr,
        "use_llm": resolved_profile.use_llm,
        "min_recon_score": settings.marker_table_min_recon_score,
    }

    if settings.marker_strip_existing_ocr:
        config["strip_existing_ocr"] = True

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


def create_marker_converter(settings: Settings) -> Any:
    """Inicializo Marker 2 una vez y reutilizo sus modelos entre documentos.

    Raises:
        RuntimeError: Si `marker-pdf` no está instalado o no puede inicializarse.
    """
    profile = resolve_marker_profile(settings)

    if profile.use_llm:
        # Se valida antes de crear Marker: ConfigParser nunca puede seleccionar
        # Gemini ni otro proveedor externo como servicio implícito.
        from rag_cliente.marker_llm import validate_marker_local_only

        validate_marker_local_only(settings)

    _require_marker_2()
    os.environ["TORCH_DEVICE"] = profile.torch_device

    if profile.inference_backend is None:
        os.environ.pop("SURYA_INFERENCE_BACKEND", None)
    else:
        os.environ["SURYA_INFERENCE_BACKEND"] = profile.inference_backend

    llama_cpp_binary = settings.llama_cpp_binary.strip()
    if profile.inference_backend == "llamacpp":
        resolved_binary = Path(llama_cpp_binary).expanduser().resolve()
        if not resolved_binary.is_file():
            raise RuntimeError(
                f"LLAMA_CPP_BINARY no existe o no es un archivo: {resolved_binary}"
            )
        os.environ["LLAMA_CPP_BINARY"] = str(resolved_binary)
        os.environ["SURYA_GGUF_LOCAL_MODEL_PATH"] = str(
            Path(settings.surya_gguf_path).expanduser().resolve()
            if settings.surya_gguf_path.strip()
            else (settings.models_path / "surya-ocr-2" / "surya-2.gguf").resolve()
        )
        os.environ["SURYA_GGUF_LOCAL_MMPROJ_PATH"] = str(
            Path(settings.surya_mmproj_path).expanduser().resolve()
            if settings.surya_mmproj_path.strip()
            else (settings.models_path / "surya-ocr-2" / "surya-2-mmproj.gguf").resolve()
        )
        # SURYA_INFERENCE_URL fuerza a Surya a adjuntarse al servidor local ya
        # administrado; así no crea PIDs ni descarga GGUF por su cuenta.
        os.environ["SURYA_INFERENCE_URL"] = settings.surya_base_url
        os.environ["SURYA_INFERENCE_PARALLEL"] = "1"
        os.environ["SURYA_INFERENCE_CTX_SIZE"] = str(settings.model_context_size)
        os.environ["SURYA_INFERENCE_STARTUP_TIMEOUT"] = str(settings.model_start_timeout)
        os.environ["SURYA_INFERENCE_TIMEOUT_SECONDS"] = str(settings.model_request_timeout)

    if profile.torch_device == "cuda":
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "PyTorch no esta instalado. Ejecuta setup.ps1 para instalar la version CUDA."
            ) from exc

        if not torch.cuda.is_available():
            raise RuntimeError(
                "El perfil gpu-quality requiere CUDA, pero PyTorch no detecta una GPU NVIDIA. "
                "Ejecuta setup.ps1 -Device cuda y comprueba el resultado con 'rag.bat gpu'."
            )

    try:
        from marker.config.parser import ConfigParser
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict
    except ImportError as exc:
        raise RuntimeError(
            "Marker 2 no está instalado con todos los proveedores. "
            "Ejecuta setup.ps1 para instalar marker-pdf[full]."
        ) from exc

    try:
        config_parser = ConfigParser(_build_marker_config(settings, profile))
        renderer = (
            "rag_cliente.pdf_loader.MarkerMarkdownRendererWithOcrMetadata"
            if settings.marker_markdown_compatibility
            else "rag_cliente.pdf_loader.MarkerJSONRendererWithOcrMetadata"
        )
        llm_service = None
        if profile.use_llm:
            from rag_cliente.marker_llm import BudgetedMarkerOpenAIService

            llm_service = BudgetedMarkerOpenAIService(settings)

        converter = PdfConverter(
            config=config_parser.generate_config_dict(),
            artifact_dict=create_model_dict(
                inference_backend=profile.inference_backend
            ),
            processor_list=config_parser.get_processors(),
            renderer=renderer,
            llm_service=llm_service,
        )
        if llm_service is not None:
            setattr(converter, "_rag_llm_service", llm_service)
        return converter
    except Exception as exc:
        raise RuntimeError(f"No se pudo inicializar Marker: {exc}") from exc


def _render_with_marker(path: Path, marker_converter: Any) -> tuple[Any, dict[str, Any]]:
    """Ejecuta el converter recibido y conserva su salida oficial completa."""
    llm_service = getattr(marker_converter, "_rag_llm_service", None)
    if llm_service is None:
        rendered = marker_converter(str(path))
    else:
        with llm_service.document_budget(str(path.resolve())):
            rendered = marker_converter(str(path))
    metadata = getattr(rendered, "metadata", {})
    if isinstance(rendered, dict):
        metadata = rendered.get("metadata", metadata)
    if not isinstance(metadata, dict):
        metadata = {}
    return rendered, metadata


def _marker_markdown_text(rendered: Any) -> str:
    """Obtiene texto del renderer Markdown oficial para el flag de compatibilidad."""
    try:
        from marker.output import text_from_rendered
    except ImportError as exc:
        raise RuntimeError(
            "Marker está instalado parcialmente o cambió su API: no se pudo importar "
            "marker.output.text_from_rendered. Reinstala/actualiza marker-pdf."
        ) from exc

    text, _, _images = text_from_rendered(rendered)
    return text if isinstance(text, str) else ""


def _to_plain_data(value: Any) -> Any:
    """Convierte modelos Pydantic de Marker a tipos JSON sin conocer sus clases."""
    if hasattr(value, "model_dump"):
        try:
            dumped_value = value.model_dump(mode="json")
        except TypeError:
            dumped_value = value.model_dump()
        return _to_plain_data(dumped_value)
    if isinstance(value, dict):
        return {str(key): _to_plain_data(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain_data(item) for item in value]
    return value


def _block_page_number(block: dict[str, Any], inherited_page: int | None = None) -> int:
    """Obtiene página humana 1-based desde campos o IDs públicos de Marker."""
    for key in ("page_id", "page"):
        if key in block:
            page_number = _as_1_based_page_number(block.get(key))
            if page_number is not None:
                return page_number

    block_id = str(block.get("id", ""))
    match = re.search(r"(?:^|/)page/(\d+)(?:/|$)", block_id, flags=re.IGNORECASE)
    if match:
        return int(match.group(1)) + 1
    return inherited_page or 1


def _text_from_marker_block(block: dict[str, Any]) -> str:
    """Deriva texto indexable sin perder la estructura JSON original."""
    direct_text = block.get("text")
    if isinstance(direct_text, str) and direct_text.strip():
        return _normalize_text(direct_text)

    direct_html = block.get("html")
    if isinstance(direct_html, str) and direct_html.strip():
        without_tags = re.sub(r"<[^>]+>", " ", direct_html)
        normalized_html_text = re.sub(
            r"[ \t]+",
            " ",
            html_lib.unescape(without_tags),
        )
        normalized_html_text = _normalize_text(normalized_html_text)
        if normalized_html_text:
            return normalized_html_text

    children = block.get("children")
    child_texts = [
        _text_from_marker_block(child)
        for child in (children if isinstance(children, list) else [])
        if isinstance(child, dict)
    ]
    return _normalize_text("\n\n".join(text for text in child_texts if text))


def _page_extraction_metadata(metadata: dict[str, Any], page_number: int) -> dict[str, Any]:
    """Conserva metadata general y la estadística correspondiente a la página."""
    page_stats = metadata.get("page_stats")
    selected_page_stats: list[dict[str, Any]] = []
    if isinstance(page_stats, list):
        for page_stat in page_stats:
            if not isinstance(page_stat, dict):
                continue
            if _as_1_based_page_number(page_stat.get("page_id")) == page_number:
                selected_page_stats.append(dict(page_stat))

    extraction_metadata = {
        key: _to_plain_data(value)
        for key, value in metadata.items()
        if key != "page_stats"
    }
    if selected_page_stats:
        extraction_metadata["page_stats"] = selected_page_stats
    return extraction_metadata


def _extract_marker_structured_chunks(
    rendered: Any,
    metadata: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Normaliza la salida JSON oficial al contrato estructurado del RAG."""
    payload = _to_plain_data(rendered)
    if not isinstance(payload, dict):
        return []

    effective_metadata = dict(metadata or {})
    payload_metadata = payload.get("metadata")
    if isinstance(payload_metadata, dict):
        effective_metadata.update(payload_metadata)

    root_children = payload.get("children")
    candidates = (
        [child for child in root_children if isinstance(child, dict)]
        if isinstance(root_children, list)
        else [payload]
    )

    page_candidates = [
        block
        for block in candidates
        if str(block.get("block_type", "")).strip().lower() == "page"
    ]
    blocks = page_candidates or candidates
    structured_chunks: list[dict[str, Any]] = []

    for block in blocks:
        page_number = _block_page_number(block)
        children = block.get("children")
        plain_children = (
            [_to_plain_data(child) for child in children]
            if isinstance(children, list)
            else []
        )
        polygon = _to_plain_data(block.get("polygon") or [])
        if isinstance(polygon, dict):
            polygon = polygon.get("polygon", [])
        if not isinstance(polygon, list):
            polygon = []

        section_hierarchy = _to_plain_data(block.get("section_hierarchy") or {})
        if not isinstance(section_hierarchy, dict):
            section_hierarchy = {}

        structured_chunks.append(
            {
                "block_type": str(block.get("block_type") or "Unknown"),
                "id": str(block.get("id") or ""),
                "html": str(block.get("html") or ""),
                "text": _text_from_marker_block(block),
                "page": page_number,
                "polygon": polygon,
                "children": plain_children,
                "section_hierarchy": section_hierarchy,
                "extraction_metadata": _page_extraction_metadata(
                    effective_metadata,
                    page_number,
                ),
            }
        )

    return structured_chunks


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
    settings: Settings,
    default_ocr_used: bool = False,
) -> list[PageDocument]:
    """Adapta JSON estructurado o el Markdown temporal a `PageDocument`."""
    try:
        rendered, metadata = _render_with_marker(path, marker_converter)
    except Exception as exc:
        from rag_cliente.marker_llm import MarkerLLMError

        if isinstance(exc, MarkerLLMError):
            raise
        raise RuntimeError(f"Marker falló procesando '{path.name}': {exc}") from exc

    marker_ocr_usage = _extract_marker_ocr_usage_by_page(metadata)
    document_ocr_used = any(marker_ocr_usage.values()) if marker_ocr_usage else default_ocr_used

    if settings.marker_markdown_compatibility:
        markdown = _marker_markdown_text(rendered)
        structured_chunks = [
            {
                "block_type": "Page",
                "id": f"/page/{page_number - 1}",
                "html": "",
                "text": text,
                "page": page_number,
                "polygon": [],
                "children": [],
                "section_hierarchy": {},
                "extraction_metadata": _page_extraction_metadata(metadata, page_number),
            }
            for page_number, text in _split_marker_markdown_by_page(markdown)
        ]
    else:
        structured_chunks = _extract_marker_structured_chunks(rendered, metadata)

    pages: list[PageDocument] = []
    for chunk in structured_chunks:
        page_number = int(chunk["page"])
        text = str(chunk["text"])
        if not text:
            continue
        ocr_used = marker_ocr_usage.get(
            page_number,
            document_ocr_used if len(structured_chunks) == 1 else default_ocr_used,
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
                block_type=str(chunk["block_type"]),
                id=str(chunk["id"]),
                html=str(chunk["html"]),
                page=page_number,
                polygon=chunk["polygon"],
                children=chunk["children"],
                section_hierarchy=chunk["section_hierarchy"],
                extraction_metadata=chunk["extraction_metadata"],
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
    profile = resolve_marker_profile(settings)
    return _marker_document_pages(
        document_path,
        source_type=suffix.lstrip("."),
        marker_converter=converter,
        settings=settings,
        default_ocr_used=suffix in IMAGE_SUFFIXES and not profile.disable_ocr,
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
                    f"({resolve_marker_profile(settings).name}): {display_path}",
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
            from rag_cliente.marker_llm import MarkerLLMError

            if isinstance(exc, MarkerLLMError):
                raise
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
