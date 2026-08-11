"""Comando manual acotado para diagnosticar el parser estructurado."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from rag_cliente.config import Settings, resolve_marker_profile
from rag_cliente.model_supervisor import ModelSupervisor
from rag_cliente.pdf_loader import load_pdf_pages, parsed_document_from_pages
from rag_cliente.resource_coordinator import get_resource_coordinator


def run_smoke_parser(pdf_path: Path, settings: Settings) -> dict[str, Any]:
    """Procesa manualmente solo el rango configurado y devuelve diagnóstico JSON."""
    resolved_path = pdf_path.expanduser().resolve()
    if not resolved_path.is_file():
        raise FileNotFoundError(f"PDF no encontrado: {resolved_path}")
    if resolved_path.suffix.lower() != ".pdf":
        raise ValueError("smoke-parser solo admite un archivo PDF")
    if not settings.marker_page_range.strip():
        raise ValueError("smoke-parser requiere un rango explícito en --pages")

    profile = resolve_marker_profile(settings)
    parser_roles = ("surya", "vlm") if profile.use_llm else ()
    supervisor = ModelSupervisor(settings) if settings.model_supervision_enabled else None
    coordinator = get_resource_coordinator()
    started_roles: list[str] = []
    started_at = time.monotonic()

    with coordinator.acquire(
        "parser_bundle",
        workload="index",
        timeout=settings.parser_job_timeout,
    ):
        try:
            if supervisor is not None:
                for role in parser_roles:
                    supervisor.ensure_started(role)
                    started_roles.append(role)
            pages = load_pdf_pages(resolved_path, settings=settings)
        finally:
            if supervisor is not None:
                supervisor.stop_bundle(started_roles)

    elapsed = time.monotonic() - started_at
    if elapsed > settings.parser_job_timeout:
        raise TimeoutError(
            f"PARSER_JOB_TIMEOUT excedido: {elapsed:.1f}s > "
            f"{settings.parser_job_timeout:g}s"
        )

    parsed = parsed_document_from_pages(resolved_path, pages)
    parsed.metadata["diagnostic"] = {
        "command": "smoke-parser",
        "profile": profile.name,
        "requested_page_range": settings.marker_page_range,
        "parser_timeout_seconds": settings.parser_job_timeout,
        "elapsed_seconds": elapsed,
    }
    return parsed.as_dict()
