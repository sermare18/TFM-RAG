"""Capacidades declaradas y versión validada del parser Marker."""

from __future__ import annotations

import warnings
from importlib.metadata import PackageNotFoundError, version
from typing import Any

VALIDATED_MARKER_VERSION = "2.0.0"


def installed_marker_version() -> str | None:
    """Devuelve la versión instalada sin importar ni inicializar Marker."""
    try:
        return version("marker-pdf")
    except PackageNotFoundError:
        return None


def marker_version_status() -> dict[str, Any]:
    """Describe compatibilidad sin bloquear otros componentes del RAG."""
    installed = installed_marker_version()
    matches = installed == VALIDATED_MARKER_VERSION
    if installed is None:
        detail = (
            "marker-pdf no está instalado; las tablas se validaron con "
            f"Marker {VALIDATED_MARKER_VERSION}"
        )
    elif matches:
        detail = f"Marker {installed}; versión validada para tablas"
    else:
        detail = (
            f"Marker instalado: {installed}; las capacidades y tests de tablas "
            f"solo se validaron con Marker {VALIDATED_MARKER_VERSION}"
        )
    return {
        "installed": installed,
        "validated": VALIDATED_MARKER_VERSION,
        "matches_validated": matches,
        "detail": detail,
    }


def marker_capabilities() -> dict[str, Any]:
    """Contrato explícito de capacidades reales de la fase 3."""
    status = marker_version_status()
    return {
        "marker_version": status["installed"] or VALIDATED_MARKER_VERSION,
        "multipage_table_merge": "preclassified_tables_only",
        "text_to_table_reclassification": False,
        "complete_multipage_table_support": False,
    }


def require_marker_installed_and_warn_if_unvalidated() -> str:
    """Exige Marker solo para parsear y advierte ante una versión distinta."""
    status = marker_version_status()
    installed = status["installed"]
    if installed is None:
        raise RuntimeError(
            "Marker no está instalado. Ejecuta setup.ps1 para instalar "
            f"marker-pdf[full]=={VALIDATED_MARKER_VERSION}."
        )
    if not status["matches_validated"]:
        warnings.warn(status["detail"], RuntimeWarning, stacklevel=2)
    return installed
