"""Diagnóstico local y no invasivo de la fase de modelos."""

from __future__ import annotations

import shutil
from pathlib import Path

from rag_cliente.config import Settings, resolve_marker_profile
from rag_cliente.local_endpoints import is_local_model_endpoint
from rag_cliente.marker_capabilities import marker_capabilities, marker_version_status
from rag_cliente.model_manifest import check_models
from rag_cliente.model_supervisor import detect_hardware


def run_doctor(settings: Settings) -> dict:
    """Valida configuración, hardware y disco sin iniciar ni cargar modelos."""
    hardware = detect_hardware()
    profile = resolve_marker_profile(
        settings,
        cuda_available=hardware.nvidia_available,
    )
    binary = Path(settings.llama_cpp_binary).expanduser().resolve()
    models_dir = settings.models_path.resolve()
    disk_anchor = models_dir if models_dir.exists() else models_dir.parent
    while not disk_anchor.exists() and disk_anchor != disk_anchor.parent:
        disk_anchor = disk_anchor.parent
    disk = shutil.disk_usage(disk_anchor)
    model_profile = "gpu" if profile.name == "gpu-quality" else "cpu"
    required_disk_gib = 20 if model_profile == "gpu" else 12
    model_reports = check_models(settings, model_profile)
    marker_status = marker_version_status()
    endpoints = {
        "surya": settings.surya_base_url,
        "vlm": settings.marker_openai_base_url,
        "embeddings": settings.llama_cpp_embedding_base_url,
        "chat": settings.llama_cpp_chat_base_url,
    }

    checks = [
        {
            "name": "llama_cpp_binary",
            "ok": binary.is_file(),
            "detail": str(binary),
        },
        {
            "name": "marker_installed",
            "ok": marker_status["installed"] is not None,
            "required": settings.marker_enabled,
            "detail": marker_status["detail"],
        },
        {
            "name": "marker_validated_version",
            "ok": marker_status["matches_validated"],
            "required": False,
            "detail": marker_status["detail"],
        },
        {
            "name": "models_dir",
            "ok": models_dir.exists() or models_dir.parent.exists(),
            "detail": str(models_dir),
        },
        {
            "name": "disk_free",
            "ok": disk.free >= required_disk_gib * 1024**3,
            "detail": (
                f"{disk.free / 1024**3:.1f} GiB libres; "
                f"presupuesto mínimo del perfil: {required_disk_gib} GiB"
            ),
        },
        {
            "name": "marker_local_endpoint",
            "ok": (
                not profile.use_llm
                or is_local_model_endpoint(
                    settings.marker_openai_base_url,
                    settings.allowed_local_model_hosts,
                )
            ),
            "detail": settings.marker_openai_base_url,
        },
        {
            "name": "all_model_endpoints_local",
            "ok": all(
                is_local_model_endpoint(url, settings.allowed_local_model_hosts)
                for url in endpoints.values()
            ),
            "detail": ", ".join(f"{role}={url}" for role, url in endpoints.items()),
        },
        {
            "name": "marker_no_external_fallback",
            "ok": not settings.marker_llm_fallback_to_base,
            "detail": f"fallback_to_base={settings.marker_llm_fallback_to_base}",
        },
        {
            "name": "models",
            "ok": all(report["valid"] for report in model_reports),
            "detail": f"perfil {model_profile}: {sum(report['valid'] for report in model_reports)}/{len(model_reports)} roles válidos",
        },
    ]
    return {
        "ok": all(
            check["ok"] or not check.get("required", True)
            for check in checks
        ),
        "hardware": {
            "cpu_threads": hardware.cpu_threads,
            "nvidia_available": hardware.nvidia_available,
            "nvidia_gpus": list(hardware.nvidia_gpus),
        },
        "profile": profile.name,
        "capabilities": marker_capabilities(),
        "warnings": (
            [] if marker_status["matches_validated"] else [marker_status["detail"]]
        ),
        "supervision_enabled": settings.model_supervision_enabled,
        "checks": checks,
        "models": model_reports,
    }
