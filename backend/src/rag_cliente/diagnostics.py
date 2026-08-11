"""Read-only diagnostics for Bedrock configuration and local models."""

from __future__ import annotations

import shutil
from importlib.util import find_spec
from pathlib import Path

from rag_cliente.config import (
    CLAUDE_SONNET_4_6_GLOBAL_MODEL_ID,
    Settings,
    resolve_local_model_profile,
)
from rag_cliente.local_endpoints import is_local_model_endpoint
from rag_cliente.model_manifest import check_models
from rag_cliente.model_supervisor import detect_hardware


def run_doctor(settings: Settings) -> dict:
    """Validate configuration without loading models or contacting AWS."""
    hardware = detect_hardware()
    profile = resolve_local_model_profile(
        settings,
        gpu_available=hardware.nvidia_available,
    )
    binary = Path(settings.llama_cpp_binary).expanduser().resolve()
    models_dir = settings.models_path.resolve()
    disk_anchor = models_dir if models_dir.exists() else models_dir.parent
    while not disk_anchor.exists() and disk_anchor != disk_anchor.parent:
        disk_anchor = disk_anchor.parent
    disk = shutil.disk_usage(disk_anchor)
    reports = check_models(settings, profile)
    endpoints = (
        settings.llama_cpp_embedding_base_url,
        settings.llama_cpp_chat_base_url,
    )
    bedrock_configured = bool(
        settings.aws_region.strip() and settings.bedrock_model_id.strip()
    )
    global_claude_profile = (
        settings.bedrock_model_id.strip() == CLAUDE_SONNET_4_6_GLOBAL_MODEL_ID
    )
    boto3_installed = find_spec("boto3") is not None

    checks = [
        {
            "name": "llama_cpp_binary",
            "ok": binary.is_file(),
            "required": settings.model_supervision_enabled,
            "detail": str(binary),
        },
        {
            "name": "models_dir",
            "ok": models_dir.exists() or models_dir.parent.exists(),
            "detail": str(models_dir),
        },
        {
            "name": "disk_free",
            "ok": disk.free >= 8 * 1024**3,
            "detail": f"{disk.free / 1024**3:.1f} GiB libres",
        },
        {
            "name": "local_model_endpoints",
            "ok": all(
                is_local_model_endpoint(url, settings.allowed_local_model_hosts)
                for url in endpoints
            ),
            "detail": ", ".join(endpoints),
        },
        {
            "name": "local_models",
            "ok": all(report["valid"] for report in reports),
            "detail": f"{sum(report['valid'] for report in reports)}/{len(reports)} roles validos",
        },
        {
            "name": "bedrock_configuration",
            "ok": bedrock_configured,
            "required": settings.bedrock_enabled,
            "detail": (
                f"enabled={settings.bedrock_enabled}; region="
                f"{settings.aws_region or '[sin configurar]'}; model="
                f"{settings.bedrock_model_id or '[sin configurar]'}"
            ),
        },
        {
            "name": "boto3_installed",
            "ok": boto3_installed,
            "required": settings.bedrock_enabled,
            "detail": "instalado" if boto3_installed else "ejecuta setup.ps1",
        },
        {
            "name": "bedrock_global_claude_profile",
            "ok": global_claude_profile,
            "required": settings.bedrock_enabled,
            "detail": settings.bedrock_model_id or "[sin configurar]",
        },
        {
            "name": "bedrock_pages_per_batch",
            "ok": settings.bedrock_pages_per_batch == 4,
            "detail": str(settings.bedrock_pages_per_batch),
        },
    ]
    return {
        "ok": all(item["ok"] or not item.get("required", True) for item in checks),
        "hardware": {
            "cpu_threads": hardware.cpu_threads,
            "nvidia_available": hardware.nvidia_available,
            "nvidia_gpus": list(hardware.nvidia_gpus),
        },
        "local_model_profile": profile,
        "supervision_enabled": settings.model_supervision_enabled,
        "bedrock": {
            "enabled": settings.bedrock_enabled,
            "configured": bedrock_configured,
            "boto3_installed": boto3_installed,
            "global_claude_profile": global_claude_profile,
            "pages_per_batch": settings.bedrock_pages_per_batch,
            "cache_dir": str(settings.bedrock_cache_path),
            "network_checked": False,
        },
        "checks": checks,
        "models": reports,
    }
