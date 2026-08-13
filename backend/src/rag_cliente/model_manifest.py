"""Manifest for the local embedding and answer models.

Document parsing is handled by Bedrock. This module only plans, downloads and
checks local GGUF files; it never loads a model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

from rag_cliente.config import Settings

ModelProfile = Literal["cpu", "gpu"]


@dataclass(frozen=True, slots=True)
class ModelRoleSpec:
    key: str
    label: str
    repository: str
    directory: str
    quantization: str
    patterns: tuple[str, ...]
    expected_size: str
    profiles: tuple[ModelProfile, ...]


MODEL_MANIFEST: tuple[ModelRoleSpec, ...] = (
    ModelRoleSpec(
        key="embeddings_cpu",
        label="Embeddings CPU Qwen3 0.6B",
        repository="Qwen/Qwen3-Embedding-0.6B-GGUF",
        directory="qwen3-embedding-0.6b",
        quantization="Q8_0",
        patterns=("*Q8_0*.gguf",),
        expected_size="aprox. 0.6-0.8 GiB",
        profiles=("cpu",),
    ),
    ModelRoleSpec(
        key="embeddings_gpu",
        label="Embeddings GPU Qwen3 8B",
        repository="Qwen/Qwen3-Embedding-8B-GGUF",
        directory="qwen3-embedding-8b",
        quantization="Q8_0",
        patterns=("Qwen3-Embedding-8B-Q8_0.gguf", "*Q8_0*.gguf"),
        expected_size="aprox. 7.5-8.5 GiB",
        profiles=("gpu",),
    ),
    ModelRoleSpec(
        key="chat_cpu",
        label="Chat CPU Qwen3 4B",
        repository="Qwen/Qwen3-4B-GGUF",
        directory="qwen3-4b",
        quantization="Q4_K_M",
        patterns=("*Q4_K_M*.gguf",),
        expected_size="aprox. 2.3-2.8 GiB",
        profiles=("cpu",),
    ),
    ModelRoleSpec(
        key="chat_gpu",
        label="Chat GPU Qwen3 14B",
        repository="Qwen/Qwen3-14B-GGUF",
        directory="qwen3-14b",
        quantization="Q5_K_M",
        patterns=("Qwen3-14B-Q5_K_M.gguf", "*Q5_K_M*.gguf"),
        expected_size="aprox. 9.5-11 GiB",
        profiles=("gpu",),
    ),
)

_EXPLICIT_PATH_FIELDS = {
    "embeddings_cpu": ("embeddings_cpu_gguf_path", "embeddings_gguf_path"),
    "embeddings_gpu": ("embeddings_gpu_gguf_path", "embeddings_gguf_path"),
    "chat_cpu": ("chat_cpu_gguf_path",),
    "chat_gpu": ("chat_gpu_gguf_path",),
}


def roles_for_profile(profile: ModelProfile) -> tuple[ModelRoleSpec, ...]:
    return tuple(role for role in MODEL_MANIFEST if profile in role.profiles)


def get_role(key: str) -> ModelRoleSpec:
    for role in MODEL_MANIFEST:
        if role.key == key:
            return role
    raise KeyError(f"Rol de modelo desconocido: {key}")


def resolve_model_path(settings: Settings, role: ModelRoleSpec) -> Path:
    for field_name in _EXPLICIT_PATH_FIELDS[role.key]:
        configured = str(getattr(settings, field_name, "") or "").strip()
        if configured:
            return Path(configured).expanduser().resolve()

    role_dir = (settings.models_path / role.directory).resolve()
    if role_dir.exists():
        for pattern in role.patterns:
            candidates = sorted(
                path
                for path in role_dir.rglob(pattern)
                if path.is_file()
            )
            if candidates:
                return candidates[0]

    fallback_name = role.patterns[0].replace("*", "") or "model.gguf"
    return role_dir / fallback_name


def resolve_runtime_model_path(settings: Settings, role_key: str) -> Path:
    return resolve_model_path(settings, get_role(role_key))


def check_artifact(path: Path, kind: str = "model") -> tuple[bool, str]:
    """Validate the GGUF container without loading any tensors."""
    if kind != "model":
        return False, f"tipo de artefacto no soportado: {kind}"
    if not path.is_file():
        return False, "no existe"
    if path.stat().st_size < 16:
        return False, "archivo demasiado pequeño"
    try:
        with path.open("rb") as handle:
            if handle.read(4) != b"GGUF":
                return False, "cabecera GGUF inválida"
    except OSError as exc:
        return False, f"no se pudo leer: {exc}"
    return True, "GGUF válido"


def check_role(settings: Settings, role: ModelRoleSpec) -> dict:
    path = resolve_model_path(settings, role)
    valid, message = check_artifact(path)
    return {
        "role": role.key,
        "label": role.label,
        "repository": role.repository,
        "quantization": role.quantization,
        "valid": valid,
        "artifacts": [
            {
                "kind": "model",
                "path": str(path),
                "expected_size": role.expected_size,
                "patterns": list(role.patterns),
                "valid": valid,
                "message": message,
            }
        ],
    }


def check_models(settings: Settings, profile: ModelProfile | None = None) -> list[dict]:
    roles: Iterable[ModelRoleSpec] = (
        roles_for_profile(profile) if profile is not None else MODEL_MANIFEST
    )
    return [check_role(settings, role) for role in roles]


def plan_models(settings: Settings, profile: ModelProfile) -> list[dict]:
    return check_models(settings, profile)


def download_models(settings: Settings, profile: ModelProfile) -> list[dict]:
    """Download only the explicitly selected local profile."""
    from huggingface_hub import snapshot_download

    results = []
    settings.models_path.mkdir(parents=True, exist_ok=True)
    for role in roles_for_profile(profile):
        local_dir = (settings.models_path / role.directory).resolve()
        local_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=role.repository,
            local_dir=local_dir,
            allow_patterns=list(role.patterns),
        )
        path = resolve_model_path(settings, role)
        valid, message = check_artifact(path)
        results.append(
            {
                "role": role.key,
                "downloaded": valid,
                "message": str(local_dir) if valid else message,
            }
        )
    return results
