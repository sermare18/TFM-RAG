"""Manifiesto local de modelos y operaciones plan/download/check.

El manifiesto nunca carga modelos. La descarga solo se ejecuta desde el comando
explícito ``models download`` y usa patrones GGUF acotados por rol.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

from rag_cliente.config import Settings

ArtifactKind = Literal["model", "mmproj"]
ModelProfile = Literal["cpu", "gpu"]


@dataclass(frozen=True, slots=True)
class ArtifactSpec:
    kind: ArtifactKind
    patterns: tuple[str, ...]
    expected_size: str


@dataclass(frozen=True, slots=True)
class ModelRoleSpec:
    key: str
    label: str
    repository: str | None
    directory: str
    quantization: str
    artifacts: tuple[ArtifactSpec, ...]
    profiles: tuple[ModelProfile, ...]
    configurable_only: bool = False


MODEL_MANIFEST: tuple[ModelRoleSpec, ...] = (
    ModelRoleSpec(
        key="surya",
        label="Surya OCR 2",
        repository="datalab-to/surya-ocr-2-gguf",
        directory="surya-ocr-2",
        quantization="publicación oficial",
        artifacts=(
            ArtifactSpec("model", ("surya-2.gguf",), "aprox. 1-4 GiB"),
            ArtifactSpec("mmproj", ("surya-2-mmproj.gguf",), "aprox. 0.2-1 GiB"),
        ),
        profiles=("cpu", "gpu"),
    ),
    ModelRoleSpec(
        key="vlm_cpu",
        label="VLM CPU Qwen3-VL 2B",
        repository="Qwen/Qwen3-VL-2B-Instruct-GGUF",
        directory="qwen3-vl-2b",
        quantization="Q4_K_M",
        artifacts=(
            ArtifactSpec("model", ("*Q4_K_M*.gguf",), "aprox. 1.3-1.8 GiB"),
            ArtifactSpec("mmproj", ("*mmproj*.gguf",), "aprox. 0.3-0.8 GiB"),
        ),
        profiles=("cpu",),
    ),
    ModelRoleSpec(
        key="vlm_gpu",
        label="VLM GPU Qwen3-VL 8B",
        repository="Qwen/Qwen3-VL-8B-Instruct-GGUF",
        directory="qwen3-vl-8b",
        quantization="Q4_K_M",
        artifacts=(
            ArtifactSpec("model", ("*Q4_K_M*.gguf",), "aprox. 4.5-5.5 GiB"),
            ArtifactSpec("mmproj", ("*mmproj*.gguf",), "aprox. 0.4-1 GiB"),
        ),
        profiles=("gpu",),
    ),
    ModelRoleSpec(
        key="embeddings",
        label="Embeddings Qwen3 0.6B",
        repository="Qwen/Qwen3-Embedding-0.6B-GGUF",
        directory="qwen3-embedding-0.6b",
        quantization="Q8_0",
        artifacts=(
            ArtifactSpec("model", ("*Q8_0*.gguf",), "aprox. 0.6-0.8 GiB"),
        ),
        profiles=("cpu", "gpu"),
    ),
    ModelRoleSpec(
        key="chat_cpu",
        label="Chat CPU Qwen3 4B",
        repository="Qwen/Qwen3-4B-GGUF",
        directory="qwen3-4b",
        quantization="Q4_K_M",
        artifacts=(
            ArtifactSpec("model", ("*Q4_K_M*.gguf",), "aprox. 2.3-2.8 GiB"),
        ),
        profiles=("cpu",),
    ),
    ModelRoleSpec(
        key="chat_gpu",
        label="Chat GPU Qwen3.5 9B",
        repository="bartowski/Qwen_Qwen3.5-9B-GGUF",
        directory="qwen3.5-9b",
        quantization="Q4_K_M",
        artifacts=(
            ArtifactSpec(
                "model",
                ("Qwen_Qwen3.5-9B-Q4_K_M.gguf",),
                "aprox. 5-7 GiB",
            ),
        ),
        profiles=("gpu",),
    ),
)

_EXPLICIT_PATH_FIELDS: dict[tuple[str, ArtifactKind], str] = {
    ("surya", "model"): "surya_gguf_path",
    ("surya", "mmproj"): "surya_mmproj_path",
    ("vlm_cpu", "model"): "vlm_cpu_gguf_path",
    ("vlm_cpu", "mmproj"): "vlm_cpu_mmproj_path",
    ("vlm_gpu", "model"): "vlm_gpu_gguf_path",
    ("vlm_gpu", "mmproj"): "vlm_gpu_mmproj_path",
    ("embeddings", "model"): "embeddings_gguf_path",
    ("chat_cpu", "model"): "chat_cpu_gguf_path",
    ("chat_gpu", "model"): "chat_gpu_gguf_path",
}


def roles_for_profile(profile: ModelProfile) -> tuple[ModelRoleSpec, ...]:
    return tuple(role for role in MODEL_MANIFEST if profile in role.profiles)


def get_role(key: str) -> ModelRoleSpec:
    for role in MODEL_MANIFEST:
        if role.key == key:
            return role
    raise KeyError(f"Rol de modelo desconocido: {key}")


def _artifact_matches(path: Path, artifact: ArtifactSpec) -> bool:
    lowered = path.name.lower()
    if artifact.kind == "model" and "mmproj" in lowered:
        return False
    if artifact.kind == "mmproj" and "mmproj" not in lowered:
        return False
    return True


def resolve_artifact_path(
    settings: Settings,
    role: ModelRoleSpec,
    artifact: ArtifactSpec,
) -> Path:
    """Resuelve primero una ruta explícita y después el directorio del rol."""
    field_name = _EXPLICIT_PATH_FIELDS.get((role.key, artifact.kind))
    configured = str(getattr(settings, field_name, "") or "").strip()
    if configured:
        return Path(configured).expanduser().resolve()

    role_dir = (settings.models_path / role.directory).resolve()
    for pattern in artifact.patterns:
        candidates = sorted(
            path
            for path in role_dir.rglob(pattern)
            if path.is_file() and _artifact_matches(path, artifact)
        ) if role_dir.exists() else []
        if candidates:
            return candidates[0]

    # La ruta aún inexistente es informativa para plan/check.
    fallback_name = artifact.patterns[0].replace("*", "") or f"{artifact.kind}.gguf"
    return role_dir / fallback_name


def resolve_runtime_role_paths(
    settings: Settings,
    role_key: str,
) -> tuple[Path, Path | None]:
    """Devuelve rutas explícitas GGUF/mmproj para construir llama-server."""
    role = get_role(role_key)
    model_spec = next(item for item in role.artifacts if item.kind == "model")
    mmproj_spec = next(
        (item for item in role.artifacts if item.kind == "mmproj"),
        None,
    )
    model_path = resolve_artifact_path(settings, role, model_spec)
    mmproj_path = (
        resolve_artifact_path(settings, role, mmproj_spec)
        if mmproj_spec is not None
        else None
    )
    return model_path, mmproj_path


def _is_gguf(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            return handle.read(4) == b"GGUF"
    except OSError:
        return False


def _has_visual_metadata(path: Path) -> bool:
    """Busca metadatos de proyector sin cargar tensores ni el modelo."""
    try:
        with path.open("rb") as handle:
            sample = handle.read(8 * 1024 * 1024).lower()
    except OSError:
        return False
    return any(marker in sample for marker in (b"clip.", b"vision", b"projector"))


def check_artifact(path: Path, kind: ArtifactKind) -> tuple[bool, str]:
    if not path.is_file():
        return False, "no existe"
    if path.stat().st_size < 16:
        return False, "archivo demasiado pequeño"
    if not _is_gguf(path):
        return False, "cabecera GGUF inválida"
    if kind == "mmproj" and not _has_visual_metadata(path):
        return False, "GGUF sin metadatos visuales/mmproj reconocibles"
    return True, "GGUF válido"


def check_role(settings: Settings, role: ModelRoleSpec) -> dict:
    artifacts = []
    for artifact in role.artifacts:
        path = resolve_artifact_path(settings, role, artifact)
        valid, message = check_artifact(path, artifact.kind)
        artifacts.append(
            {
                "kind": artifact.kind,
                "path": str(path),
                "expected_size": artifact.expected_size,
                "patterns": list(artifact.patterns),
                "valid": valid,
                "message": message,
            }
        )
    return {
        "role": role.key,
        "label": role.label,
        "repository": role.repository,
        "quantization": role.quantization,
        "valid": all(item["valid"] for item in artifacts),
        "artifacts": artifacts,
    }


def plan_models(settings: Settings, profile: ModelProfile) -> list[dict]:
    """Construye un plan puramente local; no importa clientes de descarga."""
    return check_models(settings, profile)


def check_models(settings: Settings, profile: ModelProfile | None = None) -> list[dict]:
    roles: Iterable[ModelRoleSpec] = (
        roles_for_profile(profile) if profile is not None else MODEL_MANIFEST
    )
    reports = [check_role(settings, role) for role in roles]

    # Qwen3.5 9B también puede sustituir al VLM GPU, pero solo si hay mmproj
    # válido con metadatos visuales. No se acepta por nombre ni por suposición.
    custom_model = settings.vlm_gpu_custom_gguf_path.strip()
    custom_mmproj = settings.vlm_gpu_custom_mmproj_path.strip()
    if (profile in {None, "gpu"}) and (custom_model or custom_mmproj):
        model_path = Path(custom_model).expanduser().resolve()
        mmproj_path = Path(custom_mmproj).expanduser().resolve()
        model_valid, model_message = check_artifact(model_path, "model")
        mmproj_valid, mmproj_message = check_artifact(mmproj_path, "mmproj")
        custom_report = {
                "role": "vlm_gpu_custom",
                "label": "VLM GPU Qwen3.5 9B local alternativo",
                "repository": None,
                "quantization": "Q4_K_M configurable",
                "valid": model_valid and mmproj_valid,
                "artifacts": [
                    {"kind": "model", "path": str(model_path), "valid": model_valid, "message": model_message},
                    {"kind": "mmproj", "path": str(mmproj_path), "valid": mmproj_valid, "message": mmproj_message},
                ],
            }
        if profile == "gpu":
            reports = [report for report in reports if report["role"] != "vlm_gpu"]
        reports.append(custom_report)
    return reports


def download_models(settings: Settings, profile: ModelProfile) -> list[dict]:
    """Descarga solo los roles solicitados y nunca inicia llama-server."""
    from huggingface_hub import snapshot_download

    results = []
    settings.models_path.mkdir(parents=True, exist_ok=True)
    for role in roles_for_profile(profile):
        if role.repository is None:
            results.append(
                {
                    "role": role.key,
                    "downloaded": False,
                    "message": "requiere una ruta local configurable; no hay descarga automática",
                }
            )
            continue

        local_dir = (settings.models_path / role.directory).resolve()
        local_dir.mkdir(parents=True, exist_ok=True)
        allow_patterns = [
            pattern
            for artifact in role.artifacts
            for pattern in artifact.patterns
        ]
        snapshot_download(
            repo_id=role.repository,
            local_dir=local_dir,
            allow_patterns=allow_patterns,
        )
        missing_artifacts = []
        for artifact in role.artifacts:
            artifact_path = resolve_artifact_path(settings, role, artifact)
            if not artifact_path.is_file():
                missing_artifacts.append(str(artifact_path))
        if missing_artifacts:
            results.append(
                {
                    "role": role.key,
                    "downloaded": False,
                    "message": (
                        "Hugging Face no devolvió los artefactos esperados: "
                        + ", ".join(missing_artifacts)
                    ),
                }
            )
            continue
        results.append(
            {
                "role": role.key,
                "downloaded": True,
                "message": str(local_dir),
            }
        )
    return results
