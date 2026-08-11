"""Configuración centralizada del sistema RAG.

Define `Settings`, que carga variables de entorno desde `.env` mediante
Pydantic Settings y ofrece propiedades `Path` para el resto del proyecto.

Grupos principales:
1. LLM y embeddings OpenAI-compatible.
2. LanceDB.
3. Recuperación, chunking y generación.
4. Directorios de documentos.
5. Parser de documentos con Marker.

Nota de migración:
El OCR basado en PaddleX/PaddleOCR se ha eliminado. Las variables antiguas
`ENABLE_OCR`, `OCR_*` se mantienen como campos obsoletos para que un `.env`
antiguo no rompa la carga de configuración, pero ya no controlan el parser de
PDFs. Para PDFs se usa Marker mediante las variables `MARKER_*`.

Sergio y Juan
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from dotenv import load_dotenv
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Carga variables de entorno desde ".env" antes de construir Settings.
load_dotenv()


MarkerProfileName = Literal["cpu-digital", "cpu-quality", "gpu-quality", "auto"]
ModelServerMode = Literal["managed", "external"]


@dataclass(frozen=True, slots=True)
class ResolvedMarkerProfile:
    """Configuración efectiva e inmutable de un perfil oficial de Marker."""

    name: Literal["cpu-digital", "cpu-quality", "gpu-quality"]
    mode: Literal["fast", "balanced"]
    disable_ocr: bool
    use_llm: bool
    torch_device: str
    inference_backend: str | None


_MARKER_PROFILES: dict[str, ResolvedMarkerProfile] = {
    "cpu-digital": ResolvedMarkerProfile(
        name="cpu-digital",
        mode="fast",
        disable_ocr=True,
        use_llm=False,
        torch_device="cpu",
        inference_backend=None,
    ),
    "cpu-quality": ResolvedMarkerProfile(
        name="cpu-quality",
        mode="fast",
        disable_ocr=False,
        use_llm=True,
        torch_device="cpu",
        inference_backend="llamacpp",
    ),
    "gpu-quality": ResolvedMarkerProfile(
        name="gpu-quality",
        mode="balanced",
        disable_ocr=False,
        use_llm=True,
        torch_device="cuda",
        inference_backend="llamacpp",
    ),
}


def has_usable_nvidia_gpu() -> bool:
    """Detecta CUDA de forma perezosa sin hacer que la configuración dependa de ella."""
    try:
        import torch
    except (ImportError, OSError):
        return False

    try:
        return bool(torch.cuda.is_available() and torch.cuda.device_count() > 0)
    except (AttributeError, RuntimeError):
        return False


def resolve_marker_profile(
    settings: "Settings",
    *,
    cuda_available: bool | None = None,
) -> ResolvedMarkerProfile:
    """Resuelve `auto`; cualquier nombre explícito se conserva sin fallback."""
    selected_profile = settings.marker_profile
    if selected_profile == "auto":
        usable_cuda = has_usable_nvidia_gpu() if cuda_available is None else cuda_available
        selected_profile = "gpu-quality" if usable_cuda else "cpu-quality"
    return _MARKER_PROFILES[selected_profile]


class Settings(BaseSettings):
    """Contenedor tipado de configuración del sistema RAG."""

    # ===================================================================
    # MODELOS Y ENDPOINTS
    # ===================================================================
    default_endpoint_model: str = "default"

    llama_cpp_chat_base_url: str = Field(
        default="http://127.0.0.1:8081/v1",
        alias="LLAMA_CPP_CHAT_BASE_URL",
    )
    llama_cpp_embedding_base_url: str = Field(
        default="http://127.0.0.1:8082/v1",
        alias="LLAMA_CPP_EMBEDDING_BASE_URL",
    )
    openai_api_key: str = Field(default="local-dev-key", alias="OPENAI_API_KEY")

    # ===================================================================
    # SUPERVISOR LOCAL DE LLAMA.CPP
    # ===================================================================
    # Se mantiene desactivado por defecto para que instalaciones existentes que
    # ya administran sus endpoints sigan funcionando. `.env.example` lo activa.
    model_supervision_enabled: bool = Field(
        default=False,
        alias="MODEL_SUPERVISION_ENABLED",
    )
    llama_cpp_binary: str = Field(
        default=r"C:\Users\SergioMartinReizabal\Documents\llama.cpp\llama-server.exe",
        alias="LLAMA_CPP_BINARY",
    )
    models_dir: str = Field(default="./models", alias="MODELS_DIR")
    model_logs_dir: str = Field(default="./data/logs/models", alias="MODEL_LOGS_DIR")

    model_start_timeout: float = Field(default=180.0, alias="MODEL_START_TIMEOUT", gt=0)
    model_request_timeout: float = Field(default=180.0, alias="MODEL_REQUEST_TIMEOUT", gt=0)
    model_stop_timeout: float = Field(default=15.0, alias="MODEL_STOP_TIMEOUT", gt=0)
    parser_job_timeout: float = Field(default=1800.0, alias="PARSER_JOB_TIMEOUT", gt=0)
    model_max_retries: int = Field(default=1, alias="MODEL_MAX_RETRIES", ge=0, le=1)
    model_health_connect_timeout: float = Field(
        default=5.0,
        alias="MODEL_HEALTH_CONNECT_TIMEOUT",
        gt=0,
    )
    model_health_read_timeout: float = Field(
        default=10.0,
        alias="MODEL_HEALTH_READ_TIMEOUT",
        gt=0,
    )
    model_chat_idle_timeout: float = Field(
        default=300.0,
        alias="MODEL_CHAT_IDLE_TIMEOUT",
        ge=0,
    )
    model_gpu_layers: int = Field(default=999, alias="MODEL_GPU_LAYERS", ge=0)
    model_context_size: int = Field(default=16384, alias="MODEL_CONTEXT_SIZE", ge=8192)

    model_surya_mode: ModelServerMode = Field(default="managed", alias="MODEL_SURYA_MODE")
    model_vlm_mode: ModelServerMode = Field(default="managed", alias="MODEL_VLM_MODE")
    model_embeddings_mode: ModelServerMode = Field(
        default="managed",
        alias="MODEL_EMBEDDINGS_MODE",
    )
    model_chat_mode: ModelServerMode = Field(default="managed", alias="MODEL_CHAT_MODE")

    surya_base_url: str = Field(default="http://127.0.0.1:8084/v1", alias="SURYA_BASE_URL")
    marker_openai_base_url: str = Field(
        default="http://127.0.0.1:8083/v1",
        alias="MARKER_OPENAI_BASE_URL",
    )
    marker_openai_model: str = Field(default="marker-vlm", alias="MARKER_OPENAI_MODEL")
    local_model_hosts: str = Field(default="", alias="LOCAL_MODEL_HOSTS")

    surya_gguf_path: str = Field(default="", alias="SURYA_GGUF_PATH")
    surya_mmproj_path: str = Field(default="", alias="SURYA_MMPROJ_PATH")
    vlm_cpu_gguf_path: str = Field(default="", alias="VLM_CPU_GGUF_PATH")
    vlm_cpu_mmproj_path: str = Field(default="", alias="VLM_CPU_MMPROJ_PATH")
    vlm_gpu_gguf_path: str = Field(default="", alias="VLM_GPU_GGUF_PATH")
    vlm_gpu_mmproj_path: str = Field(default="", alias="VLM_GPU_MMPROJ_PATH")
    vlm_gpu_custom_gguf_path: str = Field(default="", alias="VLM_GPU_CUSTOM_GGUF_PATH")
    vlm_gpu_custom_mmproj_path: str = Field(
        default="",
        alias="VLM_GPU_CUSTOM_MMPROJ_PATH",
    )
    embeddings_gguf_path: str = Field(default="", alias="EMBEDDINGS_GGUF_PATH")
    chat_cpu_gguf_path: str = Field(default="", alias="CHAT_CPU_GGUF_PATH")
    chat_gpu_gguf_path: str = Field(default="", alias="CHAT_GPU_GGUF_PATH")

    # ===================================================================
    # PRESUPUESTOS DEL VLM DE MARKER
    # ===================================================================
    marker_llm_max_requests: int = Field(default=50, alias="MARKER_LLM_MAX_REQUESTS", ge=1)
    marker_llm_max_tokens_per_request: int = Field(
        default=4096,
        alias="MARKER_LLM_MAX_TOKENS_PER_REQUEST",
        ge=1,
    )
    marker_llm_max_generated_tokens_per_document: int = Field(
        default=20000,
        alias="MARKER_LLM_MAX_GENERATED_TOKENS_PER_DOCUMENT",
        ge=1,
    )
    marker_llm_request_timeout: float = Field(
        default=180.0,
        alias="MARKER_LLM_REQUEST_TIMEOUT",
        gt=0,
    )
    marker_llm_job_timeout: float = Field(
        default=1800.0,
        alias="MARKER_LLM_JOB_TIMEOUT",
        gt=0,
    )
    marker_llm_max_retries: int = Field(
        default=1,
        alias="MARKER_LLM_MAX_RETRIES",
        ge=0,
        le=1,
    )
    marker_llm_fallback_to_base: bool = Field(
        default=False,
        alias="MARKER_LLM_FALLBACK_TO_BASE",
    )

    # ===================================================================
    # BASE DE DATOS VECTORIAL - LANCEDB
    # ===================================================================
    lancedb_uri: str = Field(default="./data/lancedb", alias="LANCEDB_URI")
    lancedb_table: str = Field(default="pdf_chunks", alias="LANCEDB_TABLE")

    # ===================================================================
    # BÚSQUEDA Y RECUPERACIÓN
    # ===================================================================
    # TOP_K se conserva como alias histórico. RETRIEVAL_TOP_K, si se define,
    # prevalece para el número final de resultados fusionados.
    top_k: int = Field(default=8, alias="TOP_K", ge=1)
    retrieval_top_k: int | None = Field(
        default=None,
        alias="RETRIEVAL_TOP_K",
        ge=1,
    )

    # ===================================================================
    # BÚSQUEDA HÍBRIDA - VECTORIAL + BM25
    # ===================================================================
    hybrid_search_enabled: bool = Field(default=True, alias="HYBRID_SEARCH_ENABLED")
    vector_weight: float = Field(default=0.65, alias="VECTOR_WEIGHT")
    bm25_weight: float = Field(default=0.35, alias="BM25_WEIGHT")
    bm25_top_k_multiplier: int = Field(default=3, alias="BM25_TOP_K_MULTIPLIER")
    bm25_index_dir: str = Field(default="./data/bm25", alias="BM25_INDEX_DIR")
    bm25_min_raw_score: float = Field(default=0.25, alias="BM25_MIN_RAW_SCORE")
    rrf_k: int = Field(default=60, alias="RRF_K", ge=1)
    vector_candidates: int = Field(default=40, alias="VECTOR_CANDIDATES", ge=1)
    bm25_candidates: int = Field(default=40, alias="BM25_CANDIDATES", ge=1)

    # ===================================================================
    # FRAGMENTACIÓN DE DOCUMENTOS
    # ===================================================================
    chunk_size: int = Field(default=700, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=100, alias="CHUNK_OVERLAP")
    chunk_target_tokens: int = Field(
        default=700,
        alias="CHUNK_TARGET_TOKENS",
        ge=1,
    )
    chunk_max_tokens: int = Field(
        default=900,
        alias="CHUNK_MAX_TOKENS",
        ge=1,
    )
    chunk_overlap_tokens: int = Field(
        default=100,
        alias="CHUNK_OVERLAP_TOKENS",
        ge=0,
    )
    table_chunk_max_tokens: int = Field(
        default=1200,
        alias="TABLE_CHUNK_MAX_TOKENS",
        ge=1,
    )

    # ===================================================================
    # GENERACIÓN DE RESPUESTAS
    # ===================================================================
    max_tokens: int = Field(default=1024, alias="MAX_TOKENS")
    reasoning_max_tokens: int = Field(default=1024, alias="REASONING_MAX_TOKENS")
    request_timeout: float = Field(default=60.0, alias="REQUEST_TIMEOUT")

    # ===================================================================
    # PROCESAMIENTO EN BATCH
    # ===================================================================
    embedding_batch_size: int = Field(default=16, alias="EMBEDDING_BATCH_SIZE")

    # ===================================================================
    # DIRECTORIOS Y ALMACENAMIENTO
    # ===================================================================
    data_dir: str = Field(default="./data", alias="DATA_DIR")
    documents_dir: str = Field(default="./data/pdfs", alias="DOCUMENTS_DIR")
    api_cors_allow_origins: list[str] = Field(default=["*"], alias="API_CORS_ALLOW_ORIGINS")

    # ===================================================================
    # MARKER 2 - PERFILES DEL PIPELINE OFICIAL
    # ===================================================================
    # Activo Marker full para PDF, Office, EPUB, HTML e imágenes. Si lo
    # desactivo, conservo fallbacks nativos para PDF digital, DOCX y TXT.
    marker_enabled: bool = Field(default=True, alias="MARKER_ENABLED")

    # Un perfil explícito nunca se sustituye según el hardware. Solo `auto`
    # consulta CUDA y elige entre gpu-quality y cpu-quality.
    marker_profile: MarkerProfileName = Field(
        default="auto",
        alias="MARKER_PROFILE",
    )

    # Alias obsoleto conservado para que un `.env` de fase 1 siga cargando. La
    # ejecución de fase 2 valida y utiliza exclusivamente LLAMA_CPP_BINARY.
    marker_llama_cpp_binary: str = Field(
        default=r"C:\Users\SergioMartinReizabal\Documents\llama.cpp\llama-server.exe",
        alias="MARKER_LLAMA_CPP_BINARY",
    )

    # Mantengo el umbral oficial de balanced como segunda red de seguridad para
    # tablas que lleguen al TableProcessor sin promover antes toda su página.
    marker_table_min_recon_score: float = Field(
        default=0.75,
        alias="MARKER_TABLE_MIN_RECON_SCORE",
        ge=0.0,
    )

    # Elimina texto OCR existente en el PDF antes de re-OCR. Útil para documentos
    # con capa OCR mala o duplicada.
    marker_strip_existing_ocr: bool = Field(default=False, alias="MARKER_STRIP_EXISTING_OCR")

    # Evita guardar imágenes extraídas al disco. Para RAG textual normalmente se
    # prefiere True para no generar artefactos innecesarios.
    marker_disable_image_extraction: bool = Field(
        default=True,
        alias="MARKER_DISABLE_IMAGE_EXTRACTION",
    )

    # Rango opcional de páginas en sintaxis Marker, por ejemplo "0,5-10,20".
    # Vacío = documento completo.
    marker_page_range: str = Field(default="", alias="MARKER_PAGE_RANGE")

    # La salida JSON oficial es primaria. Este flag conserva temporalmente la
    # ruta Markdown anterior para instalaciones que aún no hayan migrado.
    marker_markdown_compatibility: bool = Field(
        default=False,
        alias="MARKER_MARKDOWN_COMPATIBILITY",
    )

    model_config = SettingsConfigDict(
        populate_by_name=True,
        extra="ignore",
    )

    @field_validator(
        "llama_cpp_chat_base_url",
        "llama_cpp_embedding_base_url",
        "surya_base_url",
        "marker_openai_base_url",
    )
    @classmethod
    def reject_server_bind_addresses(cls, value: str) -> str:
        """Impide usar una direccion de escucha como destino HTTP cliente."""
        hostname = urlparse(value).hostname
        if hostname in {"0.0.0.0", "::"}:
            raise ValueError(
                "0.0.0.0/:: solo sirve para que un servidor escuche; "
                "usa 127.0.0.1, localhost o la IP real del servidor"
            )
        return value.rstrip("/")

    @field_validator("model_context_size")
    @classmethod
    def validate_initial_model_context(cls, value: int) -> int:
        if value not in {8192, 16384}:
            raise ValueError("MODEL_CONTEXT_SIZE debe ser 8192 o 16384 en esta fase")
        return value

    @property
    def effective_retrieval_top_k(self) -> int:
        return self.retrieval_top_k or self.top_k

    @property
    def data_path(self) -> Path:
        """Convierte `data_dir` a Path para operaciones con rutas."""
        return Path(self.data_dir)

    @property
    def lancedb_path(self) -> Path:
        """Convierte `lancedb_uri` a Path para operaciones con rutas."""
        return Path(self.lancedb_uri)

    @property
    def documents_path(self) -> Path:
        """Convierte `documents_dir` a Path para operaciones con rutas."""
        return Path(self.documents_dir)

    @property
    def models_path(self) -> Path:
        """Directorio local que contiene exclusivamente artefactos de modelos."""
        return Path(self.models_dir)

    @property
    def model_logs_path(self) -> Path:
        """Directorio de logs de los procesos creados por el supervisor."""
        return Path(self.model_logs_dir)

    @property
    def allowed_local_model_hosts(self) -> set[str]:
        """Hosts locales adicionales permitidos, separados por comas."""
        return {
            host.strip().lower()
            for host in self.local_model_hosts.split(",")
            if host.strip()
        }
    
    @property
    def bm25_index_path(self) -> Path:
        """Ruta del índice/corpus BM25 persistido en disco."""
        safe_table_name = self.lancedb_table.replace("/", "_").replace("\\", "_")
        return Path(self.bm25_index_dir) / f"{safe_table_name}.json"


def get_settings() -> Settings:
    """Factory para obtener la configuración validada del proyecto."""
    return Settings()
