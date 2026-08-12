"""Configuración del RAG híbrido Bedrock + modelos locales."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from dotenv import load_dotenv
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()

LocalModelProfile = Literal["cpu", "gpu", "auto"]
ModelServerMode = Literal["managed", "external"]

CLAUDE_SONNET_4_6_GLOBAL_MODEL_ID = "global.anthropic.claude-sonnet-4-6"


def has_usable_nvidia_gpu() -> bool:
    """Detecta una NVIDIA mediante nvidia-smi sin cargar librerias de modelos."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0 and bool(result.stdout.strip())


def resolve_local_model_profile(
    settings: "Settings",
    *,
    gpu_available: bool | None = None,
) -> Literal["cpu", "gpu"]:
    if settings.local_model_profile != "auto":
        return settings.local_model_profile
    available = has_usable_nvidia_gpu() if gpu_available is None else gpu_available
    return "gpu" if available else "cpu"


class Settings(BaseSettings):
    """Configuración tipada cargada desde variables de entorno."""

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

    model_supervision_enabled: bool = Field(
        default=False,
        alias="MODEL_SUPERVISION_ENABLED",
    )
    local_model_profile: LocalModelProfile = Field(
        default="auto",
        alias="LOCAL_MODEL_PROFILE",
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
    index_job_timeout: float = Field(default=1800.0, alias="INDEX_JOB_TIMEOUT", gt=0)
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
    model_context_size: int = Field(default=16384, alias="MODEL_CONTEXT_SIZE")
    model_embeddings_mode: ModelServerMode = Field(
        default="managed",
        alias="MODEL_EMBEDDINGS_MODE",
    )
    model_chat_mode: ModelServerMode = Field(default="managed", alias="MODEL_CHAT_MODE")
    local_model_hosts: str = Field(default="", alias="LOCAL_MODEL_HOSTS")
    embeddings_cpu_gguf_path: str = Field(
        default="",
        alias="EMBEDDINGS_CPU_GGUF_PATH",
    )
    embeddings_gpu_gguf_path: str = Field(
        default="",
        alias="EMBEDDINGS_GPU_GGUF_PATH",
    )
    # Compatibilidad con configuraciones anteriores que compartían el mismo
    # modelo de embeddings para CPU y GPU.
    embeddings_gguf_path: str = Field(default="", alias="EMBEDDINGS_GGUF_PATH")
    chat_cpu_gguf_path: str = Field(default="", alias="CHAT_CPU_GGUF_PATH")
    chat_gpu_gguf_path: str = Field(default="", alias="CHAT_GPU_GGUF_PATH")

    # Bedrock permanece desactivado hasta que el usuario configure AWS.
    bedrock_enabled: bool = Field(default=False, alias="BEDROCK_ENABLED")
    aws_profile: str = Field(default="", alias="AWS_PROFILE")
    aws_region: str = Field(default="", alias="AWS_REGION")
    bedrock_model_id: str = Field(
        default=CLAUDE_SONNET_4_6_GLOBAL_MODEL_ID,
        alias="BEDROCK_MODEL_ID",
    )
    bedrock_context_pages: int = Field(
        default=4,
        alias="BEDROCK_CONTEXT_PAGES",
    )
    bedrock_render_dpi: int = Field(
        default=144,
        alias="BEDROCK_RENDER_DPI",
        ge=72,
        le=300,
    )
    bedrock_max_output_tokens: int = Field(
        default=16384,
        alias="BEDROCK_MAX_OUTPUT_TOKENS",
        ge=256,
        le=64000,
    )
    bedrock_max_pages_per_document: int = Field(
        default=200,
        alias="BEDROCK_MAX_PAGES_PER_DOCUMENT",
        ge=1,
    )
    bedrock_max_calls_per_document: int = Field(
        default=200,
        alias="BEDROCK_MAX_CALLS_PER_DOCUMENT",
        ge=1,
    )
    bedrock_request_timeout: float = Field(
        default=300.0,
        alias="BEDROCK_REQUEST_TIMEOUT",
        gt=0,
    )
    bedrock_max_retries: int = Field(
        default=1,
        alias="BEDROCK_MAX_RETRIES",
        ge=0,
        le=1,
    )
    bedrock_transient_max_retries: int = Field(
        default=5,
        alias="BEDROCK_TRANSIENT_MAX_RETRIES",
        ge=0,
        le=5,
    )
    bedrock_reference_text_max_chars: int = Field(
        default=12000,
        alias="BEDROCK_REFERENCE_TEXT_MAX_CHARS",
        ge=0,
    )
    bedrock_prompt_version: str = Field(
        default="claude-sonnet-4-6-target-third-quoted-v5",
        alias="BEDROCK_PROMPT_VERSION",
    )
    bedrock_cache_dir: str = Field(
        default="./data/markdown",
        alias="BEDROCK_CACHE_DIR",
    )

    lancedb_uri: str = Field(default="./data/lancedb", alias="LANCEDB_URI")
    lancedb_table: str = Field(default="pdf_chunks", alias="LANCEDB_TABLE")
    top_k: int = Field(default=8, alias="TOP_K", ge=1)
    retrieval_top_k: int | None = Field(default=None, alias="RETRIEVAL_TOP_K", ge=1)
    retrieval_mode: Literal["vector", "bm25", "hybrid"] = Field(
        default="hybrid",
        alias="RETRIEVAL_MODE",
    )
    bm25_index_dir: str = Field(default="./data/bm25", alias="BM25_INDEX_DIR")
    rrf_k: int = Field(default=60, alias="RRF_K", ge=1)
    vector_candidates: int = Field(default=40, alias="VECTOR_CANDIDATES", ge=1)
    bm25_candidates: int = Field(default=40, alias="BM25_CANDIDATES", ge=1)

    chunk_target_tokens: int = Field(default=700, alias="CHUNK_TARGET_TOKENS", ge=1)
    chunk_max_tokens: int = Field(default=900, alias="CHUNK_MAX_TOKENS", ge=1)
    chunk_overlap_tokens: int = Field(default=100, alias="CHUNK_OVERLAP_TOKENS", ge=0)

    max_tokens: int = Field(default=1024, alias="MAX_TOKENS", ge=1)
    reasoning_max_tokens: int = Field(default=1024, alias="REASONING_MAX_TOKENS", ge=1)
    embedding_batch_size: int = Field(default=4, alias="EMBEDDING_BATCH_SIZE", ge=1)
    embedding_query_instruction: str = Field(
        default=(
            "Given a user question in Spanish, retrieve relevant passages from "
            "the indexed documents that answer the question"
        ),
        alias="EMBEDDING_QUERY_INSTRUCTION",
    )

    data_dir: str = Field(default="./data", alias="DATA_DIR")
    documents_dir: str = Field(default="./data/pdfs", alias="DOCUMENTS_DIR")
    evaluation_db: str = Field(
        default="./data/evaluation.sqlite",
        alias="EVALUATION_DB",
    )
    api_cors_allow_origins: list[str] = Field(
        default=["*"],
        alias="API_CORS_ALLOW_ORIGINS",
    )

    model_config = SettingsConfigDict(populate_by_name=True, extra="ignore")

    @field_validator("llama_cpp_chat_base_url", "llama_cpp_embedding_base_url")
    @classmethod
    def reject_server_bind_addresses(cls, value: str) -> str:
        hostname = urlparse(value).hostname
        if hostname in {"0.0.0.0", "::"}:
            raise ValueError(
                "0.0.0.0/:: solo sirve para escuchar; usa 127.0.0.1, localhost "
                "o la IP real del servidor"
            )
        return value.rstrip("/")

    @field_validator("model_context_size")
    @classmethod
    def validate_model_context(cls, value: int) -> int:
        if value not in {8192, 16384}:
            raise ValueError("MODEL_CONTEXT_SIZE debe ser 8192 o 16384")
        return value

    @field_validator("bedrock_context_pages")
    @classmethod
    def validate_bedrock_context_size(cls, value: int) -> int:
        if value != 4:
            raise ValueError("BEDROCK_CONTEXT_PAGES debe ser exactamente 4")
        return value

    @property
    def effective_retrieval_top_k(self) -> int:
        return self.retrieval_top_k or self.top_k

    @property
    def lancedb_path(self) -> Path:
        return Path(self.lancedb_uri)

    @property
    def documents_path(self) -> Path:
        return Path(self.documents_dir)

    @property
    def evaluation_db_path(self) -> Path:
        return Path(self.evaluation_db)

    @property
    def models_path(self) -> Path:
        return Path(self.models_dir)

    @property
    def model_logs_path(self) -> Path:
        return Path(self.model_logs_dir)

    @property
    def bedrock_cache_path(self) -> Path:
        return Path(self.bedrock_cache_dir)

    @property
    def allowed_local_model_hosts(self) -> set[str]:
        return {
            host.strip().lower()
            for host in self.local_model_hosts.split(",")
            if host.strip()
        }

    @property
    def bm25_index_path(self) -> Path:
        safe_table = self.lancedb_table.replace("/", "_").replace("\\", "_")
        return Path(self.bm25_index_dir) / f"{safe_table}.json"


def get_settings() -> Settings:
    return Settings()
