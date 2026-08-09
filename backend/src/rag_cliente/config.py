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

from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from dotenv import load_dotenv
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Carga variables de entorno desde ".env" antes de construir Settings.
load_dotenv()


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
    # BASE DE DATOS VECTORIAL - LANCEDB
    # ===================================================================
    lancedb_uri: str = Field(default="./data/lancedb", alias="LANCEDB_URI")
    lancedb_table: str = Field(default="pdf_chunks", alias="LANCEDB_TABLE")

    # ===================================================================
    # BÚSQUEDA Y RECUPERACIÓN
    # ===================================================================
    top_k: int = Field(default=2, alias="TOP_K")

    # ===================================================================
    # BÚSQUEDA HÍBRIDA - VECTORIAL + BM25
    # ===================================================================
    hybrid_search_enabled: bool = Field(default=True, alias="HYBRID_SEARCH_ENABLED")
    vector_weight: float = Field(default=0.65, alias="VECTOR_WEIGHT")
    bm25_weight: float = Field(default=0.35, alias="BM25_WEIGHT")
    bm25_top_k_multiplier: int = Field(default=3, alias="BM25_TOP_K_MULTIPLIER")
    bm25_index_dir: str = Field(default="./data/bm25", alias="BM25_INDEX_DIR")
    bm25_min_raw_score: float = Field(default=0.25, alias="BM25_MIN_RAW_SCORE")

    # ===================================================================
    # FRAGMENTACIÓN DE DOCUMENTOS
    # ===================================================================
    chunk_size: int = Field(default=700, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=100, alias="CHUNK_OVERLAP")

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
    # MARKER - PARSER/OCR PRINCIPAL PARA PDFS E IMÁGENES
    # ===================================================================
    # Activa Marker para PDFs e imágenes. Si se desactiva, los PDFs digitales
    # usan fallback nativo PyMuPDF y las imágenes no se indexan.
    marker_enabled: bool = Field(default=True, alias="MARKER_ENABLED")

    # Fuerza OCR visual incluso si el PDF contiene texto digital. Es útil si el
    # texto embebido está corrupto o si quieres mejor preservación de tablas/math.
    marker_force_ocr: bool = Field(default=False, alias="MARKER_FORCE_OCR")

    # Elimina texto OCR existente en el PDF antes de re-OCR. Útil para documentos
    # con capa OCR mala o duplicada.
    marker_strip_existing_ocr: bool = Field(default=False, alias="MARKER_STRIP_EXISTING_OCR")

    # Usa modo LLM de Marker para mejorar tablas, formularios y formato. Requiere
    # configurar el backend LLM que soporte Marker fuera de este proyecto.
    marker_use_llm: bool = Field(default=False, alias="MARKER_USE_LLM")

    # Evita guardar imágenes extraídas al disco. Para RAG textual normalmente se
    # prefiere True para no generar artefactos innecesarios.
    marker_disable_image_extraction: bool = Field(
        default=True,
        alias="MARKER_DISABLE_IMAGE_EXTRACTION",
    )

    # Rango opcional de páginas en sintaxis Marker, por ejemplo "0,5-10,20".
    # Vacío = documento completo.
    marker_page_range: str = Field(default="", alias="MARKER_PAGE_RANGE")

    # El proyecto se despliega con GPU NVIDIA. Marker se fuerza a CUDA y el
    # loader valida que PyTorch pueda verla antes de cargar los modelos.
    marker_torch_device: Literal["cuda"] = Field(
        default="cuda",
        alias="MARKER_TORCH_DEVICE",
    )


    model_config = SettingsConfigDict(
        populate_by_name=True,
        extra="ignore",
    )

    @field_validator("llama_cpp_chat_base_url", "llama_cpp_embedding_base_url")
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
    def bm25_index_path(self) -> Path:
        """Ruta del índice/corpus BM25 persistido en disco."""
        safe_table_name = self.lancedb_table.replace("/", "_").replace("\\", "_")
        return Path(self.bm25_index_dir) / f"{safe_table_name}.json"


def get_settings() -> Settings:
    """Factory para obtener la configuración validada del proyecto."""
    return Settings()
