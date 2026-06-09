"""API HTTP para exponer el pipeline RAG existente.

La idea es mantener el CLI actual y reutilizar exactamente la misma lógica del
pipeline detrás de una aplicación FastAPI.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any
from urllib.parse import quote
from uuid import uuid4

import json
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from rag_cliente.config import get_settings
from rag_cliente.pipeline import RagPipeline
from rag_cliente.pdf_loader import IMAGE_SUFFIXES


DOCUMENT_SUFFIXES = {".pdf", ".docx", ".txt", *IMAGE_SUFFIXES}


class ChatMessage(BaseModel):
    """Mensaje conversacional enviado por el frontend."""

    role: str = Field(..., min_length=1)
    content: str = Field(..., min_length=1)


class IndexRequest(BaseModel):
    """Payload para indexar documentos desde un directorio local."""

    doc_dir: str = Field(..., description="Directory containing PDF, DOCX, and TXT files.")
    tag: str | None = Field(default=None, description="Optional metadata tag assigned to indexed chunks.")


class IndexResponse(BaseModel):
    """Resumen de una ejecución de indexado."""

    indexed_chunks: int
    lancedb_table: str
    doc_dir: str
    tag: str | None = None


class AskRequest(BaseModel):
    """Payload para consultas RAG."""

    question: str = Field(..., min_length=1)
    top_k: int | None = Field(default=None, ge=1)
    tag: str | None = Field(default=None)
    tags: list[str] = Field(default_factory=list)
    session_id: str | None = Field(default=None, min_length=1)
    messages: list[ChatMessage] = Field(default_factory=list)


class Citation(BaseModel):
    """Metadatos de trazabilidad devueltos al cliente."""

    source_id: str | None = None
    document_id: str
    source: str
    source_path: str
    source_type: str
    page_start: int
    page_end: int
    chunk_index: int
    ocr_used: bool = False
    tag: str | None = None


class AskResponse(BaseModel):
    """Respuesta no streaming de una consulta."""

    session_id: str
    answer: str
    reasoning: str
    citations: list[Citation]
    matches: list[dict[str, Any]]


class SessionResponse(BaseModel):
    """Identificador de sesion conversacional."""

    session_id: str


class StoredFileMetadata(BaseModel):
    """Metadatos de un fichero disponible para el frontend."""

    name: str
    relative_path: str
    tag: str | None = None
    size_bytes: int
    modified_at: str
    extension: str
    download_url: str


class FileListResponse(BaseModel):
    """Respuesta con los ficheros disponibles en la carpeta configurada."""

    directory: str
    files: list[StoredFileMetadata]


class UploadFileResponse(BaseModel):
    """Resumen del fichero guardado desde multipart/form-data."""

    filename: str
    size_bytes: int
    content_type: str | None = None
    path: str
    tag: str | None = None


class SessionStore:
    """Almacenamiento en memoria para historiales conversacionales."""

    def __init__(self) -> None:
        self._sessions: dict[str, list[dict[str, str]]] = {}
        self._lock = Lock()

    def create_session(self) -> str:
        """Crea una sesion vacia y devuelve su identificador."""
        session_id = uuid4().hex
        with self._lock:
            self._sessions.setdefault(session_id, [])
        return session_id

    def resolve_session(self, session_id: str | None) -> str:
        """Devuelve una sesion existente o crea una nueva si no se indica ninguna."""
        if session_id is None:
            return self.create_session()
        with self._lock:
            if session_id not in self._sessions:
                raise KeyError(session_id)
        return session_id

    def get_messages(self, session_id: str) -> list[dict[str, str]]:
        """Obtiene una copia del historial asociado a la sesion."""
        with self._lock:
            return [message.copy() for message in self._sessions.get(session_id, [])]

    def append_messages(self, session_id: str, messages: list[dict[str, str]]) -> None:
        """Añade mensajes al historial de la sesion."""
        if not messages:
            return
        with self._lock:
            history = self._sessions.setdefault(session_id, [])
            history.extend(message.copy() for message in messages)

    def delete_session(self, session_id: str) -> bool:
        """Elimina una sesion si existe."""
        with self._lock:
            return self._sessions.pop(session_id, None) is not None


def create_app() -> FastAPI:
    """Construye la aplicación FastAPI con el pipeline compartido."""
    settings = get_settings()
    settings.lancedb_path.mkdir(parents=True, exist_ok=True)
    settings.documents_path.mkdir(parents=True, exist_ok=True)
    pipeline = RagPipeline(settings)
    session_store = SessionStore()

    app = FastAPI(
        title="RAG Cliente API",
        version="0.1.0",
        description="HTTP API for indexing local documents and querying the existing RAG pipeline.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api_cors_allow_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.settings = settings
    app.state.pipeline = pipeline
    app.state.session_store = session_store

    def get_documents_dir() -> Path:
        return app.state.settings.documents_path.resolve()

    def build_file_metadata(path: Path) -> StoredFileMetadata:
        stat = path.stat()
        documents_dir = get_documents_dir()
        relative_path = path.resolve().relative_to(documents_dir).as_posix()
        tag = relative_path.split("/", 1)[0] if "/" in relative_path else None
        download_path = "/".join(quote(part) for part in relative_path.split("/"))
        return StoredFileMetadata(
            name=path.name,
            relative_path=relative_path,
            tag=tag,
            size_bytes=stat.st_size,
            modified_at=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
            extension=path.suffix.lower(),
            download_url=f"/files/{download_path}",
        )

    def normalize_tag_name(tag: str | None) -> str | None:
        normalized = (tag or "").strip()
        if not normalized:
            return None
        if normalized in {".", ".."} or "/" in normalized or "\\" in normalized:
            raise HTTPException(status_code=400, detail="Invalid tag.")
        return normalized

    def sanitize_upload_filename(filename: str | None) -> str:
        sanitized_name = Path(filename or "").name
        if not sanitized_name or sanitized_name in {".", ".."}:
            raise HTTPException(status_code=400, detail="A valid filename is required.")
        return sanitized_name

    def resolve_document_path(relative_file_path: str) -> Path:
        relative_path = Path(relative_file_path)
        if (
            not relative_file_path.strip()
            or relative_path.is_absolute()
            or any(part in {"", ".", ".."} for part in relative_path.parts)
        ):
            raise HTTPException(status_code=400, detail="A valid file path is required.")
        candidate = (get_documents_dir() / relative_path).resolve()
        try:
            candidate.relative_to(get_documents_dir())
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Invalid file path.") from exc
        return candidate

    def build_upload_destination(filename: str, tag: str | None) -> Path:
        destination_dir = get_documents_dir()
        normalized_tag = normalize_tag_name(tag)
        if normalized_tag:
            destination_dir = destination_dir / normalized_tag
        destination_dir.mkdir(parents=True, exist_ok=True)
        return (destination_dir / filename).resolve()

    def resolve_tag_filter(payload: AskRequest) -> str | None:
        tag = (payload.tag or "").strip()
        if tag:
            return tag
        for candidate in payload.tags:
            normalized_candidate = candidate.strip()
            if normalized_candidate:
                return normalized_candidate
        return None

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/files", response_model=FileListResponse)
    def list_files() -> FileListResponse:
        documents_dir = get_documents_dir()
        files = [
            build_file_metadata(path)
            for path in sorted(documents_dir.rglob("*"))
            if path.is_file() and path.suffix.lower() in DOCUMENT_SUFFIXES
        ]
        return FileListResponse(directory=str(documents_dir), files=files)

    @app.get("/files/{file_path:path}")
    def download_file(file_path: str) -> FileResponse:
        file_path = resolve_document_path(file_path)
        if not file_path.exists() or not file_path.is_file():
            raise HTTPException(status_code=404, detail="File was not found.")
        if file_path.suffix.lower() not in DOCUMENT_SUFFIXES:
            raise HTTPException(status_code=400, detail="File type is not supported.")
        return FileResponse(path=file_path, filename=file_path.name)

    @app.post("/files/upload", response_model=UploadFileResponse, status_code=201)
    async def upload_file(
        file: UploadFile = File(...),
        overwrite: bool = Form(default=False),
        tag: str | None = Form(default=None),
    ) -> UploadFileResponse:
        filename = sanitize_upload_filename(file.filename)
        normalized_tag = normalize_tag_name(tag)
        if Path(filename).suffix.lower() not in DOCUMENT_SUFFIXES:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type. Allowed extensions: {', '.join(sorted(DOCUMENT_SUFFIXES))}.",
            )

        destination = build_upload_destination(filename, normalized_tag)
        if destination.exists() and not overwrite:
            raise HTTPException(
                status_code=409,
                detail=f"File '{filename}' already exists. Use overwrite=true to replace it.",
            )

        try:
            content = await file.read()
        finally:
            await file.close()
        destination.write_bytes(content)

        return UploadFileResponse(
            filename=destination.name,
            size_bytes=len(content),
            content_type=file.content_type,
            path=str(destination),
            tag=normalized_tag,
        )

    @app.post("/sessions", response_model=SessionResponse)
    def create_session() -> SessionResponse:
        return SessionResponse(session_id=app.state.session_store.create_session())

    @app.delete("/sessions/{session_id}", status_code=204)
    def delete_session(session_id: str) -> None:
        if not app.state.session_store.delete_session(session_id):
            raise HTTPException(status_code=404, detail=f"Session '{session_id}' was not found.")

    @app.post("/index", response_model=IndexResponse)
    def index_documents(payload: IndexRequest) -> IndexResponse:
        doc_dir = Path(payload.doc_dir)
        tag = (payload.tag or "").strip() or None
        try:
            indexed_chunks = app.state.pipeline.index_documents(doc_dir, tag=tag)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        return IndexResponse(
            indexed_chunks=indexed_chunks,
            lancedb_table=app.state.settings.lancedb_table,
            doc_dir=str(doc_dir),
            tag=tag,
        )

    @app.post("/ask", response_model=AskResponse)
    def ask(payload: AskRequest) -> AskResponse:
        try:
            session_id = app.state.session_store.resolve_session(payload.session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Session '{payload.session_id}' was not found.") from exc
        request_messages = [message.model_dump() for message in payload.messages]
        history_messages = app.state.session_store.get_messages(session_id)
        combined_messages = [*history_messages, *request_messages]
        tag = resolve_tag_filter(payload)
        try:
            result = app.state.pipeline.ask(
                payload.question,
                top_k=payload.top_k,
                messages=combined_messages,
                tag=tag,
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        app.state.session_store.append_messages(
            session_id,
            [
                *request_messages,
                {"role": "user", "content": payload.question},
                {"role": "assistant", "content": result["answer"]},
            ],
        )

        return AskResponse(session_id=session_id, **result)

    @app.post("/ask/stream")
    def ask_stream(payload: AskRequest) -> StreamingResponse:
        try:
            session_id = app.state.session_store.resolve_session(payload.session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Session '{payload.session_id}' was not found.") from exc
        request_messages = [message.model_dump() for message in payload.messages]
        history_messages = app.state.session_store.get_messages(session_id)
        combined_messages = [*history_messages, *request_messages]
        tag = resolve_tag_filter(payload)
        try:
            result = app.state.pipeline.stream_answer(
                payload.question,
                top_k=payload.top_k,
                messages=combined_messages,
                tag=tag,
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        def serialize_event(event: dict[str, Any]) -> str:
            return json.dumps(event, ensure_ascii=False) + "\n"

        def event_stream():
            emitted_answer = False
            emitted_reasoning = False
            collected_answer_parts: list[str] = []

            yield serialize_event({"type": "session", "session_id": session_id})
            for event in result["answer_stream"]:
                if event["type"] == "answer":
                    emitted_answer = True
                    collected_answer_parts.append(event["delta"])
                elif event["type"] == "reasoning":
                    emitted_reasoning = True
                yield serialize_event(event)

            if not emitted_answer and not emitted_reasoning:
                fallback_response = result["fallback_response"]()
                if fallback_response["reasoning"]:
                    yield serialize_event({"type": "reasoning", "delta": fallback_response["reasoning"]})
                if fallback_response["answer"]:
                    collected_answer_parts.append(fallback_response["answer"])
                    yield serialize_event({"type": "answer", "delta": fallback_response["answer"]})

            app.state.session_store.append_messages(
                session_id,
                [
                    *request_messages,
                    {"role": "user", "content": payload.question},
                    {"role": "assistant", "content": "".join(collected_answer_parts)},
                ],
            )

            answer_text = "".join(collected_answer_parts)
            citations = result["resolve_citations"](answer_text)

            yield serialize_event({"type": "citations", "citations": citations})
            yield serialize_event({"type": "matches", "matches": result["matches"]})
            yield serialize_event({"type": "done"})

        return StreamingResponse(
            event_stream(),
            media_type="application/x-ndjson; charset=utf-8",
            headers={"X-Session-Id": session_id},
        )

    return app


app = create_app()


def run() -> None:
    """Arranca Uvicorn para exponer la API localmente."""
    uvicorn.run("rag_cliente.api:app", host="0.0.0.0", port=8000, reload=False)
