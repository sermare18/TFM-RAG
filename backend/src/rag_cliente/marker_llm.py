"""Servicio OpenAI-compatible local de Marker con presupuestos estrictos."""

from __future__ import annotations

import json
import base64
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Callable, Iterator

import openai
import httpx

APIConnectionError = getattr(openai, "APIConnectionError", type("APIConnectionError", (Exception,), {}))
APITimeoutError = getattr(openai, "APITimeoutError", type("APITimeoutError", (Exception,), {}))
RateLimitError = getattr(openai, "RateLimitError", type("RateLimitError", (Exception,), {}))

from rag_cliente.config import Settings
from rag_cliente.local_endpoints import is_local_model_endpoint


class MarkerLLMError(RuntimeError):
    """Error serializable de la ruta LLM local de Marker."""

    def __init__(self, code: str, message: str, **details: Any) -> None:
        self.code = code
        self.message = message
        self.details = details
        super().__init__(json.dumps(self.as_dict(), ensure_ascii=False, sort_keys=True))

    def as_dict(self) -> dict[str, Any]:
        return {
            "error": {
                "code": self.code,
                "message": self.message,
                "details": self.details,
            }
        }


class LLMBudgetExceededError(MarkerLLMError):
    def __init__(self, budget: str, limit: int | float, current: int | float) -> None:
        super().__init__(
            "llm_budget_exceeded",
            f"Se excedió el presupuesto LLM '{budget}'",
            budget=budget,
            limit=limit,
            current=current,
        )


def validate_marker_local_only(settings: Settings) -> str:
    """Falla antes de crear Marker si el VLM no es inequívocamente local."""
    endpoint = settings.marker_openai_base_url.strip().rstrip("/")
    if not endpoint:
        raise MarkerLLMError(
            "local_llm_endpoint_required",
            "use_llm=true requiere MARKER_OPENAI_BASE_URL",
        )
    if not is_local_model_endpoint(endpoint, settings.allowed_local_model_hosts):
        raise MarkerLLMError(
            "external_llm_endpoint_rejected",
            "Marker solo puede usar un endpoint OpenAI-compatible local",
            endpoint=endpoint,
        )
    surya_endpoint = settings.surya_base_url.strip().rstrip("/")
    if not is_local_model_endpoint(
        surya_endpoint,
        settings.allowed_local_model_hosts,
    ):
        raise MarkerLLMError(
            "external_llm_endpoint_rejected",
            "Surya solo puede usar un endpoint OpenAI-compatible local",
            endpoint=surya_endpoint,
        )
    if settings.marker_llm_fallback_to_base:
        raise MarkerLLMError(
            "external_llm_fallback_rejected",
            "MARKER_LLM_FALLBACK_TO_BASE debe permanecer false",
        )
    return endpoint


@dataclass(slots=True)
class MarkerBudgetState:
    document_id: str
    started_at: float
    request_count: int = 0
    completion_tokens: int = 0


class BudgetedMarkerOpenAIService:
    """Wrapper local que conserva mensajes/esquema y limita toda repetición."""

    def __init__(
        self,
        settings: Settings,
        *,
        client_factory: Callable[[], Any] | None = None,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        endpoint = validate_marker_local_only(settings)
        self.settings = settings
        self._client_factory = client_factory
        self._monotonic = monotonic
        self._budget_lock = threading.RLock()
        self._state: MarkerBudgetState | None = None
        self.last_report: dict[str, Any] | None = None

        # La clave es una constante local sin valor externo. No se consulta
        # OPENAI_API_KEY ni ninguna credencial de proveedor.
        self.openai_base_url = endpoint
        self.openai_model = settings.marker_openai_model
        self.openai_api_key = "local-only"
        self.openai_image_format = "png"
        self.timeout = int(settings.marker_llm_request_timeout)
        self.max_retries = settings.marker_llm_max_retries
        self.max_output_tokens = settings.marker_llm_max_tokens_per_request

    def get_client(self):
        if self._client_factory is not None:
            return self._client_factory()
        return openai.OpenAI(
            api_key=self.openai_api_key,
            base_url=self.openai_base_url,
            timeout=self.settings.marker_llm_request_timeout,
            # El wrapper controla el único reintento; el SDK no añade otros.
            max_retries=0,
        )

    def process_images(self, images: Any) -> list[dict]:
        if not isinstance(images, list):
            images = [images]
        encoded = []
        for image in images:
            image_bytes = BytesIO()
            image.save(image_bytes, format=self.openai_image_format)
            payload = base64.b64encode(image_bytes.getvalue()).decode("utf-8")
            encoded.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/{self.openai_image_format};base64,{payload}",
                    },
                }
            )
        return encoded

    def format_image_for_llm(self, image: Any) -> list[dict]:
        if not image:
            return []
        return self.process_images(image)

    @contextmanager
    def document_budget(self, document_id: str) -> Iterator["BudgetedMarkerOpenAIService"]:
        with self._budget_lock:
            if self._state is not None:
                raise RuntimeError("Ya existe un presupuesto Marker activo")
            self._state = MarkerBudgetState(
                document_id=document_id,
                started_at=self._monotonic(),
            )
        try:
            yield self
        finally:
            with self._budget_lock:
                state = self._state
                if state is not None:
                    self.last_report = {
                        "document_id": state.document_id,
                        "request_count": state.request_count,
                        "completion_tokens": state.completion_tokens,
                        "elapsed_seconds": max(0.0, self._monotonic() - state.started_at),
                    }
                self._state = None

    def _require_state(self) -> MarkerBudgetState:
        if self._state is None:
            self._state = MarkerBudgetState(
                document_id="unscoped",
                started_at=self._monotonic(),
            )
        return self._state

    def _remaining_job_time(self, state: MarkerBudgetState) -> float:
        elapsed = self._monotonic() - state.started_at
        remaining = self.settings.marker_llm_job_timeout - elapsed
        if remaining <= 0:
            raise LLMBudgetExceededError(
                "job_timeout_seconds",
                self.settings.marker_llm_job_timeout,
                max(0.0, elapsed),
            )
        return remaining

    def _reserve_request(self, state: MarkerBudgetState) -> None:
        if state.request_count >= self.settings.marker_llm_max_requests:
            raise LLMBudgetExceededError(
                "requests",
                self.settings.marker_llm_max_requests,
                state.request_count + 1,
            )
        state.request_count += 1

    def _record_completion_tokens(self, state: MarkerBudgetState, response: Any) -> None:
        usage = getattr(response, "usage", None)
        completion_tokens = getattr(usage, "completion_tokens", None)
        if isinstance(completion_tokens, int) and completion_tokens > 0:
            state.completion_tokens += completion_tokens
        if state.completion_tokens > self.settings.marker_llm_max_generated_tokens_per_document:
            raise LLMBudgetExceededError(
                "generated_tokens_per_document",
                self.settings.marker_llm_max_generated_tokens_per_document,
                state.completion_tokens,
            )

    def __call__(
        self,
        prompt: str,
        image: Any,
        block: Any,
        response_schema: type,
        max_retries: int | None = None,
        timeout: int | None = None,
    ):
        # Un único lock implica una sola petición VLM simultánea.
        with self._budget_lock:
            state = self._require_state()
            requested_retries = self.max_retries if max_retries is None else max_retries
            retries = min(max(0, int(requested_retries)), self.settings.marker_llm_max_retries, 1)

            image_data = self.format_image_for_llm(image)
            messages = [
                {
                    "role": "user",
                    "content": [
                        *image_data,
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            client = self.get_client()
            attempts = retries + 1

            for attempt in range(1, attempts + 1):
                remaining = self._remaining_job_time(state)
                self._reserve_request(state)
                request_timeout_seconds = min(
                    self.settings.marker_llm_request_timeout,
                    float(timeout) if timeout is not None else self.settings.marker_llm_request_timeout,
                    remaining,
                )
                request_timeout = httpx.Timeout(
                    connect=min(
                        self.settings.model_health_connect_timeout,
                        request_timeout_seconds,
                    ),
                    read=request_timeout_seconds,
                    write=request_timeout_seconds,
                    pool=request_timeout_seconds,
                )
                try:
                    response = client.chat.completions.parse(
                        extra_headers={
                            "X-Title": "Marker",
                            "HTTP-Referer": "https://github.com/datalab-to/marker",
                        },
                        model=self.openai_model,
                        messages=messages,
                        timeout=request_timeout,
                        response_format=response_schema,
                        max_tokens=self.settings.marker_llm_max_tokens_per_request,
                    )
                    self._record_completion_tokens(state, response)
                    self._remaining_job_time(state)

                    response_text = response.choices[0].message.content
                    payload = json.loads(response_text)
                    usage = getattr(response, "usage", None)
                    total_tokens = getattr(usage, "total_tokens", None)
                    if block is not None and isinstance(total_tokens, int):
                        block.update_metadata(
                            llm_tokens_used=total_tokens,
                            llm_request_count=1,
                        )
                    return payload
                except LLMBudgetExceededError:
                    raise
                except (APITimeoutError, APIConnectionError, RateLimitError) as exc:
                    if attempt < attempts:
                        continue
                    raise MarkerLLMError(
                        "local_llm_request_failed",
                        "Falló la petición al VLM local tras el reintento permitido",
                        attempts=attempt,
                        reason=type(exc).__name__,
                    ) from exc
                except (json.JSONDecodeError, TypeError, ValueError, KeyError, IndexError) as exc:
                    # Nunca se entra en un ciclo para reparar JSON.
                    raise MarkerLLMError(
                        "local_llm_invalid_response",
                        "El VLM local devolvió una respuesta estructurada inválida",
                        attempts=attempt,
                        reason=type(exc).__name__,
                    ) from exc
                except Exception as exc:
                    raise MarkerLLMError(
                        "local_llm_request_failed",
                        "Falló la petición al VLM local",
                        attempts=attempt,
                        reason=type(exc).__name__,
                    ) from exc

            raise AssertionError("Bucle de solicitudes Marker inalcanzable")
