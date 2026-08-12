"""Cliente de chat y embeddings para endpoints compatibles con OpenAI."""

from __future__ import annotations

import json
import re
from typing import Any, Callable

import httpx
from openai import OpenAI

ProgressCallback = Callable[[str], None]

QUERY_AUGMENTATION_PROMPT_VERSION = "query-augmentation-v1"

from rag_cliente.config import Settings
from rag_cliente.local_endpoints import is_local_model_endpoint


class LlamaCppClient:
    @staticmethod
    def _thinking_extra_body(enable_reasoning: bool) -> dict[str, Any]:
        """Activa o desactiva el razonamiento de modelos Qwen híbridos."""
        return {"chat_template_kwargs": {"enable_thinking": enable_reasoning}}

    def _generation_max_tokens(self, enable_reasoning: bool) -> int:
        if enable_reasoning:
            return self.settings.reasoning_max_tokens
        return self.settings.max_tokens

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        for role, endpoint in (
            ("chat", settings.llama_cpp_chat_base_url),
            ("embeddings", settings.llama_cpp_embedding_base_url),
        ):
            if not is_local_model_endpoint(
                endpoint,
                settings.allowed_local_model_hosts,
            ):
                raise ValueError(
                    f"El endpoint de {role} debe ser local; recibido: {endpoint}"
                )
        request_timeout = httpx.Timeout(
            connect=min(
                settings.model_health_connect_timeout,
                settings.model_request_timeout,
            ),
            read=settings.model_request_timeout,
            write=settings.model_request_timeout,
            pool=settings.model_request_timeout,
        )

        self.chat_client = OpenAI(
            base_url=settings.llama_cpp_chat_base_url,
            api_key=settings.openai_api_key,
            timeout=request_timeout,
            max_retries=settings.model_max_retries,
        )
        self.embedding_client = OpenAI(
            base_url=settings.llama_cpp_embedding_base_url,
            api_key=settings.openai_api_key,
            timeout=request_timeout,
            max_retries=settings.model_max_retries,
        )

    def embed_texts(
        self,
        texts: list[str],
        progress_callback: ProgressCallback | None = None,
        query_mode: bool = False,
        use_query_instruction: bool | None = None,
    ) -> list[list[float]]:
        """Genera embeddings para una lista de textos en lotes pequeños."""
        if not texts:
            return []

        prepared_texts = texts
        instruction = self.settings.embedding_query_instruction.strip()
        instruction_enabled = (
            bool(instruction)
            if use_query_instruction is None
            else use_query_instruction and bool(instruction)
        )
        if query_mode and instruction_enabled:
            prepared_texts = [
                f"Instruct: {instruction}\nQuery: {text}" for text in texts
            ]

        batch_size = max(1, self.settings.embedding_batch_size)
        embeddings: list[list[float]] = []

        total_batches = (len(prepared_texts) + batch_size - 1) // batch_size

        for batch_index, start in enumerate(
            range(0, len(prepared_texts), batch_size),
            start=1,
        ):
            batch = prepared_texts[start : start + batch_size]
            if progress_callback is not None:
                progress_callback(
                    f"Embeddings lote {batch_index}/{total_batches} "
                    f"({len(batch)} textos)"
                )
            response = self.embedding_client.embeddings.create(
                model=self.settings.default_endpoint_model,
                input=batch,
            )
            embeddings.extend(item.embedding for item in response.data)

        return embeddings

    def generate_query_variants(self, question: str) -> list[str]:
        """Genera dos consultas equivalentes para un experimento de retrieval."""
        normalized_question = question.strip()
        if not normalized_question:
            raise ValueError("La pregunta no puede estar vacia.")
        response = self.chat_client.chat.completions.create(
            model=self.settings.default_endpoint_model,
            temperature=0.0,
            max_tokens=min(self.settings.max_tokens, 256),
            messages=self._build_query_augmentation_messages(normalized_question),
            extra_body=self._thinking_extra_body(False),
        )
        content = getattr(response.choices[0].message, "content", None)
        variants = self._parse_query_variants(
            normalized_question,
            content if isinstance(content, str) else "",
        )
        if len(variants) != 2:
            raise RuntimeError(
                "El modelo local no devolvio dos reformulaciones validas. "
                "Revisa el modelo o el prompt de query augmentation."
            )
        return variants

    def rewrite_question_for_retrieval(
        self,
        question: str,
        messages: list[dict[str, str]] | None = None,
    ) -> str:
        """Reescritura determinista: nunca carga ni consulta el modelo de chat."""
        normalized_history = self._normalize_messages(messages or [])
        if not normalized_history:
            return question
        previous_user_messages = [
            message["content"]
            for message in normalized_history
            if message["role"] == "user"
        ]
        if not previous_user_messages:
            return question
        # La variante contextual completa se añade después en RagPipeline. Esta
        # función conserva la pregunta literal y evita cualquier dependencia de chat.
        return question

    def generate_answer(
        self,
        question: str,
        context_blocks: list[str],
        messages: list[dict[str, str]] | None = None,
        enable_reasoning: bool = False,
    ) -> dict[str, str]:
        """Solicita una respuesta completa al modelo de chat."""
        response = self.chat_client.chat.completions.create(
            model=self.settings.default_endpoint_model,
            temperature=0.2,
            max_tokens=self._generation_max_tokens(enable_reasoning),
            messages=self._build_messages(question, context_blocks, messages=messages),
            extra_body=self._thinking_extra_body(enable_reasoning),
        )
        message = response.choices[0].message
        content = getattr(message, "content", None)
        answer = content if isinstance(content, str) else ""

        reasoning_content = getattr(message, "reasoning_content", None)
        reasoning = reasoning_content if isinstance(reasoning_content, str) else ""

        # Un modelo con thinking puede agotar el limite antes de producir la
        # respuesta final. Conserva su razonamiento y garantiza una respuesta
        # mediante una segunda llamada breve sin thinking.
        if enable_reasoning and not answer.strip():
            fallback = self.generate_answer(
                question,
                context_blocks,
                messages=messages,
                enable_reasoning=False,
            )
            answer = fallback["answer"]

        return {
            "answer": answer,
            "reasoning": reasoning,
        }

    def stream_answer(
        self,
        question: str,
        context_blocks: list[str],
        messages: list[dict[str, str]] | None = None,
        enable_reasoning: bool = False,
    ):
        """Solicita una respuesta en streaming y va emitiendo tokens/texto."""
        stream = self.chat_client.chat.completions.create(
            model=self.settings.default_endpoint_model,
            temperature=0.2,
            max_tokens=self._generation_max_tokens(enable_reasoning),
            stream=True,
            messages=self._build_messages(question, context_blocks, messages=messages),
            extra_body=self._thinking_extra_body(enable_reasoning),
        )

        emitted_answer = False
        for chunk in stream:
            if not chunk.choices:
                continue

            choice = chunk.choices[0]
            delta = getattr(choice, "delta", None)

            if delta is not None:
                content = getattr(delta, "content", None)
                if isinstance(content, str) and content:
                    emitted_answer = True
                    yield {"type": "answer", "delta": content}
                    continue

                reasoning_content = getattr(delta, "reasoning_content", None)
                if isinstance(reasoning_content, str) and reasoning_content:
                    yield {"type": "reasoning", "delta": reasoning_content}
                    continue

            text = getattr(choice, "text", None)
            if isinstance(text, str) and text:
                emitted_answer = True
                yield {"type": "answer", "delta": text}

        # Un modelo con razonamiento puede consumir todo su presupuesto sin
        # cerrar el bloque de thinking. En ese caso completa el mismo flujo con
        # una fase final sin thinking, que también se entrega token a token.
        if enable_reasoning and not emitted_answer:
            yield from self.stream_answer(
                question,
                context_blocks,
                messages=messages,
                enable_reasoning=False,
            )

    def select_used_source_ids(
        self,
        question: str,
        answer: str,
        source_options: list[dict[str, Any]],
        messages: list[dict[str, str]] | None = None,
    ) -> list[str]:
        """Selecciona solo las fuentes recuperadas que soportan realmente la respuesta.

        Esta segunda llamada no modifica la respuesta. Actúa como auditor de citas:
        recibe la respuesta final y los chunks candidatos enviados al modelo, y devuelve
        únicamente los IDs de fuente que apoyan afirmaciones concretas de la respuesta.
        """
        normalized_answer = answer.strip()
        if not normalized_answer or not source_options:
            return []

        allowed_ids = {str(option.get("source_id", "")).strip() for option in source_options}
        allowed_ids.discard("")
        if not allowed_ids:
            return []

        try:
            response = self.chat_client.chat.completions.create(
                model=self.settings.default_endpoint_model,
                temperature=0.0,
                max_tokens=min(self.settings.max_tokens, 256),
                messages=self._build_source_attribution_messages(
                    question=question,
                    answer=normalized_answer,
                    source_options=source_options,
                    messages=messages,
                ),
                extra_body=self._thinking_extra_body(False),
            )
        except Exception:
            return []

        message = response.choices[0].message
        content = getattr(message, "content", None)
        parsed_ids = self._parse_used_source_ids(content if isinstance(content, str) else "")

        used_ids: list[str] = []
        for source_id in parsed_ids:
            if source_id in allowed_ids and source_id not in used_ids:
                used_ids.append(source_id)

        return used_ids

    @classmethod
    def _build_messages(
        cls,
        question: str,
        context_blocks: list[str],
        messages: list[dict[str, str]] | None = None,
    ) -> list[dict[str, str]]:
        """Construye la conversacion final para el backend de chat."""
        normalized_history = cls._normalize_messages(messages or [])
        final_user_message = {
            "role": "user",
            "content": cls._build_rag_user_prompt(question, context_blocks),
        }

        if (
            normalized_history
            and normalized_history[-1]["role"] == "user"
            and normalized_history[-1]["content"].strip() == question.strip()
        ):
            normalized_history = normalized_history[:-1]

        return [
            {
                "role": "system",
                "content": (
                    "Answer in the same language as the user and be concise. "
                    "Use retrieved context when it is relevant, preserve the current chat history, "
                    "and say clearly when the answer is missing from both the retrieved context and "
                    "the conversation."
                ),
            },
            *normalized_history,
            final_user_message,
        ]

    @classmethod
    def _build_source_attribution_messages(
        cls,
        question: str,
        answer: str,
        source_options: list[dict[str, Any]],
        messages: list[dict[str, str]] | None = None,
    ) -> list[dict[str, str]]:
        """Construye el prompt que audita qué fuentes soportan la respuesta."""
        normalized_history = cls._normalize_messages(messages or [])
        source_blocks = []

        for option in source_options:
            source_id = str(option.get("source_id", "")).strip()
            if not source_id:
                continue

            source = str(option.get("source", "")).strip() or "unknown source"
            page_start = option.get("page_start")
            page_end = option.get("page_end")
            page_label = (
                str(page_start)
                if page_start == page_end or page_end in (None, "")
                else f"{page_start}-{page_end}"
            )
            text = cls._truncate_for_attribution(str(option.get("text", "")))
            source_blocks.append(f"[{source_id}] {source} p.{page_label}\n{text}")

        sources = "\n\n".join(source_blocks) or "No retrieved sources."

        return [
            {
                "role": "system",
                "content": (
                    "You are a strict citation auditor for a RAG system. "
                    "Your job is to decide which retrieved source IDs directly support factual claims in the final answer. "
                    "Return only sources that are actually used or necessary to justify the answer. "
                    "Do not include sources that are merely topically related, duplicate background, or unused. "
                    "If the answer is supported only by conversation history, or no retrieved source directly supports it, return an empty list. "
                    "Return valid JSON only, with this exact shape: {\"used_source_ids\": [\"S1\"]}."
                ),
            },
            *normalized_history,
            {
                "role": "user",
                "content": (
                    f"Latest user question:\n{question}\n\n"
                    f"Final answer to audit:\n{answer}\n\n"
                    f"Retrieved source candidates:\n{sources}\n\n"
                    "Select only the source IDs that directly support the final answer. "
                    "Return JSON only."
                ),
            },
        ]

    @staticmethod
    def _truncate_for_attribution(text: str, max_chars: int = 2500) -> str:
        """Limita cada chunk en la llamada de auditoría para evitar prompts excesivos."""
        normalized = text.strip()
        if len(normalized) <= max_chars:
            return normalized
        return normalized[:max_chars].rstrip() + "..."

    @staticmethod
    def _parse_used_source_ids(content: str) -> list[str]:
        """Extrae IDs de fuente desde JSON estricto o desde una respuesta imperfecta."""
        normalized = content.strip()
        if not normalized:
            return []

        candidates = [normalized]
        json_match = re.search(r"\{.*\}", normalized, flags=re.DOTALL)
        if json_match:
            candidates.append(json_match.group(0))

        for candidate in candidates:
            try:
                payload = json.loads(candidate)
            except json.JSONDecodeError:
                continue

            value = payload.get("used_source_ids") if isinstance(payload, dict) else payload
            if isinstance(value, list):
                return [str(item).strip() for item in value if str(item).strip()]

        # Fallback defensivo para modelos que devuelven texto en vez de JSON.
        return list(dict.fromkeys(re.findall(r"\bS\d+\b", normalized)))

    @staticmethod
    def _parse_query_variants(question: str, content: str) -> list[str]:
        """Acepta JSON estricto y listas simples sin perder reproducibilidad."""
        normalized = content.strip()
        candidates: list[str] = []
        json_candidates = [normalized]
        json_match = re.search(r"(?:\{.*\}|\[.*\])", normalized, flags=re.DOTALL)
        if json_match and json_match.group(0) != normalized:
            json_candidates.append(json_match.group(0))
        for candidate in json_candidates:
            try:
                payload = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            value = payload.get("queries") if isinstance(payload, dict) else payload
            if isinstance(value, list):
                candidates.extend(str(item).strip() for item in value)
                break
        if not candidates:
            candidates.extend(
                re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip(" \t\"'")
                for line in normalized.splitlines()
            )

        variants: list[str] = []
        original = question.strip().casefold()
        for candidate in candidates:
            cleaned = " ".join(candidate.split())
            if not cleaned or cleaned.casefold() == original:
                continue
            if cleaned.casefold() not in {item.casefold() for item in variants}:
                variants.append(cleaned)
            if len(variants) == 2:
                break
        return variants

    @staticmethod
    def _normalize_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
        """Filtra y normaliza mensajes con role/content validos."""
        normalized = []
        for message in messages:
            role = str(message.get("role", "")).strip()
            content = str(message.get("content", "")).strip()
            if not role or not content:
                continue
            normalized.append({"role": role, "content": content})
        return normalized

    @staticmethod
    def _build_rewrite_messages(
        question: str,
        normalized_history: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        """Construye el prompt para reescribir consultas de retrieval con historial."""
        return [
            {
                "role": "system",
                "content": (
                    "You are a query rewriter for a RAG system. "
                    "Your only task is to rewrite the user's latest question as a standalone search query. "
                    "Use the conversation history only to recover omitted context such as the subject, document, entity, or topic. "
                    "Do not answer the question. "
                    "Do not explain anything. "
                    "Do not show your reasoning. "
                    "Do not use tags such as <think>. "
                    "Return exactly one standalone search query, in the same language as the user's latest question."
                ),
            },
            *normalized_history,
            {
                "role": "user",
                "content": (
                    "Rewrite the following question for document retrieval. "
                    "The rewritten query must be self-contained, concise, and preserve the original meaning. "
                    "If the question depends on the conversation history, include the missing context. "
                    "Return only the final rewritten query.\n\n"
                    f"Latest question: {question}"
                ),
            },
        ]

    @staticmethod
    def _build_query_augmentation_messages(question: str) -> list[dict[str, str]]:
        return [
            {
                "role": "system",
                "content": (
                    "You generate search-query variants for a RAG retriever. "
                    "Return exactly two concise reformulations that preserve the "
                    "meaning and language of the original question. Do not answer it, "
                    "add facts, explain, or show reasoning. Return strict JSON only: "
                    '{"queries":["first variant","second variant"]}.'
                ),
            },
            {"role": "user", "content": question},
        ]

    @staticmethod
    def _build_rag_user_prompt(question: str, context_blocks: list[str]) -> str:
        """Construye el turno final con la pregunta y el contexto RAG."""
        context = "\n\n".join(context_blocks) if context_blocks else "No relevant context was retrieved."
        return (
            "Answer the user's latest question using the conversation history plus the retrieved context below. "
            "Prefer retrieved context for factual claims grounded in documents. "
            "If the answer comes from the chat history instead of the retrieved documents, you may answer from the chat. "
            "Do not invent details.\n\n"
            f"Latest question:\n{question}\n\n"
            f"Retrieved context:\n{context}"
        )
