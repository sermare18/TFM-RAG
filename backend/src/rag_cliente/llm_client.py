"""Cliente de chat y embeddings para endpoints compatibles con OpenAI."""

from __future__ import annotations

import json
import re
from typing import Any, Callable

from openai import OpenAI

ProgressCallback = Callable[[str], None]

from rag_cliente.config import Settings


class LlamaCppClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

        self.chat_client = OpenAI(
            base_url=settings.llama_cpp_chat_base_url,
            api_key=settings.openai_api_key,
            timeout=settings.request_timeout,
        )
        self.embedding_client = OpenAI(
            base_url=settings.llama_cpp_embedding_base_url,
            api_key=settings.openai_api_key,
            timeout=settings.request_timeout,
        )

    def embed_texts(
        self,
        texts: list[str],
        progress_callback: ProgressCallback | None = None,
    ) -> list[list[float]]:
        """Genera embeddings para una lista de textos en lotes pequeños."""
        if not texts:
            return []

        batch_size = max(1, self.settings.embedding_batch_size)
        embeddings: list[list[float]] = []

        total_batches = (len(texts) + batch_size - 1) // batch_size

        for batch_index, start in enumerate(range(0, len(texts), batch_size), start=1):
            batch = texts[start : start + batch_size]
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

    def rewrite_question_for_retrieval(
        self,
        question: str,
        messages: list[dict[str, str]] | None = None,
    ) -> str:
        """Convierte una follow-up en una consulta autocontenida para retrieval."""
        normalized_history = self._normalize_messages(messages or [])
        if not normalized_history:
            return question

        try:
            response = self.chat_client.chat.completions.create(
                model=self.settings.default_endpoint_model,
                temperature=0.0,
                max_tokens=min(self.settings.max_tokens, 2048),
                messages=self._build_rewrite_messages(question, normalized_history),
            )
        except Exception:
            return question

        message = response.choices[0].message
        content = getattr(message, "content", None)
        rewritten_question = content.strip() if isinstance(content, str) else ""
        return rewritten_question or question

    def generate_answer(
        self,
        question: str,
        context_blocks: list[str],
        messages: list[dict[str, str]] | None = None,
    ) -> dict[str, str]:
        """Solicita una respuesta completa al modelo de chat."""
        response = self.chat_client.chat.completions.create(
            model=self.settings.default_endpoint_model,
            temperature=0.2,
            max_tokens=self.settings.max_tokens,
            messages=self._build_messages(question, context_blocks, messages=messages),
        )
        message = response.choices[0].message
        content = getattr(message, "content", None)
        answer = content if isinstance(content, str) else ""

        reasoning_content = getattr(message, "reasoning_content", None)
        reasoning = reasoning_content if isinstance(reasoning_content, str) else ""

        return {
            "answer": answer,
            "reasoning": reasoning,
        }

    def stream_answer(
        self,
        question: str,
        context_blocks: list[str],
        messages: list[dict[str, str]] | None = None,
    ):
        """Solicita una respuesta en streaming y va emitiendo tokens/texto."""
        stream = self.chat_client.chat.completions.create(
            model=self.settings.default_endpoint_model,
            temperature=0.2,
            max_tokens=self.settings.max_tokens,
            stream=True,
            messages=self._build_messages(question, context_blocks, messages=messages),
        )

        for chunk in stream:
            if not chunk.choices:
                continue

            choice = chunk.choices[0]
            delta = getattr(choice, "delta", None)

            if delta is not None:
                content = getattr(delta, "content", None)
                if isinstance(content, str) and content:
                    yield {"type": "answer", "delta": content}
                    continue

                reasoning_content = getattr(delta, "reasoning_content", None)
                if isinstance(reasoning_content, str) and reasoning_content:
                    yield {"type": "reasoning", "delta": reasoning_content}
                    continue

            text = getattr(choice, "text", None)
            if isinstance(text, str) and text:
                yield {"type": "answer", "delta": text}

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
                max_tokens=4096,
                messages=self._build_source_attribution_messages(
                    question=question,
                    answer=normalized_answer,
                    source_options=source_options,
                    messages=messages,
                ),
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
