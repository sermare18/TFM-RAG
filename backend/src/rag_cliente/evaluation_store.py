"""Persistencia SQLite del dataset manual y de las evaluaciones de retrieval."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


@dataclass(frozen=True, slots=True)
class RelevantPage:
    document_id: str
    source: str
    source_path: str
    page: int
    reference_text: str = ""


@dataclass(frozen=True, slots=True)
class QuestionRecord:
    id: int
    question: str
    category: str
    notes: str
    active: bool
    created_at: str
    updated_at: str
    relevant_pages: tuple[RelevantPage, ...]

    def dataset_item(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "question": self.question,
            "relevant_pages": [
                {"document_id": page.document_id, "page": page.page}
                for page in self.relevant_pages
            ],
        }


class EvaluationStore:
    """Repositorio local pequeno; cada operacion abre una conexion corta."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connection() as connection:
            connection.executescript(
                """
                PRAGMA journal_mode = WAL;

                CREATE TABLE IF NOT EXISTS questions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    question TEXT NOT NULL,
                    category TEXT NOT NULL DEFAULT '',
                    notes TEXT NOT NULL DEFAULT '',
                    active INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS relevant_pages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    question_id INTEGER NOT NULL REFERENCES questions(id) ON DELETE CASCADE,
                    document_id TEXT NOT NULL,
                    source TEXT NOT NULL,
                    source_path TEXT NOT NULL,
                    page INTEGER NOT NULL CHECK(page >= 1),
                    reference_text TEXT NOT NULL DEFAULT '',
                    UNIQUE(question_id, document_id, page)
                );

                CREATE TABLE IF NOT EXISTS evaluations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    completed_at TEXT,
                    dataset_hash TEXT NOT NULL,
                    question_count INTEGER NOT NULL,
                    config_json TEXT NOT NULL,
                    metrics_json TEXT,
                    error TEXT
                );

                CREATE TABLE IF NOT EXISTS evaluation_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    evaluation_id INTEGER NOT NULL REFERENCES evaluations(id) ON DELETE CASCADE,
                    question_id INTEGER,
                    question_text TEXT NOT NULL,
                    expected_json TEXT NOT NULL,
                    retrieved_json TEXT NOT NULL,
                    query_variants_json TEXT NOT NULL,
                    metrics_json TEXT NOT NULL,
                    latency_ms REAL NOT NULL,
                    error TEXT
                );

                CREATE TABLE IF NOT EXISTS query_variants (
                    question_hash TEXT NOT NULL,
                    prompt_version TEXT NOT NULL,
                    question_text TEXT NOT NULL,
                    variants_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(question_hash, prompt_version)
                );

                CREATE INDEX IF NOT EXISTS idx_relevant_pages_question
                    ON relevant_pages(question_id);
                CREATE INDEX IF NOT EXISTS idx_results_evaluation
                    ON evaluation_results(evaluation_id);
                """
            )

    @staticmethod
    def _validate_question(question: str, relevant_pages: list[RelevantPage]) -> str:
        normalized = question.strip()
        if not normalized:
            raise ValueError("La pregunta no puede estar vacía.")
        if not relevant_pages:
            raise ValueError("Selecciona al menos una página relevante.")
        if any(not page.document_id.strip() or page.page < 1 for page in relevant_pages):
            raise ValueError("Las páginas relevantes no son válidas.")
        return normalized

    def save_question(
        self,
        question: str,
        relevant_pages: list[RelevantPage],
        *,
        category: str = "",
        notes: str = "",
        active: bool = True,
        question_id: int | None = None,
    ) -> int:
        normalized = self._validate_question(question, relevant_pages)
        now = _utc_now()
        with self._connection() as connection:
            if question_id is None:
                cursor = connection.execute(
                    """
                    INSERT INTO questions(question, category, notes, active, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (normalized, category.strip(), notes.strip(), int(active), now, now),
                )
                saved_id = int(cursor.lastrowid)
            else:
                cursor = connection.execute(
                    """
                    UPDATE questions
                    SET question = ?, category = ?, notes = ?, active = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        normalized,
                        category.strip(),
                        notes.strip(),
                        int(active),
                        now,
                        question_id,
                    ),
                )
                if cursor.rowcount != 1:
                    raise KeyError(f"No existe la pregunta {question_id}.")
                saved_id = question_id
                connection.execute(
                    "DELETE FROM relevant_pages WHERE question_id = ?",
                    (saved_id,),
                )

            unique_pages: dict[tuple[str, int], RelevantPage] = {}
            for page in relevant_pages:
                unique_pages[(page.document_id, page.page)] = page
            connection.executemany(
                """
                INSERT INTO relevant_pages(
                    question_id, document_id, source, source_path, page, reference_text
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        saved_id,
                        page.document_id.strip(),
                        page.source.strip(),
                        page.source_path.strip(),
                        page.page,
                        page.reference_text.strip(),
                    )
                    for page in unique_pages.values()
                ],
            )
        return saved_id

    def delete_question(self, question_id: int) -> bool:
        with self._connection() as connection:
            cursor = connection.execute("DELETE FROM questions WHERE id = ?", (question_id,))
            return cursor.rowcount == 1

    def list_questions(self, *, active_only: bool = False) -> list[QuestionRecord]:
        where = "WHERE active = 1" if active_only else ""
        with self._connection() as connection:
            question_rows = connection.execute(
                f"SELECT * FROM questions {where} ORDER BY id"  # noqa: S608 - clausula fija
            ).fetchall()
            page_rows = connection.execute(
                "SELECT * FROM relevant_pages ORDER BY question_id, document_id, page"
            ).fetchall()
        pages_by_question: dict[int, list[RelevantPage]] = {}
        for row in page_rows:
            pages_by_question.setdefault(int(row["question_id"]), []).append(
                RelevantPage(
                    document_id=str(row["document_id"]),
                    source=str(row["source"]),
                    source_path=str(row["source_path"]),
                    page=int(row["page"]),
                    reference_text=str(row["reference_text"]),
                )
            )
        return [
            QuestionRecord(
                id=int(row["id"]),
                question=str(row["question"]),
                category=str(row["category"]),
                notes=str(row["notes"]),
                active=bool(row["active"]),
                created_at=str(row["created_at"]),
                updated_at=str(row["updated_at"]),
                relevant_pages=tuple(pages_by_question.get(int(row["id"]), [])),
            )
            for row in question_rows
        ]

    def get_question(self, question_id: int) -> QuestionRecord:
        for question in self.list_questions():
            if question.id == question_id:
                return question
        raise KeyError(f"No existe la pregunta {question_id}.")

    def active_dataset(self) -> tuple[list[QuestionRecord], str]:
        questions = self.list_questions(active_only=True)
        payload = [question.dataset_item() for question in questions]
        digest = hashlib.sha256(_json_dump(payload).encode("utf-8")).hexdigest()
        return questions, digest

    def start_evaluation(
        self,
        name: str,
        config: dict[str, Any],
        dataset_hash: str,
        question_count: int,
    ) -> int:
        normalized_name = name.strip()
        if not normalized_name:
            raise ValueError("La evaluación necesita un nombre.")
        with self._connection() as connection:
            cursor = connection.execute(
                """
                INSERT INTO evaluations(
                    name, status, created_at, dataset_hash, question_count, config_json
                )
                SELECT ?, 'running', ?, ?, ?, ?
                WHERE NOT EXISTS (
                    SELECT 1 FROM evaluations WHERE name = ? COLLATE NOCASE
                )
                """,
                (
                    normalized_name,
                    _utc_now(),
                    dataset_hash,
                    question_count,
                    _json_dump(config),
                    normalized_name,
                ),
            )
            if cursor.rowcount != 1:
                raise ValueError(
                    f"Ya existe una evaluación llamada '{normalized_name}'. "
                    "Utiliza un nombre diferente."
                )
            return int(cursor.lastrowid)

    def add_evaluation_result(
        self,
        evaluation_id: int,
        *,
        question_id: int,
        question_text: str,
        expected: list[dict[str, Any]],
        retrieved: list[dict[str, Any]],
        query_variants: list[str],
        metrics: dict[str, Any],
        latency_ms: float,
        error: str | None = None,
    ) -> None:
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO evaluation_results(
                    evaluation_id, question_id, question_text, expected_json,
                    retrieved_json, query_variants_json, metrics_json, latency_ms, error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    evaluation_id,
                    question_id,
                    question_text,
                    _json_dump(expected),
                    _json_dump(retrieved),
                    _json_dump(query_variants),
                    _json_dump(metrics),
                    float(latency_ms),
                    error,
                ),
            )

    def finish_evaluation(self, evaluation_id: int, metrics: dict[str, Any]) -> None:
        with self._connection() as connection:
            connection.execute(
                """
                UPDATE evaluations
                SET status = 'completed', completed_at = ?, metrics_json = ?, error = NULL
                WHERE id = ?
                """,
                (_utc_now(), _json_dump(metrics), evaluation_id),
            )

    def fail_evaluation(self, evaluation_id: int, error: str) -> None:
        with self._connection() as connection:
            connection.execute(
                """
                UPDATE evaluations
                SET status = 'failed', completed_at = ?, error = ?
                WHERE id = ?
                """,
                (_utc_now(), error, evaluation_id),
            )

    @staticmethod
    def _decode_evaluation(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": int(row["id"]),
            "name": str(row["name"]),
            "status": str(row["status"]),
            "created_at": str(row["created_at"]),
            "completed_at": row["completed_at"],
            "dataset_hash": str(row["dataset_hash"]),
            "question_count": int(row["question_count"]),
            "config": json.loads(str(row["config_json"])),
            "metrics": json.loads(str(row["metrics_json"])) if row["metrics_json"] else {},
            "error": row["error"],
        }

    def list_evaluations(self) -> list[dict[str, Any]]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM evaluations ORDER BY id DESC"
            ).fetchall()
        return [self._decode_evaluation(row) for row in rows]

    def delete_evaluation(self, evaluation_id: int) -> bool:
        """Elimina una evaluación y sus resultados asociados en cascada."""
        with self._connection() as connection:
            cursor = connection.execute(
                "DELETE FROM evaluations WHERE id = ?",
                (evaluation_id,),
            )
            return cursor.rowcount == 1

    def get_evaluation(self, evaluation_id: int) -> dict[str, Any]:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM evaluations WHERE id = ?",
                (evaluation_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"No existe la evaluación {evaluation_id}.")
            result_rows = connection.execute(
                "SELECT * FROM evaluation_results WHERE evaluation_id = ? ORDER BY id",
                (evaluation_id,),
            ).fetchall()
        evaluation = self._decode_evaluation(row)
        evaluation["results"] = [
            {
                "id": int(result["id"]),
                "question_id": result["question_id"],
                "question_text": str(result["question_text"]),
                "expected": json.loads(str(result["expected_json"])),
                "retrieved": json.loads(str(result["retrieved_json"])),
                "query_variants": json.loads(str(result["query_variants_json"])),
                "metrics": json.loads(str(result["metrics_json"])),
                "latency_ms": float(result["latency_ms"]),
                "error": result["error"],
            }
            for result in result_rows
        ]
        return evaluation

    @staticmethod
    def _question_hash(question: str) -> str:
        return hashlib.sha256(question.strip().encode("utf-8")).hexdigest()

    def get_query_variants(self, question: str, prompt_version: str) -> list[str] | None:
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT variants_json FROM query_variants
                WHERE question_hash = ? AND prompt_version = ?
                """,
                (self._question_hash(question), prompt_version),
            ).fetchone()
        if row is None:
            return None
        return [str(item) for item in json.loads(str(row["variants_json"]))]

    def save_query_variants(
        self,
        question: str,
        prompt_version: str,
        variants: list[str],
    ) -> None:
        with self._connection() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO query_variants(
                    question_hash, prompt_version, question_text, variants_json, created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    self._question_hash(question),
                    prompt_version,
                    question.strip(),
                    _json_dump(variants),
                    _utc_now(),
                ),
            )
