"""Interfaz Streamlit para construir y evaluar un dataset de retrieval."""

from __future__ import annotations

import csv
import io
from collections import defaultdict
from typing import Any

import streamlit as st

from rag_cliente.config import get_settings
from rag_cliente.evaluation_store import EvaluationStore, QuestionRecord, RelevantPage
from rag_cliente.evaluator import EvaluationConfig, EvaluationRunner
from rag_cliente.pipeline import RagPipeline


def shorten(text: str, max_chars: int = 120) -> str:
    compact = " ".join(text.split())
    return compact if len(compact) <= max_chars else compact[:max_chars].rstrip() + "..."


@st.cache_resource
def build_components() -> tuple[EvaluationStore, RagPipeline]:
    settings = get_settings()
    return EvaluationStore(settings.evaluation_db_path), RagPipeline(settings)


def load_index_rows(pipeline: RagPipeline) -> list[dict[str, Any]]:
    if not pipeline.store.table_exists():
        raise RuntimeError("No existe el indice. Ejecuta primero '.\\rag.bat index'.")
    return pipeline.store.list_chunks(include_vector=False)


def document_options(rows: list[dict[str, Any]]) -> dict[str, str]:
    options: dict[str, str] = {}
    for row in rows:
        document_id = str(row.get("document_id", ""))
        if document_id and document_id not in options:
            tag = str(row.get("tag") or "")
            prefix = f"[{tag}] " if tag else ""
            options[document_id] = f"{prefix}{row.get('source', '')} · {row.get('source_path', '')}"
    return options


def question_table(questions: list[QuestionRecord]) -> list[dict[str, Any]]:
    return [
        {
            "id": question.id,
            "activa": question.active,
            "pregunta": question.question,
            "categoria": question.category,
            "paginas relevantes": ", ".join(
                f"{page.source} p.{page.page}" for page in question.relevant_pages
            ),
        }
        for question in questions
    ]


def render_dataset(
    store: EvaluationStore,
    rows: list[dict[str, Any]],
) -> None:
    questions = store.list_questions()
    active_count = sum(question.active for question in questions)
    st.subheader("Dataset de referencia")
    st.caption(
        "Las selecciones se muestran por chunk, pero la verdad de referencia se "
        "guarda por documento y página."
    )
    left, right = st.columns(2)
    left.metric("Preguntas", len(questions))
    right.metric("Preguntas activas", active_count)
    if questions:
        st.dataframe(question_table(questions), width="stretch", hide_index=True)

    choices = ["Nueva pregunta"]
    choices.extend(f"Editar #{item.id}: {shorten(item.question, 70)}" for item in questions)
    selected_action = st.selectbox("Acción", choices)
    editing: QuestionRecord | None = None
    if selected_action != "Nueva pregunta":
        selected_id = int(selected_action.split("#", 1)[1].split(":", 1)[0])
        editing = store.get_question(selected_id)

    form_key = str(editing.id) if editing else f"new-{st.session_state.get('form_nonce', 0)}"
    question = st.text_area(
        "Pregunta",
        value=editing.question if editing else "",
        key=f"question-{form_key}",
    )
    category = st.text_input(
        "Categoría (opcional)",
        value=editing.category if editing else "",
        placeholder="texto, tabla, multipágina...",
        key=f"category-{form_key}",
    )
    active = st.checkbox(
        "Incluir esta pregunta en las evaluaciones",
        value=editing.active if editing else True,
        key=f"active-{form_key}",
        help="Si se desmarca, la pregunta se conserva pero no participa en nuevas evaluaciones.",
    )
    notes = st.text_area(
        "Notas (opcional)",
        value=editing.notes if editing else "",
        key=f"notes-{form_key}",
    )

    documents = document_options(rows)
    if not documents:
        st.warning("El índice no contiene documentos.")
        return
    default_document = (
        editing.relevant_pages[0].document_id
        if editing and editing.relevant_pages
        else next(iter(documents))
    )
    document_ids = list(documents)
    default_index = document_ids.index(default_document) if default_document in documents else 0
    selected_document = st.selectbox(
        "Documento",
        document_ids,
        index=default_index,
        format_func=lambda item: documents[item],
        key=f"document-{form_key}",
    )
    chunks = sorted(
        [row for row in rows if str(row.get("document_id", "")) == selected_document],
        key=lambda row: (int(row.get("page_start", 0)), int(row.get("page_chunk_index", 0))),
    )
    chunk_map = {str(row.get("chunk_id") or row.get("id")): row for row in chunks}
    expected_pages = {
        page.page
        for page in (editing.relevant_pages if editing else ())
        if page.document_id == selected_document
    }
    default_chunks = [
        chunk_id
        for chunk_id, row in chunk_map.items()
        if int(row.get("page_start", 0)) in expected_pages
    ]
    selected_chunks = st.multiselect(
        "Chunks que contienen la información",
        list(chunk_map),
        default=default_chunks,
        format_func=lambda chunk_id: (
            f"p.{chunk_map[chunk_id].get('page_start')} · chunk "
            f"{chunk_map[chunk_id].get('page_chunk_index', 0)} · "
            f"{shorten(str(chunk_map[chunk_id].get('text', '')), 100)}"
        ),
        key=f"chunks-{form_key}-{selected_document}",
    )

    pages = sorted({int(row.get("page_start", 0)) for row in chunks})
    if pages:
        preview_page = st.selectbox(
            "Previsualizar página",
            pages,
            format_func=lambda page: f"Página {page}",
            key=f"preview-{form_key}-{selected_document}",
        )
        for row in chunks:
            if int(row.get("page_start", 0)) == preview_page:
                with st.expander(
                    f"Chunk {row.get('page_chunk_index', 0)} · {row.get('token_count', 0)} tokens"
                ):
                    st.code(str(row.get("text", "")), language=None)

    save_col, delete_col = st.columns([3, 1])
    if save_col.button(
        "Guardar pregunta" if editing is None else "Guardar cambios",
        type="primary",
        key=f"save-{form_key}",
    ):
        selected_rows = [chunk_map[chunk_id] for chunk_id in selected_chunks]
        grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
        for row in selected_rows:
            grouped[(selected_document, int(row.get("page_start", 0)))].append(row)
        relevant_pages = [
            RelevantPage(
                document_id=document_id,
                source=str(page_rows[0].get("source", "")),
                source_path=str(page_rows[0].get("source_path", "")),
                page=page,
                reference_text="\n\n".join(str(row.get("text", "")) for row in page_rows),
            )
            for (document_id, page), page_rows in sorted(grouped.items())
        ]
        try:
            store.save_question(
                question,
                relevant_pages,
                category=category,
                notes=notes,
                active=active,
                question_id=editing.id if editing else None,
            )
        except (ValueError, KeyError) as exc:
            st.error(str(exc))
        else:
            st.session_state["form_nonce"] = st.session_state.get("form_nonce", 0) + 1
            st.success("Pregunta guardada.")
            st.rerun()

    if editing is not None:
        confirm_delete = delete_col.checkbox("Confirmar borrado", key=f"confirm-{form_key}")
        if delete_col.button("Eliminar", disabled=not confirm_delete, key=f"delete-{form_key}"):
            store.delete_question(editing.id)
            st.success("Pregunta eliminada. Los resultados históricos se conservan.")
            st.rerun()


def render_metrics(metrics: dict[str, Any]) -> None:
    if not metrics:
        return
    top_k = int(metrics.get("top_k", 0))
    columns = st.columns(6)
    columns[0].metric("Hit@1", f"{100 * metrics.get('hit_at_1', 0):.1f}%")
    columns[1].metric(f"Hit@{top_k}", f"{100 * metrics.get('hit_at_k', 0):.1f}%")
    columns[2].metric("Recall@1", f"{100 * metrics.get('recall_at_1', 0):.1f}%")
    columns[3].metric(f"Recall@{top_k}", f"{100 * metrics.get('recall_at_k', 0):.1f}%")
    columns[4].metric(f"MRR@{top_k}", f"{metrics.get('mrr_at_k', 0):.3f}")
    columns[5].metric(f"Precision@{top_k}", f"{metrics.get('precision_at_k', 0):.3f}")
    st.caption(
        f"Falsos positivos: {metrics.get('false_positives', 0)} · "
        f"Fallos: {metrics.get('failures', 0)} · "
        f"Errores: {metrics.get('error_count', 0)} · "
        f"Latencia media: {metrics.get('latency_mean_ms', 0):.0f} ms · "
        f"p95: {metrics.get('latency_p95_ms', 0):.0f} ms"
    )


def render_new_evaluation(
    store: EvaluationStore,
    pipeline: RagPipeline,
    rows: list[dict[str, Any]],
) -> None:
    questions, dataset_hash = store.active_dataset()
    st.subheader("Nueva evaluación")
    st.caption(
        f"Dataset actual: {len(questions)} preguntas activas · versión {dataset_hash[:10]}"
    )
    if not questions:
        st.warning("Añade al menos una pregunta activa antes de evaluar.")
        return
    name = st.text_input("Nombre", placeholder="Híbrido cosine con instrucción")
    col_mode, col_k = st.columns(2)
    mode = col_mode.selectbox(
        "Modo de recuperación",
        ["hybrid", "vector", "bm25"],
        format_func=lambda value: {"hybrid": "Híbrido", "vector": "Vector", "bm25": "BM25"}[value],
    )
    top_k = int(col_k.number_input("Top K", min_value=1, max_value=20, value=5, step=1))
    if mode != "bm25":
        col_distance, col_instruction = st.columns(2)
        distance = col_distance.selectbox(
            "Distancia vectorial",
            ["cosine", "l2"],
            format_func=lambda value: "Coseno" if value == "cosine" else "L2",
        )
        use_instruction = col_instruction.checkbox(
            "Usar instrucción de embedding",
            value=True,
        )
    else:
        distance = "cosine"
        use_instruction = False
        st.info("BM25 no utiliza distancia ni instrucción de embedding.")
    use_augmentation = st.checkbox(
        "Query augmentation: pregunta original + dos reformulaciones",
        value=False,
    )
    tags = sorted({str(row.get("tag")) for row in rows if row.get("tag")})
    selected_tag = st.selectbox("Filtro por tag", ["Todos", *tags])
    tag = None if selected_tag == "Todos" else selected_tag
    if tag:
        st.warning(
            "Se evaluarán todas las preguntas activas, pero retrieval solo buscará "
            f"dentro del tag '{tag}'."
        )
    if use_augmentation:
        st.caption(
            "Las reformulaciones nuevas usan el chat local una sola vez y quedan en caché. "
            "Después se libera chat antes de cargar embeddings."
        )

    if st.button("Evaluar", type="primary"):
        config = EvaluationConfig(
            name=name,
            retrieval_mode=mode,  # type: ignore[arg-type]
            top_k=top_k,
            distance_type=distance,  # type: ignore[arg-type]
            use_query_instruction=use_instruction,
            use_query_augmentation=use_augmentation,
            tag=tag,
        )
        progress = st.progress(0.0)
        status = st.empty()

        def update_progress(completed: int, total: int, message: str) -> None:
            progress.progress(completed / total if total else 0.0)
            status.write(message)

        try:
            result = EvaluationRunner(store, pipeline).run(config, update_progress)
        except Exception as exc:
            st.error(f"No se pudo completar la evaluación: {exc}")
        else:
            st.session_state["last_evaluation_id"] = result["id"]
            st.success(f"Evaluación #{result['id']} completada.")
            render_metrics(result["metrics"])


def evaluation_table(evaluations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for evaluation in evaluations:
        config = evaluation["config"]
        metrics = evaluation["metrics"]
        top_k = config.get("top_k", "")
        rows.append(
            {
                "id": evaluation["id"],
                "nombre": evaluation["name"],
                "estado": evaluation["status"],
                "dataset": evaluation["dataset_hash"][:10],
                "preguntas": evaluation["question_count"],
                "modo": config.get("retrieval_mode"),
                "distancia": config.get("distance_type") or "—",
                "instrucción": config.get("use_query_instruction", False),
                "augmentation": config.get("use_query_augmentation", False),
                "K": top_k,
                f"Hit@K": metrics.get("hit_at_k"),
                f"Recall@K": metrics.get("recall_at_k"),
                "MRR": metrics.get("mrr_at_k"),
                "fallos": metrics.get("failures"),
            }
        )
    return rows


def results_csv(evaluation: dict[str, Any]) -> bytes:
    output = io.StringIO()
    fields = [
        "question_id",
        "question",
        "expected_pages",
        "retrieved_pages",
        "first_relevant_rank",
        "recall_at_1",
        "recall_at_k",
        "precision_at_k",
        "mrr_at_k",
        "false_positives",
        "failure",
        "latency_ms",
        "error",
    ]
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    for result in evaluation["results"]:
        metrics = result["metrics"]
        writer.writerow(
            {
                "question_id": result["question_id"],
                "question": result["question_text"],
                "expected_pages": "; ".join(
                    f"{page['source']} p.{page['page']}" for page in result["expected"]
                ),
                "retrieved_pages": "; ".join(
                    f"{page['source']} p.{page['page']}" for page in result["retrieved"]
                ),
                "first_relevant_rank": metrics.get("first_relevant_rank"),
                "recall_at_1": metrics.get("recall_at_1"),
                "recall_at_k": metrics.get("recall_at_k"),
                "precision_at_k": metrics.get("precision_at_k"),
                "mrr_at_k": metrics.get("mrr_at_k"),
                "false_positives": metrics.get("false_positives"),
                "failure": metrics.get("failure"),
                "latency_ms": result["latency_ms"],
                "error": result.get("error") or "",
            }
        )
    return output.getvalue().encode("utf-8-sig")


def render_history(store: EvaluationStore) -> None:
    st.subheader("Historial")
    evaluations = store.list_evaluations()
    if not evaluations:
        st.info("Todavía no hay evaluaciones.")
        return
    st.dataframe(evaluation_table(evaluations), width="stretch", hide_index=True)
    default_id = st.session_state.get("last_evaluation_id", evaluations[0]["id"])
    ids = [evaluation["id"] for evaluation in evaluations]
    selected_id = st.selectbox(
        "Ver evaluación",
        ids,
        index=ids.index(default_id) if default_id in ids else 0,
        format_func=lambda item: next(
            f"#{evaluation['id']} · {evaluation['name']}"
            for evaluation in evaluations
            if evaluation["id"] == item
        ),
    )
    evaluation = store.get_evaluation(selected_id)
    render_metrics(evaluation["metrics"])
    _questions, current_hash = store.active_dataset()
    if evaluation["dataset_hash"] != current_hash:
        st.warning(
            "Esta evaluación utilizó una versión distinta del dataset actual. "
            "Compárala únicamente con evaluaciones que tengan el mismo identificador de dataset."
        )
    if evaluation.get("error"):
        st.error(str(evaluation["error"]))
    st.download_button(
        "Descargar detalle CSV",
        data=results_csv(evaluation),
        file_name=f"evaluacion-{selected_id}.csv",
        mime="text/csv",
    )

    only_failures = st.checkbox("Mostrar solo fallos o errores", value=False)
    results = [
        result
        for result in evaluation["results"]
        if not only_failures or result["metrics"].get("failure") or result.get("error")
    ]
    st.dataframe(
        [
            {
                "pregunta": result["question_text"],
                "primer acierto": result["metrics"].get("first_relevant_rank"),
                "recall@K": result["metrics"].get("recall_at_k"),
                "falsos positivos": result["metrics"].get("false_positives"),
                "fallo": result["metrics"].get("failure"),
                "latencia ms": round(result["latency_ms"], 1),
                "error": result.get("error") or "",
            }
            for result in results
        ],
        width="stretch",
        hide_index=True,
    )
    if not results:
        return
    selected_result = st.selectbox(
        "Detalle de pregunta",
        range(len(results)),
        format_func=lambda index: results[index]["question_text"],
    )
    result = results[selected_result]
    st.write("**Páginas esperadas**")
    st.write(", ".join(f"{page['source']} p.{page['page']}" for page in result["expected"]))
    if result["query_variants"]:
        st.write("**Reformulaciones**")
        for variant in result["query_variants"]:
            st.write(f"- {variant}")
    st.write("**Ranking recuperado**")
    st.dataframe(
        [
            {
                "rank": item["rank"],
                "documento": item["source"],
                "página": item["page"],
                "distancia": item["distance"],
                "BM25": item["bm25_score"],
                "RRF": item["rrf_score"],
            }
            for item in result["retrieved"]
        ],
        width="stretch",
        hide_index=True,
    )
    for item in result["retrieved"]:
        with st.expander(f"#{item['rank']} · {item['source']} · p.{item['page']}"):
            st.code(item["text"], language=None)


def main() -> None:
    st.set_page_config(page_title="Evaluador RAG", layout="wide")
    st.title("Evaluador de recuperación RAG")
    st.caption("Dataset manual persistente y evaluaciones reproducibles sobre el índice actual.")
    store, pipeline = build_components()
    try:
        rows = load_index_rows(pipeline)
    except RuntimeError as exc:
        st.error(str(exc))
        st.stop()
    dataset_tab, evaluation_tab, history_tab = st.tabs(
        ["Dataset", "Nueva evaluación", "Historial"]
    )
    with dataset_tab:
        render_dataset(store, rows)
    with evaluation_tab:
        render_new_evaluation(store, pipeline, rows)
    with history_tab:
        render_history(store)


if __name__ == "__main__":
    main()
