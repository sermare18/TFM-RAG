from __future__ import annotations

from pathlib import Path

import streamlit as st

from rag_cliente.config import get_settings
from rag_cliente.vector_store import LanceDBStore


def shorten(text: str, max_chars: int = 220) -> str:
    compact = " ".join(text.split())
    return compact if len(compact) <= max_chars else compact[:max_chars] + "..."


def apply_filters(
    rows: list[dict],
    source_filter: list[str],
    source_type_filter: list[str],
    tag_filter: list[str],
    text_query: str,
) -> list[dict]:
    filtered = rows
    if source_filter:
        filtered = [row for row in filtered if row.get("source") in source_filter]
    if source_type_filter:
        filtered = [row for row in filtered if row.get("source_type") in source_type_filter]
    if tag_filter:
        filtered = [row for row in filtered if row.get("tag") in tag_filter]
    needle = text_query.strip().lower()
    if needle:
        filtered = [
            row
            for row in filtered
            if needle in str(row.get("text") or "").lower()
            or needle in str(row.get("source") or "").lower()
            or needle in str(row.get("source_path") or "").lower()
        ]
    return filtered


def build_table_rows(rows: list[dict], show_full_text: bool) -> list[dict]:
    return [
        {
            "source": row.get("source"),
            "source_type": row.get("source_type"),
            "tag": row.get("tag"),
            "page": row.get("page_start"),
            "page_chunk": row.get("page_chunk_index"),
            "tokens": row.get("token_count"),
            "parser_model": row.get("parser_model"),
            "source_path": row.get("source_path"),
            "text": row.get("text") if show_full_text else shorten(row.get("text", "")),
        }
        for row in rows
    ]


def main() -> None:
    settings = get_settings()
    db_path = Path(settings.lancedb_uri)
    st.set_page_config(page_title="LanceDB Viewer", layout="wide")
    st.title("LanceDB Viewer")
    st.caption(f"Base de datos: {db_path}")
    if not db_path.exists():
        st.error(f"No existe la ruta de LanceDB: {db_path}")
        st.stop()

    tables = LanceDBStore(db_path, settings.lancedb_table).list_tables()
    if not tables:
        st.warning("No hay tablas. Ejecuta primero el indexado.")
        st.stop()
    default = tables.index(settings.lancedb_table) if settings.lancedb_table in tables else 0
    selected_table = st.sidebar.selectbox("Tabla", tables, index=default)
    show_full = st.sidebar.checkbox("Mostrar texto completo", value=False)
    page_size = st.sidebar.slider("Chunks por pantalla", 10, 200, 25, 5)
    text_query = st.sidebar.text_input("Buscar en texto / source / path")

    rows = LanceDBStore(db_path, selected_table).list_chunks(include_vector=False)
    sources = sorted({row.get("source") for row in rows if row.get("source")})
    source_types = sorted({row.get("source_type") for row in rows if row.get("source_type")})
    tags = sorted({row.get("tag") for row in rows if row.get("tag")})
    filtered = apply_filters(
        rows,
        st.sidebar.multiselect("Source", sources),
        st.sidebar.multiselect("Tipo", source_types),
        st.sidebar.multiselect("Tag", tags),
        text_query,
    )

    total_pages = max(1, (len(filtered) + page_size - 1) // page_size)
    page = st.sidebar.number_input("Página", 1, total_pages, 1, 1)
    start = (page - 1) * page_size
    st.dataframe(
        build_table_rows(filtered[start : start + page_size], show_full),
        width="stretch",
        hide_index=True,
    )

    st.subheader("Detalle por documento")
    if not filtered:
        st.info("No hay chunks para los filtros actuales.")
        return
    selected = st.selectbox("Source", sorted({row["source"] for row in filtered}))
    for row in [item for item in filtered if item["source"] == selected]:
        title = f"p.{row['page_start']} · chunk de página {row.get('page_chunk_index', 0)}"
        with st.expander(title):
            st.write(f"**source_path:** {row['source_path']}")
            st.write(f"**parser_model:** {row.get('parser_model', '')}")
            st.write(f"**tag:** {row.get('tag') or ''}")
            st.code(row.get("text", ""), language=None)


if __name__ == "__main__":
    main()
