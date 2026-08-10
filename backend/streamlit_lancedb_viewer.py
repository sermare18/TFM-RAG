from __future__ import annotations

from pathlib import Path

import streamlit as st

from rag_cliente.config import get_settings
from rag_cliente.vector_store import LanceDBStore


def shorten(text: str, max_chars: int = 220) -> str:
    compact = " ".join(text.split())
    if len(compact) <= max_chars:
        return compact
    return compact[:max_chars] + "..."


def apply_filters(
    rows: list[dict],
    source_filter: list[str],
    source_type_filter: list[str],
    tag_filter: list[str],
    text_query: str,
    ocr_filter: str = "Todos",
) -> list[dict]:
    """
    Esta función recibe:
        rows: lista de chunks
        source_filter: lista de nombres de source seleccionados
        source_type_filter: lista de tipos seleccionados
        text_query: texto de búsqueda escrito por el usuario
    
    Devuelve una lista de diccionarios filtrada
    """

    # Seleccionamos al principio todos los registros
    filtered = rows

    # Si el usuario selecciona algún source, nos quedamos solo con las filas cuyo campo source esté dentro de esa lista
    if source_filter:
        filtered = [row for row in filtered if row.get("source") in source_filter]

    # Igual que antes, pero filtrando por source_type
    if source_type_filter:
        filtered = [row for row in filtered if row.get("source_type") in source_type_filter]

    if tag_filter:
        filtered = [row for row in filtered if row.get("tag") in tag_filter]

    # Distingo False de un campo ausente para no presentar un índice antiguo como si hubiera evitado el OCR.
    if ocr_filter == "Con OCR":
        filtered = [row for row in filtered if row.get("ocr_used") == True]
    elif ocr_filter == "Sin OCR":
        filtered = [row for row in filtered if row.get("ocr_used") == False]

    # Comprueba si el usuario escribió algo útil
    # Elimina esacios al principio y al final
    if text_query.strip():
        # Prepara el texto de búsqueda: sin espacios sobrantes, en minúsculas (case-insensitive)
        needle = text_query.strip().lower()
        # La fila se conserva si needle aparece en alguno de estos campos: text, source, source_path
        filtered = [
            row
            for row in filtered
            if needle in (row.get("text") or "").lower()
            or needle in (row.get("source") or "").lower()
            or needle in (row.get("source_path") or "").lower()
        ]

    return filtered


def build_table_rows(rows: list[dict], show_full_text: bool) -> list[dict]:
    """
    Esta función transforma los chunks en una estructura más cómoda para enseñar en la tabla visual

    Recibe:
        rows: filas originales
        show_full_text: flag booleana que decide si mostrar el texto completo o solo preview
    """

    # Inicializamos la lista que vamos a devolver
    table_rows: list[dict] = []

    # Recorremos cada chunk
    for row in rows:
        ocr_used = row.get("ocr_used")
        ocr_label = "Sí" if ocr_used is True else "No" if ocr_used is False else "No registrado"
        table_rows.append(
            {
                "source": row.get("source"),
                "source_type": row.get("source_type"),
                "ocr_used": ocr_label,
                "tag": row.get("tag"),
                "page_start": row.get("page_start"),
                "page_end": row.get("page_end"),
                "chunk_index": row.get("chunk_index"),
                "source_path": row.get("source_path"),
                "text_preview": row.get("text") if show_full_text else shorten(row.get("text", "")),
            }
        )

    return table_rows


def main() -> None:

    # Carga la configuración del proyecto (p.ej., ruta de LanceDB, nombre de la tabla por defecto, etc)
    settings = get_settings()
    db_path = Path(settings.lancedb_uri)

    # CONFIGURACIÓN PÁGINA STREAMLIT
    # Título de la pestaña del navegador y layout ancho
    st.set_page_config(page_title="LanceDB Viewer", layout="wide")
    # Pinta el título principal de la página
    st.title("LanceDB Viewer")
    # Pinta una línea pequeña debajo del título mostrando la ruta de la DB
    st.caption(f"Base de datos: {db_path}")

    # Comprueba si la carpeta LanceDB existe
    if not db_path.exists():
        # Muestra error
        st.error(f"No existe la ruta de LanceDB: {db_path}")
        # Detiene la ejecución de la app
        st.stop()

    # Creamos una instancia temporal del store
    bootstrap_store = LanceDBStore(db_path, settings.lancedb_table)
    # Leer todas las tablas disponibles en LanceDB
    tables = bootstrap_store.list_tables()

    if not tables:
        st.warning("No hay tablas en LanceDB todavía. Ejecuta primero el indexado.")
        st.stop()

    # Decide qué tabla sale seleccionada por defecto en el desplegable; si la tabla configurada en .env existe, usamos su índice; si no existe, seleccionamos la primera (0)
    default_index = tables.index(settings.lancedb_table) if settings.lancedb_table in tables else 0

    # Muestra en la barra lateral un desplegable para elegir tabla (opciones: todas la tablas existentes, seleccionada por defecto: default_index)
    selected_table = st.sidebar.selectbox("Tabla", tables, index=default_index)
    # Checkbox para elegir si ver el texto completo o preview; por defecto está desmarcado
    show_full_text = st.sidebar.checkbox("Mostrar texto completo", value=False)
    # Slider para decidir cuántos chunks enseñar por página
    page_size = st.sidebar.slider("Chunks por página", min_value=10, max_value=200, value=25, step=5)
    # Caja de texto para hacer la búsqueda libre
    text_query = st.sidebar.text_input("Buscar en texto / source / path")

    # Creo store real usando la tabla elegida por el usuario
    store = LanceDBStore(db_path, selected_table)
    # Lee los chunks de esa tabla (sin incluir el embedding)
    rows = store.list_chunks(include_vector=False)

    # Construimos las opciones posibles para el filtro de source
    source_options = sorted({row.get("source") for row in rows if row.get("source")})
    # Construimos las opciones posibles para el filtro de source_type
    source_type_options = sorted({row.get("source_type") for row in rows if row.get("source_type")})
    tag_options = sorted({row.get("tag") for row in rows if row.get("tag")})

    # Permite seleccionar varios source a la vez
    selected_sources = st.sidebar.multiselect("Filtrar por source", source_options)
    # Permite seleccionar varios source_type a la vez
    selected_source_types = st.sidebar.multiselect("Filtrar por source_type", source_type_options)
    selected_tags = st.sidebar.multiselect("Filtrar por tag", tag_options)
    # Ofrezco un único selector para poder aislar rápidamente los chunks procesados con OCR.
    selected_ocr = st.sidebar.selectbox("Filtrar por OCR", ["Todos", "Con OCR", "Sin OCR"])

    # Aplica los filtros
    filtered_rows = apply_filters(
        rows=rows,
        source_filter=selected_sources,
        source_type_filter=selected_source_types,
        tag_filter=selected_tags,
        text_query=text_query,
        ocr_filter=selected_ocr,
    )

    # Número total de chunks tras filtrar
    total_rows = len(filtered_rows)
    # Calcula cúantas páginas hacen falta
    total_pages = max(1, (total_rows + page_size - 1) // page_size)

    # Input numérico para cambiar de página
    page_number = st.sidebar.number_input(
        "Página",
        min_value=1,
        max_value=total_pages,
        value=1,
        step=1,
    )

    current_page = min(page_number, total_pages)

    # Calcula el rango de índices para hacer el "slice" de la página actual
    # Ejemplo: página 2, tamaño 25 --> start=25; end=50
    start = (current_page - 1) * page_size
    end = start + page_size

    # Nos quedamos solo con los registros de esa página
    page_rows = filtered_rows[start:end]

    # Creamos tres columnas
    col1, col2, col3 = st.columns(3)
    col1.metric("Tabla", selected_table)
    col2.metric("Chunks filtrados", total_rows)
    col3.metric("Página", f"{current_page}/{total_pages}")

    # Pone el subtítulo de la sección
    st.subheader("Vista tabular")
    # Muestra un dataframe visual en Streamlit
    st.dataframe(
        build_table_rows(page_rows, show_full_text=show_full_text),
        # Adapto la tabla al ancho disponible con la API vigente de Streamlit.
        width="stretch",
        hide_index=True,
    )

    st.subheader("Detalle por source")

    if not filtered_rows:
        st.info("No hay chunks para los filtros actuales.")
        return

    detail_source_options = sorted({row.get("source") for row in filtered_rows if row.get("source")})
    selected_detail_source = st.selectbox("Selecciona un source", detail_source_options)

    source_rows = [row for row in filtered_rows if row.get("source") == selected_detail_source]

    st.caption(f"{len(source_rows)} chunks para {selected_detail_source}")

    for row in source_rows:
        title = f"chunk {row.get('chunk_index')} · págs {row.get('page_start')}-{row.get('page_end')}"
        with st.expander(title):
            st.write(f"**source_path:** {row.get('source_path')}")
            st.write(f"**source_type:** {row.get('source_type')}")
            ocr_used = row.get("ocr_used")
            ocr_label = "Sí" if ocr_used is True else "No" if ocr_used is False else "No registrado"
            st.write(f"**ocr_used:** {ocr_label}")
            st.write(f"**tag:** {row.get('tag') or ''}")
            st.code(row.get("text", ""), language=None)

if __name__ == "__main__":
    main()
