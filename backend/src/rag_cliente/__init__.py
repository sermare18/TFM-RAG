"""Paquete base del proyecto RAG Cliente.

Este paquete agrupa las piezas principales del sistema:

- configuración (`config.py`)
- carga de documentos (`pdf_loader.py`)
- chunking (`indexer.py`)
- comunicación con endpoints compatibles con OpenAI (`llm_client.py`)
- almacenamiento vectorial en LanceDB (`vector_store.py`)
- orquestación del flujo RAG (`pipeline.py`)
- interfaz de línea de comandos (`cli.py`)
- API HTTP (`api.py`)

Limitaciones generales del paquete:
- El proyecto implementa un RAG sencillo, pensado como base didáctica o prototipo.
- No incluye re-ranking, búsqueda híbrida ni filtros por metadatos.
- La calidad final depende mucho del endpoint de embeddings y del endpoint de chat.
"""
