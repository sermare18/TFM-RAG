"""Paquete base del proyecto RAG Cliente.

Este paquete agrupa las piezas principales del sistema:

- configuración (`config.py`)
- conversion PDF/Markdown y cache Bedrock (`bedrock_parser.py`)
- chunking (`indexer.py`)
- comunicación con endpoints compatibles con OpenAI (`llm_client.py`)
- supervisor de procesos llama.cpp (`model_supervisor.py`)
- manifiesto/validación de GGUF (`model_manifest.py`)
- coordinación FIFO de memoria (`resource_coordinator.py`)
- almacenamiento vectorial en LanceDB (`vector_store.py`)
- orquestación del flujo RAG (`pipeline.py`)
- interfaz de línea de comandos (`cli.py`)
- API HTTP (`api.py`)

Los modelos son siempre locales y los procesos administrados solo pueden ser
detenidos por el supervisor que registró su PID.
"""
