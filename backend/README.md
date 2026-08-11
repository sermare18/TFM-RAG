# RAG Cliente

RAG local para indexar PDF, DOCX, PPTX, XLSX, EPUB, HTML, TXT e imágenes en
LanceDB y consultarlos mediante CLI o una API compatible con aplicaciones web.
Los documentos estructurados se procesan con Marker 2 mediante perfiles para
CPU o GPU NVIDIA.

## Requisitos

- Windows
- Conda (Miniconda o Anaconda)
- GPU NVIDIA y driver compatible con CUDA 13 (opcional)
- `llama-server` local; la ruta predeterminada es
  `C:\Users\SergioMartinReizabal\Documents\llama.cpp\llama-server.exe`
- Modelos GGUF locales, administrados o expuestos como servidores `external`

## Instalación

Desde esta carpeta:

```powershell
.\setup.ps1 -Device auto  # también admite cpu o cuda
.\rag.bat gpu
```

`setup.ps1` crea o actualiza el entorno `rag-cliente` con Python 3.11, instala
explícitamente la distribución CPU o CUDA de PyTorch 2.12.1, instala las
dependencias y el paquete local, y valida que la variante no haya cambiado.
`auto` elige CUDA si `nvidia-smi` detecta una GPU utilizable y CPU en caso
contrario. `cpu` o `cuda` siempre prevalecen sobre esa detección.

## Configuración

Copia `.env.example` como `.env` y ajusta, como mínimo:

```env
LLAMA_CPP_CHAT_BASE_URL=http://127.0.0.1:8081/v1
LLAMA_CPP_EMBEDDING_BASE_URL=http://127.0.0.1:8082/v1
OPENAI_API_KEY=local-dev-key
MARKER_PROFILE=auto
MODEL_SUPERVISION_ENABLED=true
LLAMA_CPP_BINARY=C:\Users\SergioMartinReizabal\Documents\llama.cpp\llama-server.exe
MODELS_DIR=./models
```

Las opciones principales de almacenamiento y recuperación son:

```env
LANCEDB_URI=./data/lancedb
LANCEDB_TABLE=pdf_chunks
DOCUMENTS_DIR=./data/pdfs
TOP_K=4
CHUNK_SIZE=700
CHUNK_OVERLAP=100
HYBRID_SEARCH_ENABLED=true
VECTOR_WEIGHT=0.65
BM25_WEIGHT=0.35
```

Marker 2 se controla principalmente con `MARKER_ENABLED`, `MARKER_PROFILE`,
`MARKER_STRIP_EXISTING_OCR`, `MARKER_DISABLE_IMAGE_EXTRACTION` y
`MARKER_PAGE_RANGE`.

Los perfiles son:

- `cpu-digital`: `fast`, OCR deshabilitado y sin LLM.
- `cpu-quality`: `fast`, OCR Surya mediante llama.cpp y LLM habilitado.
- `gpu-quality`: `balanced`, OCR Surya con CUDA y LLM habilitado.
- `auto`: `gpu-quality` con CUDA utilizable y `cpu-quality` en caso contrario.

Marker usa salida JSON estructurada por defecto. Para conservar temporalmente
el Markdown paginado anterior se puede definir
`MARKER_MARKDOWN_COMPATIBILITY=true`. Los perfiles quality usan exclusivamente
el servicio OpenAI-compatible de Marker mediante el wrapper local presupuestado
del proyecto. `MARKER_OPENAI_BASE_URL` debe ser loopback, una IP privada o un
host incluido en `LOCAL_MODEL_HOSTS`; no existe fallback a Gemini ni a APIs
externas.

Los procesos `managed` son creados por el supervisor, que conserva sus PIDs,
espera `/health`, escribe logs en `MODEL_LOGS_DIR` y solo detiene esos procesos.
Un rol `external` nunca se inicia ni se detiene desde la aplicación. Parser,
embeddings y chat se serializan mediante leases FIFO: durante indexación no se
carga chat y embeddings se descarga antes de cargar chat.

Presupuestos iniciales relevantes:

```env
MODEL_START_TIMEOUT=180
MODEL_REQUEST_TIMEOUT=180
MODEL_STOP_TIMEOUT=15
PARSER_JOB_TIMEOUT=1800
MODEL_MAX_RETRIES=1
MARKER_LLM_MAX_REQUESTS=50
MARKER_LLM_MAX_TOKENS_PER_REQUEST=4096
MARKER_LLM_MAX_GENERATED_TOKENS_PER_DOCUMENT=20000
MARKER_LLM_REQUEST_TIMEOUT=180
MARKER_LLM_JOB_TIMEOUT=1800
MARKER_LLM_MAX_RETRIES=1
MARKER_LLM_FALLBACK_TO_BASE=false
```

## Uso

Coloca los documentos en `data/pdfs`. Se admiten PDF, DOCX, PPTX, XLSX, EPUB,
HTML, TXT, PNG, JPG, JPEG, BMP, TIF, TIFF y WEBP. Las subcarpetas se usan como
etiquetas; por ejemplo, `data/pdfs/confidencial/contrato.pdf` recibe el tag
`confidencial`.

```powershell
# Comprobar la GPU
.\rag.bat gpu

# Diagnóstico sin arrancar modelos
.\rag.bat doctor

# Ver planes y tamaños sin descargar
.\rag.bat models plan cpu
.\rag.bat models plan gpu

# Descargar solo por petición explícita
.\rag.bat models download cpu
.\rag.bat models download gpu

# Validar cabeceras GGUF, mmproj y rutas sin cargar modelos
.\rag.bat models check

# Indexar data/pdfs
.\rag.bat index

# Indexar otra carpeta
.\rag.bat index "D:\documentos"

# Consultar
.\rag.bat ask "Resume el contrato"

# Activar thinking solo para esta consulta y mostrar el razonamiento
.\rag.bat ask "Analiza las alternativas" --stream --show-reasoning

# Arrancar la API en el puerto 8000
.\rag.bat api

# Arrancar la API en otro puerto
.\rag.bat api 8088

# Abrir el visor de LanceDB
.\rag.bat viewer

# Ejecutar los tests
.\rag.bat test
```

La API se ejecuta en primer plano y se detiene con `Ctrl+C`. Swagger queda
disponible en `http://localhost:8000/docs`.

`models plan`, `models check` y `doctor` nunca descargan ni arrancan modelos.
`models download` usa Hugging Face Hub únicamente cuando el usuario ejecuta el
comando y deja los artefactos bajo `MODELS_DIR`; tampoco arranca `llama-server`.

## API HTTP

Endpoints principales:

- `GET /health`
- `GET /files`
- `GET /files/{path}`
- `POST /files/upload`
- `POST /sessions`
- `DELETE /sessions/{session_id}`
- `POST /index`
- `POST /ask`
- `POST /ask/stream`

Ejemplos:

```powershell
curl http://localhost:8000/health

curl -X POST http://localhost:8000/index `
  -H "Content-Type: application/json" `
  -d '{"doc_dir":"data/pdfs"}'

curl -X POST http://localhost:8000/ask `
  -H "Content-Type: application/json" `
  -d '{"question":"Resume el documento","top_k":4}'
```

`POST /files/upload` admite un campo multipart `tag`. `/ask` acepta `tag` o
`tags`; si se envían varias etiquetas, se usa la primera no vacía.
El razonamiento está desactivado por defecto; envía `"enable_reasoning": true`
en `/ask` o `/ask/stream` para activarlo en una petición concreta.
En streaming, el reasoning se emite token a token hasta
`REASONING_MAX_TOKENS` y la respuesta final continúa en el mismo flujo también
token a token. Esto evita que Qwen3.5 consuma indefinidamente toda la generación
sin llegar a producir una respuesta.

## Estructura

```text
backend/
├─ data/
│  ├─ pdfs/
│  ├─ lancedb/       # generado
│  └─ bm25/          # generado
├─ src/rag_cliente/
├─ tests/
├─ .env
├─ requirements.txt
├─ pyproject.toml
├─ setup.ps1
└─ rag.bat
```

## Diagnóstico

Si CUDA no está disponible, ejecuta `setup.ps1 -Device cpu`. Si seleccionas
`MARKER_PROFILE=gpu-quality`, el indexador se detiene con un mensaje explícito
si PyTorch no detecta la GPU; nunca cambia silenciosamente a CPU.

Si LanceDB indica que no existe `pdf_chunks`, añade documentos y ejecuta
`rag.bat index`. Si un endpoint devuelve timeout, aumenta `MODEL_REQUEST_TIMEOUT` en
`.env` y comprueba que los servidores de chat y embeddings estén accesibles.
