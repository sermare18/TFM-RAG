# RAG Cliente

RAG local para indexar PDF, DOCX, PPTX, XLSX, EPUB, HTML, TXT e imágenes en
LanceDB y consultarlos mediante CLI o una API compatible con aplicaciones web.
Los documentos estructurados se procesan con Marker 2; PyTorch y Marker están
configurados para una GPU NVIDIA.

## Requisitos

- Windows
- Conda (Miniconda o Anaconda)
- GPU NVIDIA y driver compatible con CUDA 13
- Un endpoint de chat compatible con OpenAI
- Un endpoint de embeddings compatible con OpenAI

## Instalación

Desde esta carpeta:

```powershell
.\setup.ps1
.\rag.bat gpu
```

`setup.ps1` crea o actualiza el entorno `rag-cliente` con Python 3.11, instala
PyTorch 2.12.1 con CUDA 13.0, instala las dependencias y el paquete local, y
comprueba que CUDA sea visible. La instalación falla si PyTorch no detecta la
GPU; no se usa CPU como fallback.

## Configuración

Copia `.env.example` como `.env` y ajusta, como mínimo:

```env
LLAMA_CPP_CHAT_BASE_URL=http://127.0.0.1:8081/v1
LLAMA_CPP_EMBEDDING_BASE_URL=http://127.0.0.1:8082/v1
OPENAI_API_KEY=local-dev-key
MARKER_TORCH_DEVICE=cuda
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

Marker 2 se controla con `MARKER_ENABLED`, `MARKER_MODE`, `MARKER_OCR_MODE`,
`MARKER_STRIP_EXISTING_OCR`,
`MARKER_USE_LLM`, `MARKER_DISABLE_IMAGE_EXTRACTION` y `MARKER_PAGE_RANGE`.

La configuración recomendada para GPU es:

```env
MARKER_MODE=balanced
MARKER_OCR_MODE=adaptive
MARKER_INFERENCE_BACKEND=auto
MARKER_LLAMA_CPP_BINARY=
MARKER_FULL_PAGE_OCR_COMPLEX_LAYOUT=true
MARKER_TABLE_MIN_RECON_SCORE=0.75
```

En este modo Marker 2 conserva la capa digital cuando es fiable y activa el
procesamiento visual de forma selectiva para páginas escaneadas, bloques
defectuosos y tablas de baja confianza. Los valores `force` y `disabled` quedan
como overrides de diagnóstico; el funcionamiento normal usa `adaptive`.

El modo `balanced` necesita un servidor VLM de Surya. En Linux/CUDA, `auto`
puede arrancar vLLM mediante Docker. En Windows se puede evitar Docker usando:

```env
MARKER_INFERENCE_BACKEND=llamacpp
MARKER_LLAMA_CPP_BINARY=D:\ruta\a\llama-server.exe
```

La primera ejecución descarga el modelo GGUF de Surya 2 a la caché del usuario.
Si se selecciona `vllm`, Docker Desktop debe estar iniciado antes de indexar.

`MARKER_FULL_PAGE_OCR_COMPLEX_LAYOUT=true` conserva `pdftext` en páginas
sencillas y promueve a OCR completo solo las páginas donde el layout detecta
`Table`, `Form` o `ComplexRegion`. El OCR recibe así el contexto global que
necesita para ordenar columnas paralelas. `MARKER_TABLE_MIN_RECON_SCORE=0.75`
queda como fallback secundario del procesador de tablas de Marker 2.

## Uso

Coloca los documentos en `data/pdfs`. Se admiten PDF, DOCX, PPTX, XLSX, EPUB,
HTML, TXT, PNG, JPG, JPEG, BMP, TIF, TIFF y WEBP. Las subcarpetas se usan como
etiquetas; por ejemplo, `data/pdfs/confidencial/contrato.pdf` recibe el tag
`confidencial`.

```powershell
# Comprobar la GPU
.\rag.bat gpu

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

Si CUDA no está disponible, ejecuta otra vez `setup.ps1` y después
`rag.bat gpu`. Con `MARKER_TORCH_DEVICE=cuda`, el indexador se detiene con un
mensaje explícito si PyTorch no detecta la GPU.

Si LanceDB indica que no existe `pdf_chunks`, añade documentos y ejecuta
`rag.bat index`. Si un endpoint devuelve timeout, aumenta `REQUEST_TIMEOUT` en
`.env` y comprueba que los servidores de chat y embeddings estén accesibles.
