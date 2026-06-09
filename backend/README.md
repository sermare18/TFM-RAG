# RAG Cliente

RAG local para indexar documentos (`PDF`, `DOCX`, `TXT` e imágenes) en LanceDB y consultarlos con endpoints compatibles con OpenAI.

Desde esta versión, los PDFs se procesan con **Marker** como parser/OCR principal. El OCR anterior basado en PaddleX/PaddleOCR se ha eliminado de las dependencias.

## Requisitos

- Python 3.10+
- Conda
- Endpoints de **chat** y **embeddings** compatibles con OpenAI
- PyTorch compatible con tu equipo, ya sea CPU o GPU, para ejecutar Marker

## Instalación

```powershell
create_conda_env.bat
conda activate rag-cliente
python -m pip install -r requirements.txt
python -m pip install -e .
```

`requirements.txt` instala `marker-pdf`. Si quieres forzar GPU o CPU, ajusta `MARKER_TORCH_DEVICE` en `.env`:

```env
MARKER_TORCH_DEVICE=cuda
```

Valores habituales:

- `cuda`: GPU NVIDIA con CUDA disponible
- `cpu`: CPU
- `mps`: Apple Silicon
- vacío: Marker/Torch detecta el dispositivo automáticamente

## Configuración

Crea un `.env` en la raíz. Puedes partir de `.env.example`.

Ejemplo mínimo:

```env
# Endpoints
LLAMA_CPP_CHAT_BASE_URL=http://10.31.2.6:8080/v1
LLAMA_CPP_EMBEDDING_BASE_URL=http://10.31.2.6:8080/v1
OPENAI_API_KEY=local-dev-key

# LanceDB
LANCEDB_URI=./data/lancedb
LANCEDB_TABLE=pdf_chunks

# Parámetros de RAG
TOP_K=4
CHUNK_SIZE=700
CHUNK_OVERLAP=100
MAX_TOKENS=1024
REQUEST_TIMEOUT=300
EMBEDDING_BATCH_SIZE=16
DATA_DIR=./data
DOCUMENTS_DIR=./data/pdfs

# Marker
MARKER_ENABLED=true
MARKER_FORCE_OCR=false
MARKER_STRIP_EXISTING_OCR=false
MARKER_USE_LLM=false
MARKER_DISABLE_IMAGE_EXTRACTION=true
MARKER_PAGE_RANGE=
MARKER_TORCH_DEVICE=
```

### Variables Marker

| Variable | Uso |
| --- | --- |
| `MARKER_ENABLED` | Activa Marker para PDFs e imágenes. Si es `false`, los PDFs digitales se extraen con PyMuPDF como fallback y las imágenes no se indexan. |
| `MARKER_FORCE_OCR` | Fuerza OCR visual incluso si el PDF tiene texto embebido. Útil si el texto nativo está corrupto o quieres priorizar layout/tablas. |
| `MARKER_STRIP_EXISTING_OCR` | Elimina una capa OCR existente antes de re-OCR. Útil con PDFs escaneados que traen OCR malo o duplicado. |
| `MARKER_USE_LLM` | Activa el modo LLM de Marker para mejorar tablas/formularios/formato. Requiere configurar un backend LLM soportado por Marker. |
| `MARKER_DISABLE_IMAGE_EXTRACTION` | Evita guardar imágenes extraídas en disco. Recomendado para RAG textual. |
| `MARKER_PAGE_RANGE` | Rango opcional de páginas en sintaxis Marker, por ejemplo `0,5-10,20`. Vacío procesa todo. |
| `MARKER_TORCH_DEVICE` | Dispositivo opcional para Torch/Marker: `cpu`, `cuda` o `mps`. |

### Variables OCR antiguas obsoletas

Estas variables de PaddleX/PaddleOCR ya no controlan el procesamiento:

```env
ENABLE_OCR
OCR_PIPELINE_NAME
OCR_DEVICE
OCR_FORCE_FOR_PDF
OCR_MIN_NATIVE_CHARS
OCR_MIN_NATIVE_CHARS_PER_PAGE
OCR_RENDER_DPI
OCR_USE_TABLE_RECOGNITION
OCR_USE_FORMULA_RECOGNITION
OCR_USE_REGION_DETECTION
OCR_FORMAT_BLOCK_CONTENT
```

`config.py` las sigue aceptando como obsoletas para que un `.env` antiguo no rompa la carga de configuración, pero `pdf_loader.py` ya no las usa.

## Estructura mínima

```text
api-python/
├─ data/
│  ├─ pdfs/
│  └─ lancedb/
├─ src/rag_cliente/
├─ .env
├─ requirements.txt
├─ pyproject.toml
├─ run_rag_index.bat
├─ run_rag_ask.bat
└─ run_lancedb_viewer.bat
```

## Uso

### 1. Preparar documentos

Coloca tus archivos en:

```text
data/pdfs
```

Formatos soportados:

- `.pdf`: procesado con Marker por defecto
- `.docx`: comportamiento existente con `python-docx`
- `.txt`: comportamiento existente con lectura UTF-8
- Imágenes: `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif`, `.tiff`, `.webp`, procesadas con Marker si `MARKER_ENABLED=true`

Puedes organizar documentos por etiqueta usando subcarpetas:

```text
data/pdfs/confidencial/archivo1.pdf
data/pdfs/publico/archivo2.pdf
data/pdfs/archivo3.pdf
```

Al reindexar, el tag se deriva de la primera carpeta relativa: `confidencial`, `publico` o sin tag si el archivo está directamente en `data/pdfs`.

### 2. Indexar por terminal

```powershell
python -m rag_cliente.cli index --doc-dir data\pdfs
```

Para asignar una etiqueta de metadatos a todos los chunks de esa indexación:

```powershell
python -m rag_cliente.cli index --doc-dir data\pdfs --tag confidencial
```

Durante el indexado verás progreso similar a:

```text
Iniciando indexado en: data\pdfs
Inicializando Marker...
Procesando archivo 1/3: contrato.pdf
Parseando PDF con Marker: contrato.pdf
Extraídas 12 páginas/bloques de contrato.pdf (OCR usado en 3)
Chunking de documentos...
Chunks generados: 84
Generando embeddings en lotes de 16...
Embeddings lote 1/6 (16 textos)
Guardando 84 chunks en LanceDB...
Indexado completado en tabla 'pdf_chunks'.
```

También puedes usar el script:

```powershell
.\run_rag_index.bat
```

### 3. Preguntar por terminal

```powershell
python -m rag_cliente.cli ask "Como crear un correo electrónico"
```

Por defecto, el CLI **no imprime reasoning**, aunque el backend lo devuelva.

### 4. Mostrar reasoning explícitamente

```powershell
python -m rag_cliente.cli ask "Resume el documento" --show-reasoning
```

Con streaming:

```powershell
python -m rag_cliente.cli ask "Resume el documento" --stream --show-reasoning
```

### 5. Ajustar recuperación

```powershell
python -m rag_cliente.cli ask "Resume el documento" --top-k 4
```

Para limitar la recuperación a chunks con una etiqueta concreta:

```powershell
python -m rag_cliente.cli ask "Resume el contrato" --top-k 8 --tag confidencial
```

## API HTTP

El proyecto puede exponerse con FastAPI reutilizando el mismo pipeline:

```powershell
python -m pip install -r requirements.txt
python -m pip install -e .
rag-api
```

O bien:

```powershell
uvicorn rag_cliente.api:app --host 0.0.0.0 --port 8000
```

Endpoints principales:

- `GET /health`
- `GET /files`
- `POST /files/upload`
- `POST /sessions`
- `DELETE /sessions/{session_id}`
- `POST /index`
- `POST /ask`
- `POST /ask/stream`

Documentación interactiva:

- `http://localhost:8000/docs`
- `http://localhost:8000/redoc`

Ejemplos:

```powershell
curl http://localhost:8000/health
```

```powershell
curl -X POST http://localhost:8000/index `
  -H "Content-Type: application/json" `
  -d "{\"doc_dir\":\"data/pdfs\"}"
```

Con tag:

```powershell
curl -X POST http://localhost:8000/index `
  -H "Content-Type: application/json" `
  -d "{\"doc_dir\":\"data/pdfs\",\"tag\":\"confidencial\"}"
```

```powershell
curl -X POST http://localhost:8000/ask `
  -H "Content-Type: application/json" `
  -d "{\"question\":\"¿Qué dice el documento sobre la renovación?\",\"top_k\":2}"
```

Con filtro por tag:

```powershell
curl -X POST http://localhost:8000/ask `
  -H "Content-Type: application/json" `
  -d "{\"question\":\"Resume el contrato\",\"top_k\":8,\"tag\":\"confidencial\"}"
```

También se acepta `tags` con una lista y se usará la primera etiqueta no vacía. Si omites `tag`/`tags`, la búsqueda se hace sobre todo el índice como antes. Las tablas LanceDB ya creadas antes de este cambio no tienen la columna `tag`; para filtrar por tag debes reindexar los documentos.

`POST /files/upload` también acepta `tag` como campo `multipart/form-data`. Si envías `tag=confidencial`, el archivo se guarda en `data/pdfs/confidencial/<archivo>`; si no envías tag, se guarda directamente en `data/pdfs/<archivo>`.

## Visor local de LanceDB

```powershell
.\run_lancedb_viewer.bat
```

Permite:

- ver tablas
- inspeccionar chunks
- ver las columnas `ocr_used` y `tag`
- filtrar por `source`, `source_type` y `tag`
- buscar por texto

### Metadatos en LanceDB

Cada chunk guardado en LanceDB incluye ahora estas columnas de metadatos:

```text
ocr_used
tag
```

Valores:

- `true`: Marker usó OCR/Surya para la página o unidad de origen del chunk.
- `false`: el texto procede de extracción nativa del PDF (`pdftext`) o de formatos no OCR como DOCX/TXT.
- `tag`: etiqueta opcional asignada al indexar, por ejemplo `confidencial`.

La columna `ocr_used` se obtiene de `metadata["page_stats"][].text_extraction_method` devuelto por Marker. No se calcula por extensión de archivo ni por una heurística propia del proyecto.

## Scripts

- `create_conda_env.bat`: crea el entorno Conda
- `activate_conda_env.bat`: activa el entorno
- `open_rag_terminal.bat`: abre terminal preparada
- `run_rag_index.bat`: indexa documentos
- `run_rag_ask.bat`: lanza una pregunta
- `run_lancedb_viewer.bat`: abre el visor local

## Problemas habituales

### Falta Marker

```powershell
python -m pip install marker-pdf
```

O reinstala todas las dependencias:

```powershell
python -m pip install -r requirements.txt
```

### CUDA/GPU no se usa

Comprueba que PyTorch detecta CUDA en tu entorno y fuerza el dispositivo:

```env
MARKER_TORCH_DEVICE=cuda
```

### `Table 'pdf_chunks' was not found`

Todavía no has indexado o no existe la carpeta de LanceDB:

```powershell
python -m rag_cliente.cli index --doc-dir data\pdfs
```

### `openai.APITimeoutError`

El endpoint tarda demasiado. Sube el timeout en `.env`:

```env
REQUEST_TIMEOUT=300
```

### Un PDF concreto falla al indexar

El indexador muestra un aviso y continúa con el resto de archivos. Revisa el mensaje `AVISO: no se pudo procesar ...` para ver la causa.
