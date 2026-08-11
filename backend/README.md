# TFM RAG: Bedrock para documentos y modelos locales para consulta

El proyecto convierte documentos a Markdown con un VLM de Amazon Bedrock y
mantiene en local las dos piezas del RAG que se usan repetidamente: embeddings
y generación de respuestas mediante `llama.cpp`.

## Arquitectura

1. PyMuPDF renderiza cada página del PDF y obtiene texto digital auxiliar.
2. Bedrock recibe cuatro páginas consecutivas por llamada. Las imágenes son la
   fuente autoritativa; el texto extraído se etiqueta como referencia no fiable.
3. La respuesta JSON contiene Markdown separado por página.
4. El Markdown y un manifiesto con hash, modelo y prompt quedan en
   `data/markdown`. Cada lote completado se guarda inmediatamente, por lo que
   una cuota o corte posterior no obliga a pagar otra vez por esas páginas.
5. El chunking nunca cruza una página. Una página grande puede producir varios
   chunks con el mismo número de página.
6. LanceDB guarda los vectores y BM25 el índice léxico. La recuperación admite
   `vector`, `bm25` o `hybrid` mediante RRF.
7. El modelo local de chat redacta la respuesta y devuelve citas con página.

Los `.md` entran directamente. Si contienen separadores `<!-- PAGE N -->`, se
conserva la paginación; sin separadores se consideran una sola página.

## Configuración de Bedrock

`.env.example` deja Bedrock desactivado para que una instalación nueva nunca
genere coste por accidente. Las credenciales se guardan en el perfil estándar
de AWS, nunca en `.env` ni en el repositorio:

```powershell
aws configure --profile rag-bedrock
```

Después se activa el perfil en `.env`:

```dotenv
BEDROCK_ENABLED=true
AWS_PROFILE=rag-bedrock
AWS_REGION=eu-west-1
BEDROCK_MODEL_ID=qwen.qwen3-vl-235b-a22b
BEDROCK_PAGES_PER_BATCH=4
BEDROCK_MAX_PAGES_PER_DOCUMENT=200
BEDROCK_MAX_BATCHES_PER_DOCUMENT=50
```

Mientras esté desactivado, los PDF solo se indexan si ya existe una caché
Markdown válida. Los Markdown directos sí se pueden indexar. Los dos límites
anteriores actúan como presupuesto duro por documento.

Las cuentas AWS nuevas pueden recibir cuotas diarias reducidas. El error
`Too many tokens per day` no indica un fallo del PDF: hay que esperar al reinicio
diario o solicitar un aumento en **Service Quotas > Amazon Bedrock**. La siguiente
ejecución reanuda el primer lote pendiente. Los modelos Anthropic requieren
además enviar una vez el formulario de caso de uso desde la consola de Bedrock.

## Preparación

```powershell
cd D:\master\RAG\backend
.\setup.ps1
Copy-Item .env.example .env   # solo si todavía no existe .env
.\rag.bat doctor
```

Los modelos locales nunca se descargan durante el indexado:

```powershell
.\rag.bat models plan gpu
.\rag.bat models download gpu
.\rag.bat models check --profile gpu
```

El perfil GPU conserva Qwen3 Embedding 0.6B y Qwen3.5 9B para respuesta. El
perfil CPU usa el mismo embedding y Qwen3 4B para respuesta.

## Uso

```powershell
# Indexa PDF y Markdown; reutiliza la caché Bedrock válida
.\rag.bat index data\pdfs

# Fuerza una nueva conversión de los PDF
.\rag.bat index data\pdfs --refresh-bedrock

# Consulta
.\rag.bat ask "¿Qué indica el documento?" --top-k 5

# API y visor
.\rag.bat api
.\rag.bat viewer
```

Cambiar `RETRIEVAL_MODE=vector|bm25|hybrid` y `RETRIEVAL_TOP_K` permite comparar
configuraciones sobre el mismo índice. El resultado recuperado se colapsa por
página, lo que deja preparado el cálculo posterior de precisión y recall.

## Verificación

```powershell
.\rag.bat test
```

Los tests de Bedrock usan un cliente falso: no acceden a AWS ni cargan modelos.
Al cambiar desde un índice antiguo hay que ejecutar de nuevo `index`, porque el
esquema actual guarda procedencia Bedrock y posición del chunk dentro de página.
