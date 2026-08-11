# TFM RAG: Bedrock para documentos y modelos locales para consulta

El proyecto convierte documentos a Markdown con un VLM de Amazon Bedrock y
mantiene en local las dos piezas del RAG que se usan repetidamente: embeddings
y generación de respuestas mediante `llama.cpp`.

## Arquitectura

1. PyMuPDF renderiza cada página del PDF y obtiene texto digital auxiliar.
2. Bedrock recibe cuatro páginas consecutivas por llamada. Las imágenes son la
   fuente autoritativa; el texto extraído se etiqueta como referencia no fiable.
3. Bedrock fuerza la respuesta mediante un esquema de salida estructurada. El
   JSON validado contiene Markdown separado por página.
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

### Relación entre páginas, lotes y chunks

Los lotes de cuatro páginas solo sirven para dar contexto visual a Bedrock
durante la conversión y no se almacenan como una unidad. El indexador mantiene
siempre la frontera de página: cada chunk pertenece a una única página y nunca
mezcla contenido de páginas distintas.

Una página corta genera normalmente un chunk. Una página larga puede generar
varios, con los valores predeterminados de 700 tokens objetivo, 900 tokens
máximos y 100 tokens de solapamiento. Una página vacía puede no generar ningún
chunk. Todos los chunks conservan su número de página en `source_pages`,
`page_start` y `page_end` para permitir recuperación, citas y evaluación por
página.

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
BEDROCK_MODEL_ID=global.anthropic.claude-sonnet-4-6
BEDROCK_PAGES_PER_BATCH=4
BEDROCK_MAX_OUTPUT_TOKENS=16384
BEDROCK_MAX_PAGES_PER_DOCUMENT=200
BEDROCK_MAX_BATCHES_PER_DOCUMENT=50
```

Mientras esté desactivado, los PDF solo se indexan si ya existe una caché
Markdown válida. Los Markdown directos sí se pueden indexar. Los dos límites
anteriores actúan como presupuesto duro por documento.

El identificador anterior es el perfil **Global cross-region** de Claude Sonnet
4.6, no el identificador del modelo base. Puede invocarse desde `eu-west-1` y
aprovecha la cuota global concedida a la cuenta. AWS puede enrutar el contenido
a cualquier región comercial; si se necesita residencia europea debe usarse el
perfil `eu.anthropic.claude-sonnet-4-6` y disponer de su cuota correspondiente.

Si se alcanza la cuota de tokens por minuto, la siguiente ejecución reanuda el
primer lote pendiente. Cada lote correcto ya queda persistido en la caché. Los
modelos Anthropic requieren además enviar una vez el formulario de caso de uso
desde la consola de Bedrock.

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

El perfil GPU usa Qwen3-Embedding-8B Q8_0 y Qwen3-14B Q5_K_M. El perfil CPU
conserva Qwen3-Embedding-0.6B Q8_0 y Qwen3-4B Q4_K_M. El servidor de embeddings
usa `pooling last` y añade a las consultas la instrucción configurada en
`EMBEDDING_QUERY_INSTRUCTION`; el contenido de los documentos se indexa sin esa
instrucción.

Al cambiar el modelo de embeddings hay que ejecutar de nuevo `index` para
recrear los vectores. La caché Markdown de Bedrock se reutiliza: no es necesario
usar `--refresh-bedrock` ni volver a pagar la conversión de los PDF.

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

Para la primera prueba conviene usar una carpeta que contenga un solo PDF:

```powershell
.\rag.bat doctor
.\rag.bat index data\prueba-bedrock --tag prueba
```

`doctor` no llama a AWS ni consume tokens. El segundo comando sí realiza las
llamadas necesarias y guarda el Markdown por página en `data/markdown`.

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
