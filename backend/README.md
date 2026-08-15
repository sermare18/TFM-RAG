# TFM RAG: Bedrock para documentos y modelos locales para consulta

El proyecto convierte documentos a Markdown con un VLM de Amazon Bedrock y
mantiene en local las dos piezas del RAG que se usan repetidamente: embeddings
y generación de respuestas mediante `llama.cpp`.

## Arquitectura

1. PyMuPDF renderiza cada página del PDF y obtiene texto digital auxiliar.
2. Cada llamada a Bedrock extrae una sola página objetivo, pero muestra al VLM
   una ventana deslizante de hasta cuatro páginas para conservar el contexto de
   tablas y estructuras multipágina. Solo la imagen objetivo es transcribible.
3. Bedrock fuerza la respuesta mediante un esquema de salida estructurada con
   un único campo `markdown`. El programa asigna ese resultado a la página
   objetivo; el modelo no elige ni devuelve el número de página.
4. El Markdown y un manifiesto con hash, modelo y prompt quedan en
   `data/markdown`. Cada página completada se guarda inmediatamente, por lo que
   una cuota o corte posterior no obliga a pagar otra vez por ella.
5. El chunking nunca cruza una página. Una página grande puede producir varios
   chunks con el mismo número de página.
6. LanceDB guarda los vectores y BM25 el índice léxico. La recuperación admite
   `vector`, `bm25` o `hybrid` mediante RRF.
7. El modelo local de chat redacta la respuesta y devuelve citas con página.

Los `.md` entran directamente. Si contienen separadores `<!-- PAGE N -->`, se
conserva la paginación; sin separadores se consideran una sola página.

### Relación entre páginas, contexto y chunks

La ventana de cuatro páginas solo da contexto visual a Bedrock y no se almacena
como una unidad. En cada llamada una página está marcada como objetivo y las
demás como contexto no transcribible. El texto digital auxiliar se envía solo
para la página objetivo. El indexador mantiene siempre la frontera de página:
cada chunk pertenece a una única página y nunca mezcla contenido de páginas
distintas.

Siempre que los límites del documento lo permiten, la ventana coloca la página
objetivo en tercera posición: dos páginas anteriores, la objetivo y una posterior.
Esto prioriza los encabezados y el inicio de las tablas que continúan desde
páginas anteriores.

Este contrato evita el desplazamiento entre `slots` observado cuando una sola
respuesta contenía cuatro páginas. Las páginas vecinas pueden ayudar a recuperar
los encabezados o la estructura de una tabla continuada, pero el prompt prohíbe
copiar sus filas o su texto al resultado objetivo.

Antes de guardar una página, el parser exige que Bedrock termine con
`stopReason=end_turn`, rechaza Markdown con estructuras abiertas y, cuando el
PDF contiene texto digital suficiente, comprueba que no falte una parte grande
respecto al texto auxiliar. Una salida incompleta se repite una sola vez; si el
segundo intento también falla, la página queda registrada como fallida y vacía,
no genera chunks y la extracción continúa con las siguientes. El segundo intento
recibe solo la imagen objetivo y su texto auxiliar para evitar que las páginas
vecinas vuelvan a distraer al modelo. Los errores de cuota, autenticación o
servicio sí detienen el proceso, ya que afectarían también al resto de páginas.

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
BEDROCK_CONTEXT_PAGES=4
BEDROCK_MAX_OUTPUT_TOKENS=16384
BEDROCK_MAX_PAGES_PER_DOCUMENT=200
BEDROCK_MAX_CALLS_PER_DOCUMENT=200
BEDROCK_TRANSIENT_MAX_RETRIES=5
```

Mientras esté desactivado, los PDF solo se indexan si ya existe una caché
Markdown válida. Los Markdown directos sí se pueden indexar. Los dos límites
anteriores actúan como presupuesto duro por documento.

El identificador anterior es el perfil **Global cross-region** de Claude Sonnet
4.6, no el identificador del modelo base. Puede invocarse desde `eu-west-1` y
aprovecha la cuota global concedida a la cuenta. AWS puede enrutar el contenido
a cualquier región comercial; si se necesita residencia europea debe usarse el
perfil `eu.anthropic.claude-sonnet-4-6` y disponer de su cuota correspondiente.

Si se alcanza la cuota de tokens por minuto, la siguiente ejecución reanuda la
primera página pendiente. Cada página correcta ya queda persistida en la caché.
El coste pasa a ser una llamada por página, aunque cada llamada lleve hasta
cuatro imágenes para aportar contexto. Los modelos Anthropic requieren además
enviar una vez el formulario de caso de uso desde la consola de Bedrock.

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

# Fuerza una nueva conversión de los PDF aunque exista una caché válida.
.\rag.bat index data\pdfs --refresh-bedrock

# Prueba solo la página 31 con contexto vecino; consume una llamada y no cambia
# ni la caché ni el índice.
.\rag.bat bedrock-preview data\pdfs\guias\guia.pdf 31

# Consulta
.\rag.bat ask "¿Qué indica el documento?"

# Sobrescribe la configuración de recuperación predeterminada del comando ask.
.\rag.bat ask "¿Qué indica el documento?" --top-k 4 --retrieval-mode hybrid

# Query augmentation está activo por defecto. Muestra la pregunta original y
# las dos reformulaciones utilizadas para recuperar.
.\rag.bat ask "¿Qué indica el documento?" --show-queries

# Desactiva query augmentation y recupera solo con la pregunta original.
.\rag.bat ask "¿Qué indica el documento?" --no-query-augmentation

# Muestra páginas, chunks y medidas de ranking de cada resultado recuperado.
.\rag.bat ask "¿Qué indica el documento?" --stream --show-top-k

# API y visor (visor en http://localhost:8501)
.\rag.bat api
.\rag.bat viewer

# Dataset manual y evaluaciones de retrieval (http://localhost:8502)
.\rag.bat evaluate
```

## Evaluador visual de retrieval

`evaluate` abre una interfaz Streamlit con tres pestanas:

1. **Dataset** permite escribir preguntas y marcar los chunks que contienen la
   informacion correcta. La verdad de referencia se guarda por documento y
   pagina, de modo que varios chunks de una misma pagina cuentan como el mismo
   resultado relevante.
2. **Nueva evaluacion** ejecuta todas las preguntas activas sobre el indice
   actual. Se pueden cambiar `bm25|vector|hybrid`, Top K, `cosine|l2`, la
   instruccion de embedding y query augmentation sin reindexar.
3. **Historial** compara ejecuciones, muestra los fallos y permite descargar el
   detalle en CSV.

Preguntas, paginas relevantes, configuraciones, rankings y metricas se guardan
en `data/evaluation.sqlite`. Cada evaluacion conserva el hash de la version del
dataset que utilizo; la interfaz avisa si se intenta compararla con el dataset
actual despues de editar o anadir preguntas.

El evaluador mide recuperacion por pagina y no genera una respuesta final. Sus
metricas son Hit@1, Hit@K, Recall@1, Recall@K, Precision@K, MRR@K, falsos
positivos, fallos y latencia. Para comparar configuraciones de forma valida se
deben usar filas con el mismo hash de dataset y el mismo Top K.

Con query augmentation activado, el chat local genera dos reformulaciones a
temperatura cero. Las variantes se versionan y almacenan en SQLite para
reutilizarlas. El chat se libera antes de cargar embeddings, y el modelo de
embeddings permanece cargado durante toda la evaluacion para evitar un arranque
por pregunta. Esta operacion no llama a Bedrock ni modifica el indice.

Para la primera prueba conviene usar una carpeta que contenga un solo PDF:

```powershell
.\rag.bat doctor
.\rag.bat index data\prueba-bedrock --tag prueba
```

El cambio desde el antiguo contrato por lotes/`slots` invalida automáticamente
su caché mediante `BEDROCK_PROMPT_VERSION`; no hace falta añadir
`--refresh-bedrock` para esa primera migración.

`doctor` no llama a AWS ni consume tokens. El segundo comando sí realiza las
llamadas necesarias y guarda el Markdown por página en `data/markdown`.
Los errores temporales `503`, `500` y `ModelNotReady` se reintentan con espera
exponencial y jitter antes de interrumpir el indexado.

Cambiar `RETRIEVAL_MODE=vector|bm25|hybrid` y `RETRIEVAL_TOP_K` permite cambiar
la configuración global. El resultado recuperado se colapsa por
página, lo que deja preparado el cálculo posterior de precisión y recall.
`--show-top-k` muestra esos resultados antes de que el auditor de citas descarte
los que no soportan directamente la respuesta final. Incluye distancia
vectorial (menor es mejor), score BM25 (mayor es mejor) y, en modo híbrido,
score RRF y rangos de cada recuperador cuando estén disponibles.

`rag.bat ask` usa por defecto recuperación vectorial, distancia coseno, Top K 9,
instrucción de consulta y query augmentation. El chat local genera dos
reformulaciones y la recuperación combina las tres consultas. El chat se libera
antes de cargar embeddings. `--top-k`, `--retrieval-mode` y `--distance-type`
permiten sobrescribir la recuperación; `--no-query-augmentation` y
`--no-query-instruction` desactivan los dos apoyos de consulta. `--show-queries`
muestra las consultas realmente utilizadas. Si la reformulación falla, `ask`
avisa y continúa con la pregunta original. El CLI nunca activa ni muestra
razonamiento.

Las respuestas están cerradas al contexto documental recuperado: el prompt
prohíbe usar conocimiento general o ejecutar peticiones ajenas a los documentos.
Además, el auditor de citas aplica una segunda barrera; si ninguna página
recuperada respalda directamente la generación, la salida se sustituye por
`No consta en los documentos recuperados.`. En modo `--stream`, el texto se
retiene hasta terminar esta comprobación para no mostrar primero una respuesta
sin respaldo.

## Verificación

```powershell
.\rag.bat test
```

Los tests de Bedrock usan un cliente falso: no acceden a AWS ni cargan modelos.
Al cambiar desde un índice antiguo hay que ejecutar de nuevo `index`, porque el
esquema actual guarda procedencia Bedrock y posición del chunk dentro de página.
