@echo off
setlocal

set "ENV_NAME=rag-cliente"
set "PROJECT_ROOT=%~dp0"
set "COMMAND=%~1"
set "FORWARD_ARGS=%*"
call set "FORWARD_ARGS=%%FORWARD_ARGS:*%COMMAND%=%%"

if "%COMMAND%"=="" goto :help
where conda >nul 2>nul
if errorlevel 1 (
    echo ERROR: Conda no esta disponible en PATH.
    exit /b 1
)

cd /d "%PROJECT_ROOT%"

if /I "%COMMAND%"=="api" goto :api
if /I "%COMMAND%"=="index" goto :index
if /I "%COMMAND%"=="ask" goto :ask
if /I "%COMMAND%"=="viewer" goto :viewer
if /I "%COMMAND%"=="evaluate" goto :evaluate
if /I "%COMMAND%"=="test" goto :test
if /I "%COMMAND%"=="gpu" goto :gpu
if /I "%COMMAND%"=="doctor" goto :doctor
if /I "%COMMAND%"=="models" goto :models
if /I "%COMMAND%"=="bedrock-preview" goto :bedrock_preview
goto :help

:api
set "PORT=%~2"
if "%PORT%"=="" set "PORT=8000"
echo API: http://localhost:%PORT%/docs
conda run --no-capture-output -n "%ENV_NAME%" python -m uvicorn rag_cliente.api:app --host 0.0.0.0 --port "%PORT%"
exit /b %ERRORLEVEL%

:index
if "%~2"=="" goto :index_default
set "DOC_DIR=%~2"
conda run --no-capture-output -n "%ENV_NAME%" python -m rag_cliente.cli index --doc-dir "%DOC_DIR%" %3 %4 %5 %6 %7 %8 %9
exit /b %ERRORLEVEL%

:index_default
conda run --no-capture-output -n "%ENV_NAME%" python -m rag_cliente.cli index --doc-dir "data\pdfs"
exit /b %ERRORLEVEL%

:ask
if "%~2"=="" (
    echo Uso: rag.bat ask "pregunta" [--top-k N] [--retrieval-mode MODO] [--distance-type TIPO] [--tag TAG] [--stream] [--no-query-augmentation] [--no-query-instruction] [--show-queries] [--show-top-k]
    exit /b 1
)
conda run --no-capture-output -n "%ENV_NAME%" python -m rag_cliente.cli ask %FORWARD_ARGS%
exit /b %ERRORLEVEL%

:viewer
set "PORT=%~2"
if "%PORT%"=="" set "PORT=8501"
echo Visor de LanceDB: http://localhost:%PORT%
conda run --no-capture-output -n "%ENV_NAME%" python -m streamlit run streamlit_lancedb_viewer.py --server.port "%PORT%"
exit /b %ERRORLEVEL%

:evaluate
set "PORT=%~2"
if "%PORT%"=="" set "PORT=8502"
echo Evaluador RAG: http://localhost:%PORT%
conda run --no-capture-output -n "%ENV_NAME%" python -m streamlit run evaluation_app.py --server.port "%PORT%"
exit /b %ERRORLEVEL%

:test
conda run --no-capture-output -n "%ENV_NAME%" python -m unittest discover -s tests -v
exit /b %ERRORLEVEL%

:gpu
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
exit /b %ERRORLEVEL%

:doctor
conda run --no-capture-output -n "%ENV_NAME%" python -m rag_cliente.cli doctor
exit /b %ERRORLEVEL%

:models
if "%~2"=="" (
    echo Uso: rag.bat models ^<plan^|download^|check^> [cpu^|gpu]
    exit /b 1
)
conda run --no-capture-output -n "%ENV_NAME%" python -m rag_cliente.cli models %2 %3 %4 %5 %6 %7 %8 %9
exit /b %ERRORLEVEL%

:bedrock_preview
if "%~2"=="" (
    echo Uso: rag.bat bedrock-preview ^<pdf^> ^<pagina^> [pagina...]
    exit /b 1
)
conda run --no-capture-output -n "%ENV_NAME%" python -m rag_cliente.cli bedrock-preview %FORWARD_ARGS%
exit /b %ERRORLEVEL%

:help
echo Uso: rag.bat ^<comando^> [argumento]
echo.
echo   api [puerto]       Arranca FastAPI. Puerto por defecto: 8000
echo   index [carpeta]    Indexa documentos; admite --tag despues de la carpeta
echo   ask [opciones]     Consulta el RAG; admite todas las opciones del CLI
echo   viewer [puerto]    Abre el visor de LanceDB. Puerto por defecto: 8501
echo   evaluate [puerto]  Abre el evaluador visual. Puerto por defecto: 8502
echo   test               Ejecuta los tests
echo   gpu                Muestra la GPU NVIDIA mediante nvidia-smi
echo   doctor             Valida Bedrock, llama.cpp, disco y modelos sin arrancarlos
echo   models plan cpu    Muestra el plan local de modelos CPU sin descargar
echo   models plan gpu    Muestra el plan local de modelos GPU sin descargar
echo   models download cpu/gpu  Descarga solo al solicitarlo explicitamente
echo   models check       Valida los GGUF locales sin cargar modelos
echo   bedrock-preview    Extrae paginas concretas sin cambiar cache ni indice
exit /b 0
