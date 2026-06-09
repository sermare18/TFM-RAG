@echo off
setlocal

cd /d "%~dp0"
set "PYTHONPATH=%~dp0src;%PYTHONPATH%"

python -m streamlit run streamlit_lancedb_viewer.py