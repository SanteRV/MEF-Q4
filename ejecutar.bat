@echo off
REM ============================================================
REM  MEF-Q4 - Lanzador del aplicativo
REM  Usa el Python del entorno virtual directamente, por lo que
REM  NO hace falta activar el venv ni configurar nada.
REM ============================================================
cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
    echo El entorno virtual no existe todavia. Crealo una sola vez con:
    echo.
    echo     python -m venv venv
    echo     venv\Scripts\python.exe -m pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

venv\Scripts\python.exe -m src.main
if errorlevel 1 pause
