"""Launcher del aplicativo MEF Q4.

Este archivo es el punto de entrada para PyInstaller (y también para
ejecución directa). Importa el módulo src.main correctamente como
parte del paquete `src`, lo que permite que los imports relativos
internos (from .ui.main_window import ...) funcionen.

Uso:
    Dev:           python run.py
    PyInstaller:   build_exe.bat (usa este archivo como entry)
"""
import sys
from src.main import main

if __name__ == "__main__":
    sys.exit(main())
