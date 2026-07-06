# PyInstaller spec — Aplicativo MEF Q4 (Trabajo de Tesis)
#
# Ejecutar con:
#     .\venv\Scripts\pyinstaller.exe build_exe.spec --noconfirm
#
# Produce: dist/MEF_Q4/MEF_Q4.exe  (con todas las dependencias en la misma carpeta)

from PyInstaller.utils.hooks import collect_data_files, collect_submodules
from pathlib import Path

# ---- Datos a empaquetar ----
datas = []
# Fuentes de qtawesome (Font Awesome) — DEBEN estar para que se vean los iconos
datas += collect_data_files("qtawesome")
# Carpeta examples (proyectos JSON de ejemplo)
examples_dir = Path("examples")
if examples_dir.exists():
    for f in examples_dir.glob("*.json"):
        datas.append((str(f), "examples"))

# ---- Imports ocultos (matplotlib backends + scipy submodules + src + OpenGL) ----
hiddenimports = []
hiddenimports += collect_submodules("scipy")
hiddenimports += collect_submodules("src")        # ← TODOS los módulos del paquete src
# OpenGL: requerido por el modo 3D (Hex8/Tet4). PyInstaller no detecta el wrapper
# nativo de OpenGL.platform, hay que listarlo explicitamente.
hiddenimports += collect_submodules("OpenGL")
hiddenimports += collect_submodules("OpenGL_accelerate")
hiddenimports += collect_submodules("pyqtgraph.opengl")
hiddenimports += ["matplotlib.backends.backend_qtagg",
                  "matplotlib.backends.backend_agg",
                  "openpyxl",
                  "reportlab",
                  "qtawesome",
                  "OpenGL.GL",
                  "OpenGL.GLU",
                  "OpenGL.platform.win32",
                  "PIL"]  # PIL: usado por pdf_export para preservar aspect ratio

# ---- Análisis ----
a = Analysis(
    ['run.py'],                # ← Launcher que importa src.main correctamente
    pathex=['.'],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[
        # Reducir tamaño: dependencias que no usamos
        'tkinter', 'PyQt5', 'PyQt6', 'PySide2',
        'IPython', 'jupyter', 'pytest', 'notebook',
    ],
    noarchive=False,
)

pyz = PYZ(a.pure)

# ---- Modo --onedir (carpeta MEF_Q4/) ----
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='MEF_Q4',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,                 # UPX comprime pero ralentiza; lo dejamos off
    console=False,             # GUI app: no abrir consola
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    # icon='assets/icon.ico',  # (poner ruta si tenemos un ícono)
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='MEF_Q4',             # carpeta final: dist/MEF_Q4/
)
