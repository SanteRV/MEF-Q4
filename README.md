# MEF-Q4

**Aplicativo de escritorio didáctico para el análisis por el Método de Elementos
Finitos (MEF) con el elemento cuadrilátero bilineal Q4.**

Resuelve problemas planos (tensión / deformación plana) desarrollando el
procedimiento completo **paso a paso en 15 etapas** —funciones de forma, puntos
de Gauss, matriz constitutiva, jacobiano, matriz B, ensamblaje, condiciones de
borde, desplazamientos, reacciones y esfuerzos— y presenta el modelo en una
**vista 3D orbital**. Exporta todo el desarrollo a **Excel** y **PDF**.

Trabajo de tesis de Ingeniería Civil.

![Python](https://img.shields.io/badge/Python-3.13-blue)
![PySide6](https://img.shields.io/badge/GUI-PySide6-green)
![Plataforma](https://img.shields.io/badge/Windows-10%2F11-lightgrey)
![Licencia](https://img.shields.io/badge/Licencia-MIT-yellow)

---

## Capturas de pantalla

> Pendiente: agregar imágenes en `assets/` y enlazarlas aquí. Sugerencia de
> capturas: pestaña "Paso a paso" (paso 3 con funciones de forma), editor
> gráfico, vista 3D con deformada, mapa de esfuerzos y círculo de Mohr.

```
![Paso a paso](assets/captura_pasos.png)
![Vista 3D](assets/captura_3d.png)
```

---

## Características

- **Elemento Q4 bilineal isoparamétrico**: 4 nodos, 2 GDL por nodo,
  convención de nodos `N1=(--), N2=(+-), N3=(++), N4=(-+)` (CCW desde la
  esquina inferior izquierda). La matriz B se construye como `B = A·G`.
- **15 pasos didácticos**: cada etapa muestra la fórmula, la sustitución
  numérica y el resultado en tablas y matrices.
- **Dos niveles de usuario**: Novato (explicaciones detalladas) y
  Experimentado (compacto).
- **Tres pestañas**:
  - *Paso a paso* — navegación por las 15 etapas con gráficos (funciones de
    forma como superficies 3D, puntos de Gauss en perspectiva, malla,
    deformada, mapa de esfuerzos, esfuerzos principales).
  - *Editor gráfico* — lienzo estilo SAP2000 para crear modelos con el mouse
    (nodos, elementos Q4, malla rectangular, apoyos, cargas), con
    deshacer/rehacer y selección por rectángulo.
  - *Vista 3D* — cámara orbital sobre el modelo Q4 plano: rotar, pan, zoom,
    deformada superpuesta, apoyos, cargas y coloreo por campo de esfuerzos.
- **Herramientas**: estudio de convergencia automático, comparación con
  solución analítica, círculo de Mohr, sonda de esfuerzos/deformaciones.
- **Exportaciones**: reporte de análisis (PDF), Excel (por pasos o por
  categorías), manual teórico (PDF), guardado/carga de proyecto (`.json`).
- **Precisión numérica** validada contra un Excel de referencia a precisión de
  máquina (errores < 1e-10).

---

## Requisitos

- **Windows 10 u 11** (64 bits). El desarrollo y la validación se hacen en
  Windows + PowerShell.
- **Python 3.13** (recomendado; el proyecto se compila con CPython 3.13).
- Tarjeta gráfica con **OpenGL 2.0 o superior** para la Vista 3D.

---

## Instalación paso a paso (se hace UNA sola vez)

### Paso 0 — Requisitos previos

Antes de empezar, verifica que tienes **Python** instalado. Abre PowerShell
y escribe:

```powershell
python --version
```

- Si responde algo como `Python 3.13.x`, estás listo.
- Si dice *"python no se reconoce como un comando..."*, descarga Python
  desde [python.org/downloads](https://www.python.org/downloads/) e
  instálalo **marcando la casilla "Add python.exe to PATH"** (es la opción
  más importante del instalador). Luego cierra y vuelve a abrir PowerShell.

Para descargar el proyecto necesitas **Git** ([git-scm.com](https://git-scm.com/))
o, si no quieres instalarlo, puedes bajar el proyecto como ZIP (se explica
en el paso 1).

### Paso 1 — Descargar el proyecto

**Opción con Git** (recomendada, permite actualizar con `git pull`):

```powershell
git clone https://github.com/SanteRV/MEF-Q4.git
cd MEF-Q4
```

**Opción sin Git**: en la página del repositorio pulsa el botón verde
`Code` → `Download ZIP`, descomprime el archivo donde quieras y abre
PowerShell dentro de esa carpeta (en el Explorador de Windows: clic en la
barra de direcciones, escribe `powershell` y presiona Enter).

### Paso 2 — Crear el entorno virtual

```powershell
python -m venv venv
```

**Qué hace**: crea una carpeta `venv/` con una copia privada de Python
solo para este proyecto. Así las librerías que instalemos no tocan el
Python del sistema ni interfieren con otros proyectos.

**Qué esperar**: tarda unos segundos y no imprime nada. Al terminar
aparece la carpeta `venv/` dentro del proyecto.

### Paso 3 — Instalar las dependencias

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

**Qué hace**: descarga e instala dentro del `venv` las librerías que el
aplicativo necesita (PySide6 para la interfaz gráfica, NumPy/SciPy para el
cálculo matricial, Matplotlib y PyQtGraph para los gráficos, openpyxl y
ReportLab para exportar a Excel/PDF, entre otras).

**Qué esperar**: tarda entre 2 y 5 minutos según tu internet (descarga
~500 MB). Verás muchas líneas `Collecting ...` e `Installing ...`. La
instalación fue exitosa si al final aparece `Successfully installed ...`.

Con esto la instalación terminó. **No hay que repetir estos pasos nunca
más** (salvo que borres la carpeta `venv/`).

---

## Ejecutar el aplicativo (cada vez que quieras usarlo)

Desde la carpeta del proyecto, cualquiera de estas dos opciones:

```powershell
# Opción A (la más simple): doble clic en ejecutar.bat, o desde la terminal:
.\ejecutar.bat

# Opción B: invocar directamente el Python del entorno virtual
.\venv\Scripts\python.exe -m src.main
```

**Qué esperar**: la primera vez tarda de 3 a 10 segundos; luego se abre la
ventana "Aplicativo MEF — Tesis (Elemento Q4)" con un ejemplo ya cargado.

### Por qué `python -m src.main` a secas NO funciona

Es la duda más común. En tu computadora conviven ahora DOS Python:

| Comando | Cuál Python usa | ¿Tiene las librerías? |
|---|---|---|
| `python ...` | El del sistema (el que instalaste de python.org) | No — da `ModuleNotFoundError` |
| `.\venv\Scripts\python.exe ...` | El del entorno virtual del proyecto | Sí — aquí instaló pip en el paso 3 |

El comando "activar" (`.\venv\Scripts\Activate.ps1`) hace que `python` a
secas apunte temporalmente al del venv, **pero solo en esa ventana de
terminal**: al cerrarla, la activación se pierde. Por eso, si cierras y
vuelves a entrar, `python -m src.main` falla de nuevo. Usar
`ejecutar.bat` (o la ruta completa al python del venv) evita el problema
de raíz — funciona siempre, sin activar nada.

> Además, en Windows la política de ejecución suele bloquear `Activate.ps1`
> con el error *"la ejecución de scripts está deshabilitada en este
> sistema"*. Si aun así prefieres activar el entorno, habilítalo una sola
> vez con: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

---

## Errores comunes al instalar o ejecutar

| Mensaje o síntoma | Causa | Solución |
|---|---|---|
| `python no se reconoce como un comando` | Python no está instalado o no se agregó al PATH | Reinstalar Python marcando "Add python.exe to PATH"; cerrar y reabrir PowerShell |
| `ModuleNotFoundError: No module named 'PySide6'` | Ejecutaste con el Python del sistema, no el del venv | Usar `.\ejecutar.bat` o `.\venv\Scripts\python.exe -m src.main` |
| `la ejecución de scripts está deshabilitada en este sistema` | La política de PowerShell bloquea `Activate.ps1` | No hace falta activar; usa `ejecutar.bat`. O habilita scripts: `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser` |
| `attempted relative import with no known parent package` | Ejecutaste `python src/main.py` directo | Usar `-m src.main` o `run.py` (los imports internos requieren cargarse como paquete) |
| Falla instalando `PyOpenGL_accelerate` (error de compilación) | Falta el compilador C++ de Microsoft | Borrar la línea `PyOpenGL_accelerate` de `requirements.txt` y reinstalar — es solo una optimización, la Vista 3D funciona sin ella. Alternativa: instalar *Microsoft C++ Build Tools* |
| La Vista 3D se ve negra | Tarjeta gráfica sin OpenGL 2.0+ o driver antiguo | Actualizar el controlador de video |
| El antivirus bloquea `ejecutar.bat` o el `.exe` | Falso positivo común con PyInstaller | Permitir la carpeta del proyecto en el antivirus |

---

## Construir el ejecutable (.exe)

El proyecto se empaqueta con PyInstaller en un `.exe` autocontenido (no requiere
Python instalado en la máquina destino):

```powershell
.\build_exe.bat
```

Esto limpia, compila y genera:
- `dist/MEF_Q4/MEF_Q4.exe` — carpeta ejecutable.
- `dist/MEF_Q4.zip` — comprimido listo para distribuir.

> El `.exe` y el `.zip` **no** se versionan en git (ver `.gitignore`). Para
> distribuirlos usa **GitHub Releases**, que admite binarios grandes.

---

## Ejecutar los tests

```powershell
# Smoke test del núcleo (sin UI), contra examples/q4_placa.json
python test_core.py

# Validación contra el Excel de referencia PLANE.xlsx.
# La ruta se toma del argumento, la variable PLANE_XLSX, o ./PLANE.xlsx
python compare_excel.py ruta\a\PLANE.xlsx
```

El criterio de éxito de `compare_excel.py` es que todos los valores físicos
(matriz D, |J|, K^e, sistema reducido, desplazamientos y esfuerzos) coincidan
con el Excel a precisión de máquina (errores < 1e-10).

---

## Estructura del proyecto

```
MEF-Q4/
├── run.py                    # Launcher (entry para desarrollo y PyInstaller)
├── requirements.txt          # Dependencias
├── build_exe.bat / .spec     # Construcción del .exe
├── test_core.py              # Smoke test sin UI
├── compare_excel.py          # Validación contra PLANE.xlsx
├── examples/                 # Proyectos demo (.json)
├── assets/                   # Iconos, logos, capturas
└── src/
    ├── main.py               # Función main() (invocada por run.py)
    ├── fem/                  # Núcleo de cálculo (Python puro, sin Qt)
    │   ├── node.py           #   Nodos y grados de libertad
    │   ├── q4_element.py     #   Q4 bilineal: NATURAL_COORDS, matriz B, jacobiano
    │   ├── structure.py      #   Contenedor de nodos y elementos
    │   ├── solver.py         #   Ensamblaje y solución del sistema
    │   ├── steps.py          #   Generador de los 15 pasos didácticos
    │   ├── analytical.py     #   Soluciones cerradas para validar
    │   ├── materials.py      #   Catálogo de materiales (Acero, Hormigón, ...)
    │   └── project_io.py     #   Guardar/cargar proyecto .json
    ├── ui/                   # Interfaz gráfica (PySide6)
    │   ├── main_window.py    #   Ventana principal con las 3 pestañas
    │   ├── canvas_editor.py  #   Editor gráfico estilo SAP2000
    │   ├── pg_canvas.py      #   Lienzo 2D (PyQtGraph)
    │   ├── pg_canvas_3d.py   #   Vista 3D orbital (OpenGL)
    │   ├── plot_widget.py    #   Gráficos por paso (Matplotlib)
    │   ├── model_tree.py     #   Árbol del modelo
    │   ├── property_inspector.py
    │   ├── workflow_stepper.py
    │   ├── theme.py          #   Paleta y estilos (QSS)
    │   ├── icons.py          #   Iconos vectoriales (qtawesome)
    │   └── ... (diálogos: Mohr, convergencia, analítica, materiales)
    └── export/               # Exportadores
        ├── excel_export.py
        ├── pdf_export.py     #   Reporte del análisis
        └── manual_pdf.py     #   Manual teórico estático
```

---

## Dependencias

| Paquete | Para qué se usa |
|---|---|
| **PySide6** | Interfaz gráfica (Qt). |
| **NumPy / SciPy** | Cálculo matricial y solución del sistema. |
| **Matplotlib** | Gráficos de los pasos (funciones de forma, Gauss, deformada). |
| **PyQtGraph** | Lienzo 2D del editor y base de la Vista 3D. |
| **PyOpenGL / PyOpenGL_accelerate** | Renderizado 3D orbital del modelo. |
| **pandas** | Modelos de datos para las tablas de la UI. |
| **openpyxl** | Exportación e importación de Excel. |
| **ReportLab** | Exportación a PDF. |
| **Pillow** | Manejo de imágenes en los PDF. |
| **qtawesome** | Iconos vectoriales (Font Awesome). |
| **PyInstaller** | Empaquetado del `.exe`. |

---

## Cómo citar

Si utilizas este software, cítalo mediante el archivo [`CITATION.cff`](CITATION.cff)
(GitHub muestra el botón *"Cite this repository"*). Completa el título de la
tesis, la institución y el año antes de la versión final.

---

## Licencia

Distribuido bajo licencia [MIT](LICENSE). Puedes usar, modificar y redistribuir
el código con atribución.

---

## Autor

**Manuel Carbajal** — Trabajo de tesis, Ingeniería Civil.
