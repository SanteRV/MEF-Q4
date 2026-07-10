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

## Instalación y ejecución (desde el código fuente)

```powershell
# 1. Clonar el repositorio
git clone https://github.com/SanteRV/MEF-Q4.git
cd MEF-Q4

# 2. Crear y activar el entorno virtual
python -m venv venv
.\venv\Scripts\Activate.ps1

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Ejecutar el aplicativo
python -m src.main
```

> **Nota sobre el punto de entrada.** Usa `python -m src.main` (o
> `python run.py`). **No** ejecutes `python src/main.py` de forma suelta: falla
> porque los imports relativos internos necesitan cargarse como parte del
> paquete `src`.

> **Nota sobre PyOpenGL en Windows.** Si `pip install -r requirements.txt` falla
> al compilar `PyOpenGL_accelerate`, tienes tres opciones: (1) instalar
> *Microsoft C++ Build Tools*; (2) usar una wheel precompilada; o (3) quitar la
> línea `PyOpenGL_accelerate` de `requirements.txt` — es solo una optimización,
> `PyOpenGL` a secas basta para la Vista 3D (aunque algo más lenta).

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
