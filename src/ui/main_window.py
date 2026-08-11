"""Ventana principal con stepper paso a paso y edición interactiva."""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QStandardItemModel, QStandardItem, QDoubleValidator, QColor, QFont
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QListWidget,
    QListWidgetItem, QSplitter, QLabel, QTableView, QFileDialog, QMessageBox,
    QGroupBox, QPlainTextEdit, QScrollArea, QLineEdit, QFormLayout, QComboBox,
    QCheckBox, QGridLayout, QHeaderView, QTabWidget, QFrame, QSlider, QDialog,
    QApplication,
)
from .theme import Colors
from .icons import (
    icon, ICON_RECALC, ICON_EXCEL, ICON_PDF, ICON_OPEN, ICON_SAVE,
    ICON_TAB_STEPS, ICON_TAB_EDITOR, ICON_LOGO,
)
from PySide6.QtCore import QSize

from ..fem.node import Node
from ..fem.q4_element import Q4Element
from ..fem.structure import Structure
from ..fem.steps import build_procedure, Procedure
from ..fem.project_io import save_project, load_project, structure_to_dict
from ..export.excel_export import export_to_excel, export_to_excel_organized
from ..export.pdf_export import export_to_pdf
from .plot_widget import PlotWidget, STRESS_FIELDS
from .canvas_editor import CanvasEditor
from .pg_canvas_3d import Canvas3DQ4
from .user_level import UserLevel, get_level, set_level, load_saved_level, is_novato

# Lista de archivos recientes (max 8) — guardada en archivo en home del usuario
RECENT_FILES_PATH = Path.home() / ".trabajo_tesis_recent.json"
MAX_RECENT_FILES = 8


# Cantidad de dígitos significativos mostrados en matrices y tablas.
# 15 = precisión completa float64. Para presentación didáctica se puede
# bajar a 3 o 4 con el selector "Decimales display" en el header.
_DISPLAY_SIG_DIGITS = 15


def set_display_sig_digits(n: int) -> None:
    """Cambia globalmente la cantidad de dígitos en display (no afecta cálculo)."""
    global _DISPLAY_SIG_DIGITS
    _DISPLAY_SIG_DIGITS = max(2, min(n, 17))


def _fmt_num(val: float, sig: int | None = None) -> str:
    """Formato 'general' con `sig` dígitos significativos.

    Si sig es None, usa el valor global _DISPLAY_SIG_DIGITS (default 15).
    Para presentación didáctica el usuario lo baja a 3 o 4 vía el header.
    Limpia ceros residuales muy pequeños (< 1e-13) a 0.
    """
    if not np.isfinite(val):
        return str(val)
    if abs(val) < 1e-13:
        return "0"
    use_sig = sig if sig is not None else _DISPLAY_SIG_DIGITS
    return f"{val:.{use_sig}g}"


def _fmt_input(val: float) -> str:
    """Formato para campos editables: round-trip EXACTO garantizado.

    Usa repr() de Python, que produce la representación decimal más corta
    que reproduce exactamente el float64 al releer. Round-trip 100% exacto:
        v == float(_fmt_input(v))   siempre.

    Ejemplos:
        2173706.0          -> '2173706.0'
        0.1                -> '0.1'  (no '0.10000000000000001')
        0.7886751345948129 -> '0.7886751345948129'
    """
    if not np.isfinite(val):
        return str(val)
    return repr(float(val))


def df_to_qt_model(df: pd.DataFrame, editable: bool = False) -> QStandardItemModel:
    model = QStandardItemModel(df.shape[0], df.shape[1])
    model.setHorizontalHeaderLabels([str(c) for c in df.columns])
    for r in range(df.shape[0]):
        for c in range(df.shape[1]):
            val = df.iat[r, c]
            if isinstance(val, float):
                txt = _fmt_num(val)   # default = 15 sig digits
            else:
                txt = str(val)
            item = QStandardItem(txt)
            item.setEditable(editable)
            model.setItem(r, c, item)
    return model


def matrix_to_text(name: str, M: np.ndarray) -> str:
    rows = []
    shape = f"{M.shape[0]}×{M.shape[1] if M.ndim > 1 else 1}"
    rows.append(f"{name}  (forma {shape})")
    rows.append("")
    if M.ndim == 1:
        M = M.reshape(-1, 1)
    # Calcular ancho de columna basado en el valor más largo
    formatted = [[_fmt_num(v) for v in row] for row in M]   # default = 15 sig
    col_w = max((len(s) for row in formatted for s in row), default=12)
    col_w = max(col_w, 14)
    for row in formatted:
        line = "  ".join(s.rjust(col_w) for s in row)
        rows.append(f"[ {line} ]")
    return "\n".join(rows)


class _ComboAsList:
    """Adaptador mínimo que expone una API tipo QListWidget pero
    delega a un QComboBox subyacente.

    Sólo expone los métodos que MainWindow usa: count, currentRow,
    setCurrentRow, clear, addItem. Suficiente para mantener compatibilidad
    sin reescribir todo.
    """
    def __init__(self, combo: QComboBox, on_row_changed):
        self._combo = combo
        self._on_row_changed = on_row_changed   # no se usa, ya está conectado

    def count(self) -> int:
        return self._combo.count()

    def currentRow(self) -> int:
        return self._combo.currentIndex()

    def setCurrentRow(self, row: int) -> None:
        if not (0 <= row < self._combo.count()):
            return
        # Si el índice ya está en row, setCurrentIndex no dispara la señal.
        # Hacemos un "toque" -1 → row para forzar el refresh.
        if self._combo.currentIndex() == row:
            self._combo.blockSignals(True)
            self._combo.setCurrentIndex(-1)
            self._combo.blockSignals(False)
        self._combo.setCurrentIndex(row)

    def clear(self) -> None:
        self._combo.blockSignals(True)
        self._combo.clear()
        self._combo.blockSignals(False)

    def addItem(self, item) -> None:
        # Acepta tanto QListWidgetItem (con texto e icono) como str
        # Bloqueamos señales para que no dispare currentIndexChanged al agregar
        # (el primer addItem auto-selecciona el index 0)
        self._combo.blockSignals(True)
        try:
            if hasattr(item, "text"):
                text = item.text()
                ic = item.icon() if hasattr(item, "icon") else None
                if ic is not None and not ic.isNull():
                    self._combo.addItem(ic, text)
                else:
                    self._combo.addItem(text)
            else:
                self._combo.addItem(str(item))
        finally:
            self._combo.blockSignals(False)

    def item(self, idx: int):
        """Devuelve un wrapper para mantener compatibilidad con .icon()."""
        class _Wrapper:
            def __init__(self, combo, i):
                self._combo = combo; self._i = i
            def icon(self):
                return self._combo.itemIcon(self._i)
            def text(self):
                return self._combo.itemText(self._i)
        return _Wrapper(self._combo, idx)


def make_table_view(model: QStandardItemModel, min_rows: int = 4,
                    max_visible_rows: int = 10) -> QTableView:
    """Crea un QTableView con altura responsiva:

    - Si la tabla tiene <= max_visible_rows filas: muestra todas sin scroll.
    - Si tiene más: muestra max_visible_rows y habilita scroll vertical.

    Esto evita que con 25+ nodos la tabla se "vaya al fondo" desbordando
    el panel inferior (correcciones A1 / A2 del documento).
    """
    view = QTableView()
    view.setModel(model)
    view.setAlternatingRowColors(True)
    view.setShowGrid(True)
    view.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
    view.horizontalHeader().setStretchLastSection(False)
    view.horizontalHeader().setHighlightSections(False)
    view.verticalHeader().setVisible(True)
    view.verticalHeader().setDefaultSectionSize(28)

    visible_rows = min(max(model.rowCount(), min_rows), max_visible_rows)
    height = view.horizontalHeader().height() + visible_rows * 30 + 14
    view.setMinimumHeight(height)
    view.setMaximumHeight(height + 4)
    # Si la tabla tiene más filas que max_visible_rows, habilita scroll interno
    if model.rowCount() > max_visible_rows:
        from PySide6.QtCore import Qt as _Qt
        view.setVerticalScrollBarPolicy(_Qt.ScrollBarAsNeeded)
    return view


def _dof_labels(n_nodes: int) -> list[str]:
    """Etiquetas de DOFs con índice numérico para ensamblaje.

    Ejemplo: u1x[1], u1y[2], u2x[3], ..., u4y[8].
    El [N] hace explícito el número de fila/columna global para ensamblaje.
    """
    labels = []
    idx = 1
    for n in range(1, n_nodes + 1):
        labels.append(f"u{n}x [{idx}]"); idx += 1
        labels.append(f"u{n}y [{idx}]"); idx += 1
    return labels


def _guess_matrix_labels(name: str, shape: tuple[int, int]) -> tuple[list[str], list[str]]:
    """Devuelve (row_labels, col_labels) según el nombre/forma de la matriz.

    Reconoce matrices típicas de MEF Q4 para mostrar etiquetas semánticas
    (e.g. B 3x8 → filas εx,εy,γxy y columnas u1x[1],u1y[2],...,u4y[8]).
    """
    rows, cols = shape
    row_labels = [str(i) for i in range(rows)]
    col_labels = [str(i) for i in range(cols)]
    lname = name.lower()

    # Matriz D 3×3 → constitutiva
    if "d " in lname or lname.startswith("d ") or lname == "d" or lname.startswith("d ("):
        if rows == 3 and cols == 3:
            return (["σx / εx", "σy / εy", "τxy / γxy"],
                    ["εx", "εy", "γxy"])

    # Matriz B 3×8 → strain-displacement
    if " b " in f" {lname} " or lname.startswith("b ") or "—  b " in lname or "— b " in lname:
        if rows == 3 and cols == 8:
            return (["εx", "εy", "γxy"], _dof_labels(4))

    # Cualquier matriz 8×8 → presumimos K^e
    if rows == 8 and cols == 8:
        labels = _dof_labels(4)
        return (labels, labels)

    # Jacobiano 2×2
    if rows == 2 and cols == 2 and ("j" in lname or "jac" in lname):
        return (["∂x", "∂y"], ["∂ξ", "∂η"])

    # Vectores columna (n×1) — vector de cargas o desplazamientos
    if cols == 1 and rows in (4, 6, 8):
        return (_dof_labels(rows // 2)[:rows], ["valor"])

    return row_labels, col_labels


def matrix_to_widget(name: str, M: np.ndarray) -> QGroupBox:
    """Construye un QGroupBox con la matriz `M` mostrada como QTableView.

    - Encabezados navy (heredados del QSS global)
    - Filas/columnas etiquetadas según el tipo de matriz (DOFs, ε, etc.)
    - Ceros exactos en gris claro para resaltar estructura (sparsity)
    - Forma y tamaño anotados en el título del grupo
    """
    if M.ndim == 1:
        M = M.reshape(-1, 1)
    rows, cols = M.shape

    # Título con la forma incluida
    title = f"{name}   —   forma {rows}×{cols}"
    box = QGroupBox(title)
    layout = QVBoxLayout(box)
    layout.setContentsMargins(8, 14, 8, 8)

    # Construir modelo
    model = QStandardItemModel(rows, cols)
    row_labels, col_labels = _guess_matrix_labels(name, M.shape)
    model.setHorizontalHeaderLabels(col_labels)
    model.setVerticalHeaderLabels(row_labels)

    for r in range(rows):
        for c in range(cols):
            v = float(M[r, c])
            if abs(v) < 1e-13:
                txt = "0"
                is_zero = True
            else:
                txt = _fmt_num(v)
                is_zero = False
            item = QStandardItem(txt)
            item.setEditable(False)
            item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            # Tooltip con valor pleno (debug/verificación)
            item.setToolTip(f"[{r}, {c}] = {v!r}")
            # Ceros en gris claro para resaltar el patrón de la matriz
            if is_zero:
                item.setForeground(QColor(Colors.BORDER))
            model.setItem(r, c, item)

    view = QTableView()
    view.setModel(model)
    view.setAlternatingRowColors(True)
    view.setShowGrid(True)
    view.setSelectionMode(QTableView.SingleSelection)
    view.setFont(QFont("Consolas", 10))
    view.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
    view.horizontalHeader().setHighlightSections(False)
    view.verticalHeader().setHighlightSections(False)
    view.verticalHeader().setDefaultSectionSize(26)
    # Altura responsiva: matrices con muchas filas (como K^e 8×8) tienen
    # límite máximo para que no desborden el panel al agrandar la ventana.
    # Si la matriz es más alta que el límite, se habilita scroll interno.
    max_visible_rows = 10
    visible = min(rows, max_visible_rows)
    h = view.horizontalHeader().height() + visible * 28 + 6
    view.setMinimumHeight(h)
    view.setMaximumHeight(h + 4)
    layout.addWidget(view)
    return box


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Aplicativo MEF — Tesis (Elemento Q4)")
        self.resize(1380, 880)

        # Cargar preferencia de nivel guardada
        load_saved_level()

        self.structure: Structure | None = None
        self.procedure: Procedure | None = None
        # Inputs editables
        self.node_inputs: list[dict] = []   # filas con QLineEdits / QCheckBoxes
        self.E_input: QLineEdit | None = None
        self.nu_input: QLineEdit | None = None
        self.t_input: QLineEdit | None = None
        self.plane_combo: QComboBox | None = None

        self._build_menu()
        self._build_ui()
        self._load_default_example()

    # ---------------- Menú ----------------
    def _build_menu(self):
        menu = self.menuBar()
        archivo = menu.addMenu("&Archivo")

        # --- Proyecto ---
        a_new = QAction(icon(ICON_OPEN), "&Nuevo proyecto", self)
        a_new.setShortcut("Ctrl+N")
        a_new.triggered.connect(self.new_project)
        archivo.addAction(a_new)

        a_open = QAction(icon(ICON_OPEN), "&Abrir proyecto…", self)
        a_open.setShortcut("Ctrl+O")
        a_open.triggered.connect(self.open_project)
        archivo.addAction(a_open)

        a_save = QAction(icon(ICON_SAVE), "&Guardar", self)
        a_save.setShortcut("Ctrl+S")
        a_save.triggered.connect(self.save_project)
        archivo.addAction(a_save)

        a_save_as = QAction(icon(ICON_SAVE), "Guardar &como…", self)
        a_save_as.setShortcut("Ctrl+Shift+S")
        a_save_as.triggered.connect(self.save_project_as)
        archivo.addAction(a_save_as)

        # Submenú de recientes
        self.recent_menu = archivo.addMenu("&Recientes")
        self._refresh_recent_menu()

        archivo.addSeparator()

        # --- Exportar ---
        a_excel = QAction(icon(ICON_EXCEL), "Exportar a &Excel (.xlsx)…", self)
        a_excel.setShortcut("Ctrl+E")
        a_excel.triggered.connect(self.export_excel)
        archivo.addAction(a_excel)

        a_pdf = QAction(icon(ICON_PDF), "Exportar &Reporte de análisis (PDF)…", self)
        a_pdf.setShortcut("Ctrl+P")
        a_pdf.setToolTip(
            "Reporte completo del análisis actual con los valores del "
            "proyecto: matrices, fórmulas sustituidas, gráficas, resultados."
        )
        a_pdf.triggered.connect(self.export_pdf)
        archivo.addAction(a_pdf)

        a_manual = QAction(icon(ICON_PDF), "Exportar &Manual teórico (PDF)…", self)
        a_manual.setToolTip(
            "Documento de referencia con la teoría del Q4 (sin valores "
            "específicos): funciones de forma, Jacobiano, matriz B, "
            "cuadratura de Gauss, K^e. Útil como anexo de tesis."
        )
        a_manual.triggered.connect(self.export_manual_pdf)
        archivo.addAction(a_manual)

        archivo.addSeparator()
        a_quit = QAction("&Salir", self)
        a_quit.setShortcut("Ctrl+Q")
        a_quit.triggered.connect(self.close)
        archivo.addAction(a_quit)

        # ===== Menú Herramientas =====
        herramientas = menu.addMenu("&Herramientas")

        a_conv = QAction("Estudio de &convergencia…", self)
        a_conv.setShortcut("Ctrl+K")
        a_conv.triggered.connect(self.show_convergence_dialog)
        herramientas.addAction(a_conv)

        a_analytical = QAction("Comparar con solución &analítica…", self)
        a_analytical.setShortcut("Ctrl+A")
        a_analytical.triggered.connect(self.show_analytical_dialog)
        herramientas.addAction(a_analytical)

        a_mohr = QAction("Círculo de &Mohr…", self)
        a_mohr.setShortcut("Ctrl+M")
        a_mohr.setToolTip(
            "Visualiza el círculo de Mohr para un punto seleccionado: "
            "σ1, σ2, τ_max, ángulo principal θp."
        )
        a_mohr.triggered.connect(self.show_mohr_dialog)
        herramientas.addAction(a_mohr)

        # ===== Menú Ayuda =====
        ayuda = menu.addMenu("A&yuda")
        a_modelos = QAction("¿Qué &modelo debo usar?", self)
        a_modelos.setShortcut("F1")
        a_modelos.triggered.connect(self.show_model_guide)
        ayuda.addAction(a_modelos)

        # Estado del proyecto actual
        self._current_project_path: Path | None = None

    def show_model_guide(self) -> None:
        """Guía breve para elegir entre los cuatro modelos del documento.

        Con cuatro formulaciones disponibles, la pregunta que se hace quien
        abre el programa no es "qué botón toco" sino "cuál de los cuatro
        corresponde a mi estructura". Esto la responde en una pantalla.
        """
        QMessageBox.information(
            self, "¿Qué modelo debo usar?",
            "<b>El aplicativo cubre los cuatro elementos del documento "
            "teórico. Elija según cómo actúa la carga sobre su estructura:</b>"
            "<br><br>"
            "<b>Plano (Q4) — cap. 01.01.02, 2 GDL por nodo</b><br>"
            "La carga actúa DENTRO del plano de la pieza. Muros, vigas de "
            "gran peralte, chapas. Pestañas «Plano: paso a paso», «editor» "
            "y «vista 3D» — son tres vistas del mismo modelo."
            "<br><br>"
            "<b>Placa — cap. 01.01.03, 3 GDL por nodo</b><br>"
            "La carga actúa PERPENDICULAR al plano y la pieza es delgada. "
            "Losas, tapas. Teoría de Kirchhoff; elementos rectangulares."
            "<br><br>"
            "<b>Lámina (flat shell) — cap. 01.01.04, 5 GDL por nodo</b><br>"
            "La carga actúa en AMBAS direcciones a la vez. Es la suma "
            "exacta de los dos anteriores: membrana + flexión."
            "<br><br>"
            "<b>Pórtico (frame 3D) — cap. 01.02, 6 GDL por nodo</b><br>"
            "La estructura es de BARRAS, no de área. Vigas, columnas, "
            "pórticos planos y espaciales, con diagramas de fuerzas "
            "internas."
            "<br><br>"
            "Para comprobar que la formulación cumple los criterios de "
            "convergencia del cap. 01.01.08, use "
            "<i>Herramientas → Estudio de convergencia</i>, pestaña "
            "«Criterios de la formulación»."
        )

    def show_convergence_dialog(self) -> None:
        from .convergence_dialog import ConvergenceDialog
        d = ConvergenceDialog(self, base_structure=self.structure)
        d.exec()

    def show_analytical_dialog(self) -> None:
        from .analytical_dialog import AnalyticalDialog
        d = AnalyticalDialog(self, structure=self.structure)
        d.exec()

    def show_mohr_dialog(self) -> None:
        from .mohr_dialog import MohrDialog
        if not self.procedure:
            QMessageBox.information(
                self, "Sin resultados",
                "No hay resultados disponibles. Primero define una estructura "
                "válida con apoyos, cargas y propiedades, luego presiona Recalcular."
            )
            return
        d = MohrDialog(self, structure=self.structure,
                       fem_result=self.procedure.result)
        d.exec()

    # ---------------- UI principal ----------------
    def _build_ui(self):
        # Contenedor raíz: header arriba + tabs debajo
        central = QWidget()
        central_layout = QVBoxLayout(central)
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_layout.setSpacing(0)

        # Header banner
        central_layout.addWidget(self._build_header())

        # Tabs (con iconos vectoriales)
        self.tabs = QTabWidget()
        self.tabs.setIconSize(QSize(18, 18))
        # MODELO: la ventana única de trabajo (Corrección 2). Un solo modelo
        # donde conviven frame, plane, plate y shell. Va primera porque es
        # el flujo principal del aplicativo.
        from .model_mode import ModelModeWidget
        self.model_mode = ModelModeWidget()
        self.tabs.addTab(self.model_mode, icon("fa5s.drafting-compass"),
                         "Modelo")
        # Las pestañas siguientes son los modos DIDÁCTICOS por formulación:
        # las tres primeras son vistas del mismo modelo plano (Q4), las tres
        # últimas son modelos independientes por tipo de elemento.
        self.tabs.addTab(self._build_step_tab(), icon(ICON_TAB_STEPS),
                         "Plano: paso a paso")
        self.tabs.addTab(self._build_editor_tab(), icon(ICON_TAB_EDITOR),
                         "Plano: editor")
        self.tabs.addTab(self._build_view3d_tab(), icon(ICON_TAB_EDITOR),
                         "Plano: vista 3D")
        # Modo PLACA (flexión): pestaña autocontenida, familia paralela al Q4
        from .plate_mode import PlateModeWidget
        self.plate_mode = PlateModeWidget()
        self.tabs.addTab(self.plate_mode, icon("fa5s.layer-group"),
                         "Placa (flexión)")
        # Modo LÁMINA (flat shell): membrana + flexión, cap. 01.01.04
        from .shell_mode import ShellModeWidget
        self.shell_mode = ShellModeWidget()
        self.tabs.addTab(self.shell_mode, icon("fa5s.cubes"), "Lámina (shell)")
        # Modo PÓRTICO (frame 3D): elementos de barra, cap. 01.02
        from .frame_mode import FrameModeWidget
        self.frame_mode = FrameModeWidget()
        self.tabs.addTab(self.frame_mode, icon("fa5s.project-diagram"),
                         "Pórtico (frame 3D)")
        self._set_tab_tooltips()
        central_layout.addWidget(self.tabs, 1)

        self.setCentralWidget(central)
        self.statusBar().showMessage("Listo")

    def _build_header(self) -> QFrame:
        """Banner superior con el título del aplicativo."""
        header = QFrame()
        header.setObjectName("headerBanner")
        h = QHBoxLayout(header)
        h.setContentsMargins(20, 0, 20, 0)

        # Logo a la izquierda (icono vectorial blanco)
        import qtawesome as qta
        logo = QLabel()
        logo_pixmap = qta.icon(ICON_LOGO, color="white").pixmap(QSize(36, 36))
        logo.setPixmap(logo_pixmap)
        h.addWidget(logo)

        # Título + subtítulo en columna
        title_col = QVBoxLayout()
        title_col.setSpacing(0)
        title = QLabel("MEF Q4 — Análisis por Elementos Finitos")
        title.setObjectName("headerTitle")
        subtitle = QLabel("Trabajo de Tesis · Ingeniería Civil · Cuadrilátero bilineal en tensión / deformación plana")
        subtitle.setObjectName("headerSubtitle")
        title_col.addWidget(title)
        title_col.addWidget(subtitle)
        h.addLayout(title_col)
        h.addStretch()

        # Selector de decimales en display (no afecta cálculo)
        dec_label = QLabel("Decimales:")
        dec_label.setStyleSheet("color: #B7CCE0; font-size: 10pt;")
        h.addWidget(dec_label)
        self.dec_combo = QComboBox()
        for label, sig in [("2", 2), ("3", 3), ("4", 4), ("6", 6), ("10", 10), ("15 (completo)", 15)]:
            self.dec_combo.addItem(label, userData=sig)
        self.dec_combo.setCurrentIndex(5)   # 15 por default
        self.dec_combo.setToolTip(
            "<b>Decimales en display</b><br><br>"
            "Cantidad de dígitos significativos al mostrar matrices y tablas.<br>"
            "<b>NO afecta el cálculo</b> — el solver siempre usa precisión "
            "completa (float64, 15-17 dígitos).<br><br>"
            "Para presentación didáctica use 2-3. Para verificación numérica "
            "(comparar con Excel/MATLAB) use 15."
        )
        self.dec_combo.setStyleSheet("""
            QComboBox {
                background-color: rgba(255, 255, 255, 0.1);
                color: white;
                border: 1px solid #B7CCE0;
                border-radius: 4px;
                padding: 4px 10px;
            }
            QComboBox QAbstractItemView {
                background: white; color: #212529;
                selection-background-color: #1F4E78; selection-color: white;
            }
            QComboBox::drop-down { border: none; width: 18px; }
        """)
        self.dec_combo.currentIndexChanged.connect(self._on_dec_changed)
        h.addWidget(self.dec_combo)
        h.addSpacing(12)

        # Switch de nivel de usuario en el header
        level_label = QLabel("Modo:")
        level_label.setStyleSheet("color: #B7CCE0; font-size: 10pt;")
        h.addWidget(level_label)

        self.level_combo = QComboBox()
        for lvl in UserLevel:
            self.level_combo.addItem(lvl.label, userData=lvl)
        # Setear el actual
        cur_idx = list(UserLevel).index(get_level())
        self.level_combo.setCurrentIndex(cur_idx)
        self.level_combo.setMinimumWidth(150)
        self.level_combo.setToolTip(
            "<b>Nivel de usuario</b><br><br>"
            "<b>Novato:</b> descripciones detalladas, fórmulas explicadas, "
            "tooltips visibles, ayuda contextual.<br><br>"
            "<b>Experimentado:</b> solo matrices y resultados, UI compacta."
        )
        self.level_combo.setStyleSheet("""
            QComboBox {
                background-color: rgba(255, 255, 255, 0.1);
                color: white;
                border: 1px solid #B7CCE0;
                border-radius: 4px;
                padding: 4px 10px;
            }
            QComboBox QAbstractItemView {
                background-color: white;
                color: #212529;
                selection-background-color: #1F4E78;
                selection-color: white;
            }
            QComboBox::drop-down { border: none; width: 20px; }
            QComboBox::down-arrow {
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 5px solid white;
            }
        """)
        self.level_combo.currentIndexChanged.connect(self._on_level_changed)
        h.addWidget(self.level_combo)

        # Indicador de versión a la derecha
        version = QLabel("v1.0")
        version.setStyleSheet("color: #B7CCE0; font-size: 10pt; padding-left: 8px;")
        h.addWidget(version)

        return header

    def _on_dec_changed(self, idx: int) -> None:
        """El usuario cambia los decimales de display: re-renderizar el paso actual."""
        sig = self.dec_combo.itemData(idx)
        try:
            n = int(sig)
        except (TypeError, ValueError):
            return
        set_display_sig_digits(n)
        # Refrescar el paso actual para que las matrices/tablas se actualicen
        cur = self.steps_list.currentRow()
        if cur >= 0:
            self.show_step(cur)
        self.statusBar().showMessage(
            f"Display: {n} dígitos significativos (cálculo siempre 15+)", 3000
        )

    def _on_level_changed(self, idx: int) -> None:
        """El usuario cambia el nivel: persistir y recargar la vista."""
        raw = self.level_combo.itemData(idx)
        # Qt puede serializar el enum a str — convertir de vuelta
        if isinstance(raw, UserLevel):
            level = raw
        else:
            try:
                level = UserLevel(str(raw))
            except ValueError:
                return
        set_level(level)
        # Recalcular para que las descripciones se regeneren con el nuevo nivel
        if self.procedure is not None:
            current_step = self.steps_list.currentRow()
            self.procedure = build_procedure(self.structure)
            self._populate_steps()
            if 0 <= current_step < len(self.procedure.steps):
                self.steps_list.setCurrentRow(current_step)
        self.statusBar().showMessage(f"Modo cambiado a: {level.label}", 3000)

    # ---------------- Pestaña 2: editor gráfico ----------------
    def _build_editor_tab(self) -> QWidget:
        self.canvas_editor = CanvasEditor(structure=None, parent=self)
        # Cuando el editor modifica la estructura, recalculamos los pasos
        self.canvas_editor.structure_changed.connect(self._on_editor_changed)
        return self.canvas_editor

    def _on_editor_changed(self) -> None:
        # Sincroniza la estructura del editor con la del MainWindow y recalcula
        self.structure = self.canvas_editor.structure
        if self.structure.nodes and self.structure.elements:
            try:
                self.procedure = build_procedure(self.structure)
                self._populate_steps()
                # también actualizar el panel de edición lateral
                self._build_input_panel()
                self.statusBar().showMessage(
                    f"Estructura actualizada desde editor: "
                    f"{len(self.structure.nodes)} nodos, "
                    f"{len(self.structure.elements)} elemento(s)"
                )
            except Exception as e:
                self.statusBar().showMessage(f"Editor: {e}")
        else:
            # Estructura incompleta, limpiar pasos
            self.procedure = None
            self.steps_list.clear()
            self.statusBar().showMessage(
                "Editor: necesita al menos 1 nodo y 1 elemento para calcular"
            )

    def _set_tab_tooltips(self) -> None:
        """Explica en cada pestaña qué elemento y qué capítulo usa.

        Con cuatro modelos conviviendo, el nombre de la pestaña no alcanza:
        el tooltip dice qué elemento resuelve, cuántos GDL tiene por nodo y
        para qué tipo de estructura sirve.
        """
        tips = [
            ("Plano: paso a paso",
             "<b>Modelo PLANO (elemento Q4)</b> — cap. 01.01.02<br>"
             "2 GDL por nodo (u, v). Estructuras cargadas en su propio "
             "plano: muros, vigas de gran peralte, chapas.<br><br>"
             "Esta pestaña recorre los 15 pasos del cálculo con todas las "
             "matrices intermedias."),
            ("Plano: editor",
             "<b>Modelo PLANO (elemento Q4)</b> — cap. 01.01.02<br>"
             "Editor gráfico del MISMO modelo de la pestaña anterior: "
             "dibujar nodos, elementos, apoyos y cargas."),
            ("Plano: vista 3D",
             "<b>Modelo PLANO (elemento Q4)</b> — cap. 01.01.02<br>"
             "Vista tridimensional del MISMO modelo, útil para ver la "
             "deformada y los mapas de esfuerzos con relieve."),
            ("Placa (flexión)",
             "<b>Modelo PLACA (elemento plate de 12 GDL)</b> — cap. 01.01.03<br>"
             "3 GDL por nodo (w, θx, θy). Teoría de Kirchhoff: losas "
             "delgadas cargadas perpendicularmente a su plano.<br><br>"
             "Solo elementos rectangulares (lo exige la formulación)."),
            ("Lámina (shell)",
             "<b>Modelo LÁMINA (flat shell de 20 GDL)</b> — cap. 01.01.04<br>"
             "5 GDL por nodo (u, v, w, θx, θy). Superpone el "
             "comportamiento de membrana (plano) con el de flexión "
             "(placa) en un solo elemento.<br><br>"
             "Úselo cuando la estructura recibe carga en su plano Y "
             "perpendicular a él."),
            ("Pórtico (frame 3D)",
             "<b>Modelo PÓRTICO (elemento frame de 12 GDL)</b> — cap. 01.02<br>"
             "6 GDL por nodo (3 desplazamientos y 3 giros). Estructuras "
             "de barras: vigas, columnas, pórticos planos y espaciales.<br><br>"
             "Incluye diagramas de fuerzas internas por elemento."),
        ]
        for i in range(self.tabs.count()):
            titulo = self.tabs.tabText(i)
            for nombre, tip in tips:
                if titulo == nombre:
                    self.tabs.setTabToolTip(i, tip)
                    break

    # ---------------- Pestaña 1: paso a paso ----------------
    def _build_step_tab(self) -> QWidget:
        splitter_main = QSplitter(Qt.Horizontal)

        # ===== Izquierda: pasos + datos =====
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(8, 8, 8, 8)

        # ----- Navegador de pasos: dropdown + ◀ ▶ + indicador -----
        import qtawesome as qta
        nav_group = QGroupBox("Pasos del análisis")
        nav_layout = QVBoxLayout(nav_group)
        nav_layout.setContentsMargins(6, 14, 6, 6)
        nav_layout.setSpacing(6)

        # Fila 1: indicador "Paso X / 15" centrado
        self.step_indicator = QLabel("Paso 0 / 0")
        self.step_indicator.setAlignment(Qt.AlignCenter)
        self.step_indicator.setStyleSheet(
            f"color:{Colors.PRIMARY}; font-weight:bold; font-size:10pt;"
            f"background-color:{Colors.BG_SUBTLE}; padding:6px;"
            f"border-radius:4px; border:1px solid {Colors.BORDER};"
        )
        nav_layout.addWidget(self.step_indicator)

        # Fila 2: ◀ [dropdown] ▶
        sel_row = QHBoxLayout()
        sel_row.setSpacing(4)
        self.btn_prev = QPushButton()
        self.btn_prev.setIcon(qta.icon("fa5s.chevron-left", color=Colors.PRIMARY))
        self.btn_prev.setIconSize(QSize(14, 14))
        self.btn_prev.setToolTip("Paso anterior (Ctrl+←)")
        self.btn_prev.setMaximumWidth(38)
        self.btn_prev.clicked.connect(self._step_prev)
        sel_row.addWidget(self.btn_prev)

        # Dropdown con todos los pasos
        self.steps_combo = QComboBox()
        self.steps_combo.setObjectName("stepsCombo")
        self.steps_combo.setToolTip(
            "<b>Selector de paso</b><br>"
            "Click para ver los 15 pasos del análisis MEF.<br>"
            "También: use los botones ◀ ▶ o las teclas Ctrl+← / Ctrl+→."
        )
        self.steps_combo.currentIndexChanged.connect(self.show_step)
        # Aseguramos que el popup sea amplio para mostrar títulos largos
        self.steps_combo.view().setMinimumWidth(380)
        self.steps_combo.setStyleSheet(f"""
            QComboBox#stepsCombo {{
                background:{Colors.BG};
                border:1.5px solid {Colors.PRIMARY};
                border-radius:4px;
                padding:6px 10px;
                color:{Colors.PRIMARY};
                font-weight:bold;
                font-size:9.5pt;
            }}
            QComboBox#stepsCombo::drop-down {{
                width:24px;
                border-left:1px solid {Colors.BORDER};
            }}
            QComboBox#stepsCombo::down-arrow {{
                image:none;
                border-left:5px solid transparent;
                border-right:5px solid transparent;
                border-top:6px solid {Colors.PRIMARY};
            }}
            QComboBox#stepsCombo QAbstractItemView {{
                background:{Colors.BG};
                border:1px solid {Colors.BORDER};
                selection-background-color:{Colors.PRIMARY};
                selection-color:white;
                outline:none;
                padding:4px;
            }}
            QComboBox#stepsCombo QAbstractItemView::item {{
                padding:6px 8px;
                min-height:20px;
            }}
        """)
        sel_row.addWidget(self.steps_combo, 1)

        self.btn_next = QPushButton()
        self.btn_next.setIcon(qta.icon("fa5s.chevron-right", color=Colors.PRIMARY))
        self.btn_next.setIconSize(QSize(14, 14))
        self.btn_next.setToolTip("Paso siguiente (Ctrl+→)")
        self.btn_next.setMaximumWidth(38)
        self.btn_next.clicked.connect(self._step_next)
        sel_row.addWidget(self.btn_next)

        nav_layout.addLayout(sel_row)
        left_layout.addWidget(nav_group)

        # Compatibilidad: algunas funciones internas usan `steps_list`.
        # Mantenemos una referencia que delega al combo (lectura, count).
        self.steps_list = _ComboAsList(self.steps_combo, self.show_step)

        # Atajos de teclado Ctrl+← / Ctrl+→
        from PySide6.QtGui import QShortcut, QKeySequence
        QShortcut(QKeySequence("Ctrl+Left"), self,
                  activated=self._step_prev)
        QShortcut(QKeySequence("Ctrl+Right"), self,
                  activated=self._step_next)

        # Panel de edición rápida
        self.input_box = QGroupBox("Editar datos de entrada")
        self.input_box_layout = QVBoxLayout(self.input_box)
        left_layout.addWidget(self.input_box)

        # Botones (con iconos)
        import qtawesome as qta
        btn_recalc = QPushButton(" Recalcular")
        # Para el botón primario el icono debe ser blanco
        btn_recalc.setIcon(qta.icon(ICON_RECALC, color="white"))
        btn_recalc.setIconSize(QSize(16, 16))
        btn_recalc.setProperty("primary", True)
        btn_recalc.setToolTip(
            "<b>Recalcular (Ctrl+S de tu trabajo)</b><br>"
            "Recorre los 15 pasos del MEF con los valores actuales del "
            "proyecto y actualiza todas las matrices, desplazamientos y "
            "esfuerzos.<br><br>"
            "Úsalo después de editar nodos, propiedades (E, ν, t) o cargas."
        )
        btn_recalc.clicked.connect(self.recalculate)
        left_layout.addWidget(btn_recalc)

        btn_excel = QPushButton(" Exportar Excel (.xlsx)")
        btn_excel.setIcon(icon(ICON_EXCEL))
        btn_excel.setIconSize(QSize(16, 16))
        btn_excel.setToolTip(
            "<b>Exportar a Excel (Ctrl+E)</b><br>"
            "Genera un .xlsx con una hoja por cada uno de los 15 pasos. "
            "Cada hoja incluye tablas y matrices con precisión completa "
            "(14 decimales). Ideal para anexar a la tesis o reutilizar "
            "los números en otra herramienta."
        )
        btn_excel.clicked.connect(self.export_excel)

        btn_pdf = QPushButton(" Exportar PDF")
        btn_pdf.setIcon(icon(ICON_PDF))
        btn_pdf.setIconSize(QSize(16, 16))
        btn_pdf.setToolTip(
            "<b>Exportar a PDF (Ctrl+P)</b><br>"
            "Genera un reporte profesional con todos los pasos del análisis: "
            "geometría, matrices, gráficas, resultados.<br><br>"
            "Documento listo para presentar como anexo a la tesis."
        )
        btn_pdf.clicked.connect(self.export_pdf)

        left_layout.addWidget(btn_excel)
        left_layout.addWidget(btn_pdf)

        # ===== Derecha: contenido del paso =====
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(8, 8, 8, 8)

        self.title_label = QLabel("")
        self.title_label.setProperty("heading", True)
        right_layout.addWidget(self.title_label)

        self.desc_label = QLabel("")
        self.desc_label.setWordWrap(True)
        self.desc_label.setProperty("muted", True)
        right_layout.addWidget(self.desc_label)

        # ===== Layout vertical: gráfica arriba (siempre visible) + tabs Tablas/Matrices abajo =====
        content_split = QSplitter(Qt.Vertical)
        content_split.setHandleWidth(8)
        content_split.setChildrenCollapsible(False)

        # ----- Zona superior: gráfica con slider opcional -----
        self.plot_box = QGroupBox("Gráfica")
        plot_box = self.plot_box   # alias local para no romper el resto del bloque
        pl = QVBoxLayout(plot_box)
        pl.setContentsMargins(8, 12, 8, 8)
        self.plot_widget = PlotWidget()
        pl.addWidget(self.plot_widget, 1)
        # Slider de escala visible solo cuando se muestra la deformada
        self.scale_widget = QWidget()
        sl = QHBoxLayout(self.scale_widget)
        sl.setContentsMargins(0, 4, 0, 0)
        sl.addWidget(QLabel("Escala visual de la deformada:"))
        self.scale_slider = QSlider(Qt.Horizontal)
        self.scale_slider.setMinimum(0)
        self.scale_slider.setMaximum(200)
        self.scale_slider.setValue(100)
        self.scale_slider.setToolTip(
            "<b>Escala visual de la deformada</b><br>"
            "Los desplazamientos reales suelen ser microscópicos (10⁻⁶ m). "
            "Este slider los amplifica visualmente para que la deformada "
            "sea apreciable en pantalla.<br>"
            "<b>No afecta el cálculo</b>, solo la visualización."
        )
        self.scale_value_lbl = QLabel("automática")
        self.scale_value_lbl.setMinimumWidth(80)
        self.scale_slider.valueChanged.connect(self._on_scale_changed)
        sl.addWidget(self.scale_slider, 1)
        sl.addWidget(self.scale_value_lbl)
        self.scale_widget.setVisible(False)
        pl.addWidget(self.scale_widget)
        self._deformed_auto_scale: float | None = None

        # Animador de carga (también visible solo en paso de deformada)
        from .load_animator import LoadAnimator
        self.load_animator = LoadAnimator()
        self.load_animator.setVisible(False)
        self.load_animator.progress_changed.connect(self._on_load_progress)
        self._load_progress_percent: float = 100.0
        pl.addWidget(self.load_animator)

        # Selector de campo para el mapa de contornos (visible solo en paso de esfuerzos)
        self.contour_widget = QWidget()
        cw = QHBoxLayout(self.contour_widget)
        cw.setContentsMargins(0, 4, 0, 0)
        # Selector de modo de visualización
        cw.addWidget(QLabel("Visualización:"))
        self.vis_mode_combo = QComboBox()
        self.vis_mode_combo.addItems(["Contornos", "Esfuerzos principales"])
        self.vis_mode_combo.setToolTip(
            "<b>Modo de visualización de esfuerzos</b><br><br>"
            "<b>Contornos:</b> mapa de color continuo sobre el elemento (σx, σy, σ_VM, etc.).<br>"
            "<b>Esfuerzos principales:</b> flechas de σ₁ (tracción, rojo) y σ₂ "
            "(compresión, azul) en cada punto de Gauss, orientadas según θp."
        )
        self.vis_mode_combo.currentTextChanged.connect(self._on_vis_mode_changed)
        cw.addWidget(self.vis_mode_combo)
        cw.addSpacing(12)

        cw.addWidget(QLabel("Campo:"))
        self.contour_combo = QComboBox()
        for k in STRESS_FIELDS.keys():
            self.contour_combo.addItem(k)
        self.contour_combo.setCurrentText("σ_VM")
        self.contour_combo.setToolTip(
            "<b>Campo a graficar como mapa de contornos</b><br><br>"
            "<b>σx, σy:</b> esfuerzos normales horizontal/vertical.<br>"
            "<b>τxy:</b> esfuerzo cortante.<br>"
            "<b>σ_VM (von Mises):</b> esfuerzo equivalente — el criterio "
            "más común para evaluar fluencia en metales.<br>"
            "<b>σ1, σ2:</b> esfuerzos principales (eigenvalores de σ)."
        )
        self.contour_combo.currentTextChanged.connect(self._on_contour_field_changed)
        cw.addWidget(self.contour_combo)

        # Botón toggle del Probe Tool (paso de esfuerzos)
        cw.addSpacing(16)
        self.btn_probe = QPushButton(" Sonda")
        self.btn_probe.setIcon(icon("fa5s.crosshairs"))
        self.btn_probe.setIconSize(QSize(14, 14))
        self.btn_probe.setCheckable(True)
        self.btn_probe.setToolTip(
            "<b>Herramienta de sonda</b><br>"
            "Activa el modo medición: clic en cualquier punto del elemento "
            "para ver los valores de σx, σy, τxy, εx, εy, γxy interpolados ahí.<br><br>"
            "Click de nuevo para desactivar."
        )
        self.btn_probe.toggled.connect(self._on_probe_toggled)
        cw.addWidget(self.btn_probe)

        cw.addStretch()
        self.contour_widget.setVisible(False)
        pl.addWidget(self.contour_widget)

        # ProbeHandler se crea de manera diferida (cuando hay resultados)
        self._probe_handler = None

        content_split.addWidget(plot_box)

        # ----- Zona inferior: QTabWidget con Tablas + Matrices -----
        self.data_tabs = QTabWidget()
        self.data_tabs.setDocumentMode(True)

        # --- Tab 1: Tablas (con scroll para múltiples tablas)
        tables_tab = QWidget()
        tt = QVBoxLayout(tables_tab)
        tt.setContentsMargins(4, 4, 4, 4)
        self.tables_scroll = QScrollArea()
        self.tables_scroll.setWidgetResizable(True)
        self.tables_scroll.setFrameShape(QFrame.NoFrame)
        self.tables_container = QWidget()
        self.tables_layout = QVBoxLayout(self.tables_container)
        self.tables_layout.setSpacing(12)
        self.tables_layout.setContentsMargins(4, 4, 4, 4)
        self.tables_scroll.setWidget(self.tables_container)
        tt.addWidget(self.tables_scroll)
        self.data_tabs.addTab(tables_tab, "Tablas")

        # --- Tab 2: Matrices / vectores (cada matriz como tabla visual)
        matrices_tab = QWidget()
        mt = QVBoxLayout(matrices_tab)
        mt.setContentsMargins(4, 4, 4, 4)
        self.matrices_scroll = QScrollArea()
        self.matrices_scroll.setWidgetResizable(True)
        self.matrices_scroll.setFrameShape(QFrame.NoFrame)
        self.matrices_container = QWidget()
        self.matrices_layout = QVBoxLayout(self.matrices_container)
        self.matrices_layout.setSpacing(12)
        self.matrices_layout.setContentsMargins(4, 4, 4, 4)
        self.matrices_scroll.setWidget(self.matrices_container)
        mt.addWidget(self.matrices_scroll)
        self.data_tabs.addTab(matrices_tab, "Matrices / vectores")
        # Recordar índices para mostrar/ocultar dinámicamente
        self._tab_idx_tables = 0
        self._tab_idx_matrices = 1

        content_split.addWidget(self.data_tabs)

        # Pesos iniciales: gráfica ocupa ~55%, datos ~45%
        content_split.setStretchFactor(0, 11)
        content_split.setStretchFactor(1, 9)
        content_split.setSizes([550, 450])
        right_layout.addWidget(content_split, 1)

        splitter_main.addWidget(left)
        splitter_main.addWidget(right)
        splitter_main.setSizes([340, 1040])

        return splitter_main

    # ---------------- Panel de edición ----------------
    def _build_input_panel(self):
        # Limpiar
        while self.input_box_layout.count():
            it = self.input_box_layout.takeAt(0)
            w = it.widget()
            if w:
                w.setParent(None)
        self.node_inputs = []

        if not self.structure or not self.structure.elements:
            return

        # ---- Nodos editables ----
        n_nodos = len(self.structure.nodes)
        nodos_lbl = QLabel(
            f"<b>Nodos</b> ({n_nodos}) — doble-clic en una celda para editar"
        )
        self.input_box_layout.addWidget(nodos_lbl)

        # Tabla scrollable para edición de nodos (escala bien con muchos nodos)
        from PySide6.QtWidgets import QTableWidget, QTableWidgetItem, QAbstractItemView
        self.nodes_table = QTableWidget(n_nodos, 7)
        self.nodes_table.setHorizontalHeaderLabels(
            ["#", "x (m)", "y (m)", "Fix X", "Fix Y", "Fx (N)", "Fy (N)"]
        )
        self.nodes_table.verticalHeader().setVisible(False)
        self.nodes_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.nodes_table.setEditTriggers(
            QAbstractItemView.DoubleClicked | QAbstractItemView.SelectedClicked
            | QAbstractItemView.EditKeyPressed
        )
        self.nodes_table.setAlternatingRowColors(True)
        self.nodes_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        # Altura responsiva: hasta ~8 filas visibles antes de habilitar scroll
        row_h = 26
        header_h = self.nodes_table.horizontalHeader().height()
        max_visible = 8
        target_h = header_h + min(n_nodos, max_visible) * row_h + 8
        self.nodes_table.setMinimumHeight(target_h)
        self.nodes_table.setMaximumHeight(target_h + 4)

        for r, n in enumerate(self.structure.nodes):
            # Columna # (sólo lectura)
            id_item = QTableWidgetItem(str(n.id + 1))
            id_item.setFlags(id_item.flags() & ~Qt.ItemIsEditable)
            id_item.setTextAlignment(Qt.AlignCenter)
            id_item.setData(Qt.UserRole, n.id)
            self.nodes_table.setItem(r, 0, id_item)
            # x, y (editables)
            for col, val in enumerate((n.x, n.y), start=1):
                it = QTableWidgetItem(_fmt_input(val))
                it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.nodes_table.setItem(r, col, it)
            # Fix X / Fix Y como checkboxes centrados
            for col, val in zip((3, 4), (n.restraint_x, n.restraint_y)):
                cb = QCheckBox()
                cb.setChecked(val)
                holder = QWidget()
                hl = QHBoxLayout(holder)
                hl.setContentsMargins(0, 0, 0, 0)
                hl.setAlignment(Qt.AlignCenter)
                hl.addWidget(cb)
                self.nodes_table.setCellWidget(r, col, holder)
                # Guardamos referencia al checkbox para leerlo después
                setattr(cb, "_node_id", n.id)
            # Fx, Fy (editables)
            for col, val in zip((5, 6), (n.load_x, n.load_y)):
                it = QTableWidgetItem(_fmt_input(val))
                it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.nodes_table.setItem(r, col, it)

        self.input_box_layout.addWidget(self.nodes_table)

        # ---- Propiedades del elemento ----
        self.input_box_layout.addWidget(QLabel("<b>Propiedades del elemento</b>"))
        form = QFormLayout()
        el0 = self.structure.elements[0]
        self.E_input = QLineEdit(_fmt_input(el0.E))
        self.E_input.setToolTip(
            "<b>Módulo de elasticidad E (Pa)</b><br>"
            "Rigidez del material. Valores típicos:<br>"
            "• Acero: 2.1 × 10¹¹ Pa<br>"
            "• Hormigón: 2.5 × 10¹⁰ Pa<br>"
            "• Aluminio: 7.0 × 10¹⁰ Pa<br>"
            "Determina la magnitud de los desplazamientos: a mayor E, "
            "estructura más rígida y menores deformaciones."
        )
        self.nu_input = QLineEdit(_fmt_input(el0.nu))
        self.nu_input.setToolTip(
            "<b>Coeficiente de Poisson ν</b><br>"
            "Adimensional, típicamente 0.0 ≤ ν < 0.5. Mide la contracción "
            "lateral relativa cuando se aplica tracción axial.<br>"
            "• Acero: 0.30<br>"
            "• Hormigón: 0.20<br>"
            "• Caucho: ≈ 0.49 (casi incompresible)"
        )
        self.t_input = QLineEdit(_fmt_input(el0.t))
        self.t_input.setToolTip(
            "<b>Espesor t (m)</b><br>"
            "Espesor de la placa fuera del plano (en dirección Z).<br>"
            "En tensión plana: el espesor afecta linealmente la matriz K^e "
            "(K^e ∝ t)."
        )
        for w in (self.E_input, self.nu_input, self.t_input):
            w.setMaximumWidth(180)
        self.plane_combo = QComboBox()
        self.plane_combo.addItems(["Tensión plana", "Deformación plana"])
        self.plane_combo.setCurrentIndex(0 if el0.plane_stress else 1)
        self.plane_combo.setToolTip(
            "<b>Hipótesis del problema plano</b><br><br>"
            "<b>Tensión plana:</b> σz = τxz = τyz = 0.<br>"
            "Apropiada para <i>placas delgadas</i> cargadas en su plano "
            "(ej. una chapa, una membrana).<br><br>"
            "<b>Deformación plana:</b> εz = γxz = γyz = 0.<br>"
            "Apropiada para <i>sólidos largos</i> cuya sección transversal "
            "no cambia (ej. túneles, presas largas, tuberías)."
        )
        form.addRow("E (Pa):", self.E_input)
        form.addRow("ν (Poisson):", self.nu_input)
        form.addRow("t (m):", self.t_input)
        form.addRow("Hipótesis:", self.plane_combo)

        # Botón de biblioteca de materiales
        btn_material = QPushButton(" Elegir material…")
        btn_material.setIcon(icon("fa5s.flask"))
        btn_material.setIconSize(QSize(14, 14))
        btn_material.setToolTip(
            "<b>Biblioteca de materiales</b><br>"
            "Selecciona un material del catálogo (Acero, Hormigón, Aluminio…) "
            "y el aplicativo copia automáticamente E y ν a los campos de arriba."
        )
        btn_material.clicked.connect(self._on_choose_material)
        form.addRow("Material:", btn_material)
        form_wrap = QWidget()
        form_wrap.setLayout(form)
        self.input_box_layout.addWidget(form_wrap)

    def _apply_input_panel(self) -> bool:
        """Lee los inputs y reconstruye la estructura. Devuelve True si OK."""
        if not self.structure:
            return False
        try:
            # Leer nodos desde la QTableWidget
            if hasattr(self, "nodes_table") and self.nodes_table is not None:
                for r in range(self.nodes_table.rowCount()):
                    id_item = self.nodes_table.item(r, 0)
                    if id_item is None:
                        continue
                    nid = id_item.data(Qt.UserRole)
                    node = next((n for n in self.structure.nodes if n.id == nid), None)
                    if node is None:
                        continue
                    node.x = float(self.nodes_table.item(r, 1).text())
                    node.y = float(self.nodes_table.item(r, 2).text())
                    # Checkboxes (en cellWidget)
                    fx_holder = self.nodes_table.cellWidget(r, 3)
                    fy_holder = self.nodes_table.cellWidget(r, 4)
                    if fx_holder is not None:
                        cb = fx_holder.findChild(QCheckBox)
                        node.restraint_x = cb.isChecked() if cb else node.restraint_x
                    if fy_holder is not None:
                        cb = fy_holder.findChild(QCheckBox)
                        node.restraint_y = cb.isChecked() if cb else node.restraint_y
                    node.load_x = float(self.nodes_table.item(r, 5).text() or 0)
                    node.load_y = float(self.nodes_table.item(r, 6).text() or 0)
            E = float(self.E_input.text())
            nu = float(self.nu_input.text())
            t = float(self.t_input.text())
            plane_stress = self.plane_combo.currentIndex() == 0
            for el in self.structure.elements:
                el.E = E
                el.nu = nu
                el.t = t
                el.plane_stress = plane_stress
            return True
        except (ValueError, StopIteration) as e:
            QMessageBox.critical(self, "Error en datos de entrada", str(e))
            return False

    # ---------------- Carga de datos ----------------
    def _load_default_example(self):
        path = Path(__file__).resolve().parents[2] / "examples" / "q4_placa.json"
        if path.exists():
            self.load_from_json(path)

    def load_from_json(self, path: Path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        structure = Structure()
        nodes_by_id: dict[int, Node] = {}
        for nd in data["nodes"]:
            node = Node(
                id=nd["id"],
                x=nd["x"], y=nd["y"],
                restraint_x=nd.get("restraint_x", False),
                restraint_y=nd.get("restraint_y", False),
                load_x=nd.get("load_x", 0.0),
                load_y=nd.get("load_y", 0.0),
            )
            structure.add_node(node)
            nodes_by_id[node.id] = node

        for el in data["elements"]:
            element_nodes = [nodes_by_id[i] for i in el["nodes"]]
            structure.add_element(Q4Element(
                id=el["id"],
                nodes=element_nodes,
                E=el["E"],
                nu=el["nu"],
                t=el["t"],
                plane_stress=el.get("plane_stress", True),
            ))

        self.structure = structure
        # Sincronizar editor gráfico con la misma estructura
        if hasattr(self, "canvas_editor"):
            self.canvas_editor.set_structure(structure)
            if structure.elements:
                el0 = structure.elements[0]
                self.canvas_editor.set_default_properties(
                    el0.E, el0.nu, el0.t, el0.plane_stress
                )
        self._build_input_panel()
        self.recalculate()
        self.statusBar().showMessage(
            f"Ejemplo cargado: {path.name} — {len(structure.nodes)} nodos, "
            f"{len(structure.elements)} elemento(s) Q4"
        )

    # ---------------- Recalcular ----------------
    def recalculate(self):
        if not self.structure:
            return
        if hasattr(self, "nodes_table") and self.nodes_table is not None and not self._apply_input_panel():
            return
        # Feedback visual mientras el programa calcula: cursor de espera y
        # mensaje de estado. Al terminar se muestra CUANTO tardo el calculo
        # (ensamblaje + solucion + generacion de los 15 pasos).
        import time
        from PySide6.QtGui import QCursor
        self.statusBar().showMessage("Calculando...")
        QApplication.setOverrideCursor(QCursor(Qt.CursorShape.WaitCursor))
        QApplication.processEvents()   # pinta el mensaje antes de bloquear
        t0 = time.perf_counter()
        try:
            self.procedure = build_procedure(self.structure)
        except Exception as e:
            self.statusBar().showMessage("Error en el cálculo")
            QMessageBox.critical(self, "Error en el cálculo", str(e))
            return
        finally:
            QApplication.restoreOverrideCursor()
        dt_calculo = time.perf_counter() - t0

        self._populate_steps()
        # Sincronizar editor (sin disparar señales para no loopear)
        if hasattr(self, "canvas_editor"):
            self.canvas_editor.blockSignals(True)
            self.canvas_editor.set_structure(self.structure)
            if self.structure.elements:
                el0 = self.structure.elements[0]
                self.canvas_editor.set_default_properties(
                    el0.E, el0.nu, el0.t, el0.plane_stress
                )
            self.canvas_editor.blockSignals(False)
        # Sincronizar vista 3D con la nueva estructura + deformada
        if hasattr(self, "canvas_3d"):
            disp = self.procedure.result.displacements if self.procedure and self.procedure.result else None
            self.canvas_3d.set_structure(self.structure, disp)
            self.canvas_3d.fit_view()
            # Refrescar coloreo de esfuerzos si hay un campo seleccionado
            self._update_3d_stress()
        # Tiempo en formato intuitivo: milisegundos si fue rapido,
        # segundos con 2 decimales si tardo mas.
        if dt_calculo < 1.0:
            t_txt = f"{dt_calculo * 1000:.0f} ms"
        else:
            t_txt = f"{dt_calculo:.2f} s"
        n_dof = self.structure.n_dofs
        n_el = len(self.structure.elements)
        self.statusBar().showMessage(
            f"Cálculo completado en {t_txt} — "
            f"{n_el} elemento{'s' if n_el != 1 else ''}, {n_dof} GDL"
        )

    # ---------------- Vista 3D del Q4 plano ----------------
    def _update_3d_stress(self, field_name: str | None = None):
        """Calcula el campo de esfuerzo seleccionado promediado por elemento y
        lo pasa al canvas 3D para colorear las caras."""
        if not hasattr(self, "canvas_3d") or not hasattr(self, "cmb3d_stress"):
            return
        if field_name is None:
            field_name = self.cmb3d_stress.currentText()
        # "(Ninguno)" o sin resultado: limpiar coloreo
        if field_name == "(Ninguno)" or not self.procedure or not self.procedure.result:
            self.canvas_3d.set_stress_field(None, None, 0.0, 1.0)
            self.lbl3d_stress_range.setText("(sin datos)")
            return
        if field_name not in STRESS_FIELDS:
            return
        getter = STRESS_FIELDS[field_name]
        per_elem = []
        for er in self.procedure.result.elements:
            # Promedio de los 4 puntos de Gauss del elemento (suaviza)
            vals = [getter(sig) for sig in er.stresses_at_gauss]
            per_elem.append(float(np.mean(vals)))
        arr = np.array(per_elem)
        vmin = float(arr.min())
        vmax = float(arr.max())
        self.canvas_3d.set_stress_field(field_name, arr, vmin, vmax)
        unit = "Pa"
        self.lbl3d_stress_range.setText(
            f"{field_name}: [{vmin:+.3e}, {vmax:+.3e}] {unit}"
        )

    def _build_view3d_tab(self):
        """Tab con el Canvas3DQ4 + panel de controles de visualización."""
        from PySide6.QtWidgets import QGroupBox, QHBoxLayout as _H, QVBoxLayout as _V
        tab = QWidget()
        root = _H(tab)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        # Canvas 3D al centro
        self.canvas_3d = Canvas3DQ4()
        root.addWidget(self.canvas_3d, 1)

        # Panel derecho de controles
        right = QWidget()
        col = _V(right)
        col.setSpacing(10)

        gb = QGroupBox("Visualización")
        gv = _V(gb)
        self.cb3d_undef = QCheckBox("Malla original")
        self.cb3d_undef.setChecked(True)
        self.cb3d_undef.toggled.connect(self.canvas_3d.set_show_undeformed)
        self.cb3d_def = QCheckBox("Deformada (rojo)")
        self.cb3d_def.setChecked(True)
        self.cb3d_def.toggled.connect(self.canvas_3d.set_show_deformed)
        self.cb3d_faces = QCheckBox("Caras semitransparentes")
        self.cb3d_faces.setChecked(True)
        self.cb3d_faces.toggled.connect(self.canvas_3d.set_show_faces)
        self.cb3d_supp = QCheckBox("Apoyos")
        self.cb3d_supp.setChecked(True)
        self.cb3d_supp.toggled.connect(self.canvas_3d.set_show_supports)
        self.cb3d_loads = QCheckBox("Cargas")
        self.cb3d_loads.setChecked(True)
        self.cb3d_loads.toggled.connect(self.canvas_3d.set_show_loads)
        self.cb3d_labels = QCheckBox("Etiquetas de nodos")
        self.cb3d_labels.setChecked(False)
        self.cb3d_labels.toggled.connect(self.canvas_3d.set_show_labels)
        for cb in (self.cb3d_undef, self.cb3d_def, self.cb3d_faces,
                   self.cb3d_supp, self.cb3d_loads, self.cb3d_labels):
            gv.addWidget(cb)
        col.addWidget(gb)

        gb2 = QGroupBox("Escala de deformada")
        gv2 = _V(gb2)
        self.sl3d_scale = QSlider(Qt.Orientation.Horizontal)
        self.sl3d_scale.setRange(0, 10000)
        self.sl3d_scale.setValue(100)
        self.lbl3d_scale = QLabel("x 100")
        def _on_scale(v):
            self.lbl3d_scale.setText(f"x {v}")
            self.canvas_3d.set_scale(float(v))
        self.sl3d_scale.valueChanged.connect(_on_scale)
        gv2.addWidget(self.sl3d_scale)
        gv2.addWidget(self.lbl3d_scale)
        col.addWidget(gb2)

        # Selector de campo de esfuerzos para colorear las caras
        gb3 = QGroupBox("Mapa de esfuerzos")
        gv3 = _V(gb3)
        self.cmb3d_stress = QComboBox()
        self.cmb3d_stress.addItem("(Ninguno)")
        for fname in STRESS_FIELDS.keys():
            self.cmb3d_stress.addItem(fname)
        self.cmb3d_stress.currentTextChanged.connect(self._update_3d_stress)
        gv3.addWidget(self.cmb3d_stress)
        self.lbl3d_stress_range = QLabel("(sin datos)")
        self.lbl3d_stress_range.setStyleSheet("color: #666; font-size: 11px;")
        gv3.addWidget(self.lbl3d_stress_range)
        col.addWidget(gb3)

        btn_fit = QPushButton("Centrar vista")
        btn_fit.clicked.connect(self.canvas_3d.fit_view)
        col.addWidget(btn_fit)

        # Pista de interaccion (picking + identificacion de ejes)
        lbl_hint = QLabel(
            "Clic sobre un nodo o sobre el área de un elemento para ver "
            "su información junto al cursor.\n\n"
            "Ejes: X (rojo), Y (verde), Z (azul)."
        )
        lbl_hint.setWordWrap(True)
        lbl_hint.setStyleSheet("color: #666; font-size: 11px;")
        col.addWidget(lbl_hint)

        col.addStretch(1)
        right.setMaximumWidth(240)
        root.addWidget(right, 0)

        # Inicializar con la estructura actual si ya existe
        if self.structure is not None:
            disp = self.procedure.result.displacements if (
                self.procedure and self.procedure.result) else None
            self.canvas_3d.set_structure(self.structure, disp)
            self.canvas_3d.set_scale(100.0)
            self.canvas_3d.fit_view()

        return tab

    def _populate_steps(self):
        current = self.steps_list.currentRow()
        self.steps_list.clear()
        if not self.procedure:
            self._update_step_indicator()
            return
        import qtawesome as qta
        # Iconos según tipo de paso (mostrar pictogramas a la izquierda)
        plot_icons = {
            "mesh": "fa5s.draw-polygon",
            "bc": "fa5s.anchor",
            "deformed": "fa5s.expand-arrows-alt",
            "stress": "fa5s.thermometer-half",
            "stress_corners": "fa5s.dot-circle",
        }
        for st in self.procedure.steps:
            item = QListWidgetItem(st.title)
            # Icono según el plot_key (o genérico si solo tiene matrices)
            if st.plot_key in plot_icons:
                item.setIcon(qta.icon(plot_icons[st.plot_key],
                                      color=Colors.TEXT_MUTED))
            elif st.matrices:
                item.setIcon(qta.icon("fa5s.calculator",
                                      color=Colors.TEXT_MUTED))
            elif st.tables:
                item.setIcon(qta.icon("fa5s.table",
                                      color=Colors.TEXT_MUTED))
            self.steps_list.addItem(item)
        # Mantener el paso actual si es posible
        if 0 <= current < len(self.procedure.steps):
            self.steps_list.setCurrentRow(current)
        else:
            self.steps_list.setCurrentRow(0)

    # ---------------- Navegación paso anterior / siguiente ----------------
    def _step_prev(self) -> None:
        cur = self.steps_list.currentRow()
        if cur > 0:
            self.steps_list.setCurrentRow(cur - 1)

    def _step_next(self) -> None:
        cur = self.steps_list.currentRow()
        if cur < self.steps_list.count() - 1:
            self.steps_list.setCurrentRow(cur + 1)

    def _update_step_indicator(self) -> None:
        total = self.steps_list.count()
        cur = self.steps_list.currentRow()
        if total == 0:
            self.step_indicator.setText("Sin pasos")
            self.btn_prev.setEnabled(False)
            self.btn_next.setEnabled(False)
            return
        self.step_indicator.setText(f"Paso {cur + 1} / {total}")
        self.btn_prev.setEnabled(cur > 0)
        self.btn_next.setEnabled(cur < total - 1)

    # ---------------- Mostrar paso ----------------
    def show_step(self, index: int):
        if not self.procedure or index < 0 or index >= len(self.procedure.steps):
            self._update_step_indicator()
            return
        step = self.procedure.steps[index]
        self.title_label.setText(step.title)
        self.desc_label.setText(step.description)
        self._update_step_indicator()

        # ---- Limpiar tablas ----
        while self.tables_layout.count():
            it = self.tables_layout.takeAt(0)
            w = it.widget()
            if w:
                w.setParent(None)

        has_tables = bool(step.tables)
        has_matrices = bool(step.matrices)

        # ---- Tablas ----
        if has_tables:
            for name, df in step.tables.items():
                lbl = QLabel(f"<b>{name}</b>")
                lbl.setStyleSheet(f"color:{Colors.PRIMARY}; font-size:11pt; padding-top:6px;")
                self.tables_layout.addWidget(lbl)
                model = df_to_qt_model(df, editable=False)
                view = make_table_view(model, min_rows=len(df) + 1)
                self.tables_layout.addWidget(view)
            self.tables_layout.addStretch()

        # ---- Matrices (cada una como tabla en su propio QGroupBox) ----
        while self.matrices_layout.count():
            it = self.matrices_layout.takeAt(0)
            w = it.widget()
            if w:
                w.setParent(None)
        if has_matrices:
            for name, M in step.matrices.items():
                self.matrices_layout.addWidget(matrix_to_widget(name, M))
            self.matrices_layout.addStretch()

        # ---- Mostrar/ocultar tabs según el contenido del paso ----
        # Resetear visibilidad
        self.data_tabs.setTabVisible(self._tab_idx_tables, has_tables)
        self.data_tabs.setTabVisible(self._tab_idx_matrices, has_matrices)
        # Etiquetas con conteo para que se sepa cuánto hay
        n_t = len(step.tables) if has_tables else 0
        n_m = len(step.matrices) if has_matrices else 0
        self.data_tabs.setTabText(self._tab_idx_tables,
                                  f"Tablas ({n_t})" if has_tables else "Tablas")
        self.data_tabs.setTabText(self._tab_idx_matrices,
                                  f"Matrices / vectores ({n_m})" if has_matrices else "Matrices / vectores")
        # Seleccionar la tab que tenga contenido (preferir Tablas si hay ambas)
        if has_tables:
            self.data_tabs.setCurrentIndex(self._tab_idx_tables)
        elif has_matrices:
            self.data_tabs.setCurrentIndex(self._tab_idx_matrices)

        # Gráfica + widgets contextuales según el plot_key
        result = self.procedure.result if self.procedure else None
        is_deformed = step.plot_key == "deformed"
        is_stress = step.plot_key == "stress"
        has_plot = bool(step.plot_key)

        # Ocultar el GroupBox entero de la gráfica si el paso no lo usa,
        # para que los datos (tablas/matrices) ocupen todo el espacio.
        self.plot_box.setVisible(has_plot)
        self.scale_widget.setVisible(is_deformed)
        self.contour_widget.setVisible(is_stress)
        self.load_animator.setVisible(is_deformed)

        # Al salir del paso de esfuerzos, apagar Probe si quedó activo
        if not is_stress and hasattr(self, "btn_probe") and self.btn_probe.isChecked():
            self.btn_probe.setChecked(False)
        # Si entramos al paso de deformada, resetear animator a 100% (estado actual)
        if is_deformed:
            self.load_animator.set_percent(100.0)
            self._load_progress_percent = 100.0

        if not has_plot:
            self.plot_widget.clear()
        elif is_deformed:
            if result is not None and self.structure is not None:
                xs = [n.x for n in self.structure.nodes]
                ys = [n.y for n in self.structure.nodes]
                size = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
                max_disp = float(np.max(np.abs(result.displacements))) or 1.0
                self._deformed_auto_scale = 0.15 * size / max_disp if max_disp > 0 else 1.0
            self._render_deformed_with_current_scale()
        elif is_stress and result is not None:
            self._render_stress_contour()
        elif self.structure is not None:
            self.plot_widget.render_plot_key(step.plot_key, self.structure, result)

    def _on_contour_field_changed(self, _txt: str) -> None:
        self._render_stress_contour()

    def _render_stress_contour(self) -> None:
        if not (self.procedure and self.structure):
            return
        field = self.contour_combo.currentText()
        vis_mode = self.vis_mode_combo.currentText() if hasattr(self, "vis_mode_combo") else "Contornos"
        # El selector de campo solo aplica al modo "Contornos"
        if hasattr(self, "contour_combo"):
            self.contour_combo.setEnabled(vis_mode == "Contornos")
        self.plot_widget.render_plot_key(
            "stress", self.structure, self.procedure.result,
            contour_field=field, vis_mode=vis_mode,
        )

    def _on_vis_mode_changed(self, _txt: str) -> None:
        self._render_stress_contour()

    def _on_scale_changed(self, val: int) -> None:
        """Slider va de 0 a 200; 100 = escala automática.
        0 → sin deformación; 200 → escala doble de la automática."""
        if val == 100:
            self.scale_value_lbl.setText("automática")
        elif val == 0:
            self.scale_value_lbl.setText("× 0  (sin def.)")
        else:
            factor = val / 100.0
            self.scale_value_lbl.setText(f"× {factor:.2f}")
        self._render_deformed_with_current_scale()

    def _render_deformed_with_current_scale(self) -> None:
        if not (self.procedure and self.structure and self._deformed_auto_scale):
            return
        val = self.scale_slider.value()
        # Combinar escala visual del slider con porcentaje de carga del animator
        load_factor = self._load_progress_percent / 100.0
        scale = self._deformed_auto_scale * (val / 100.0) * load_factor
        self.plot_widget.render_plot_key(
            "deformed", self.structure, self.procedure.result, scale=scale
        )

    def _on_load_progress(self, percent: float) -> None:
        """Llamado por LoadAnimator cuando cambia el % de carga aplicada."""
        self._load_progress_percent = float(percent)
        self._render_deformed_with_current_scale()

    def _on_probe_toggled(self, checked: bool) -> None:
        """Activa/desactiva el modo Probe en el paso de esfuerzos."""
        from .probe_handler import ProbeHandler
        if checked:
            if not (self.procedure and self.structure):
                self.btn_probe.setChecked(False)
                return
            if self._probe_handler is None:
                self._probe_handler = ProbeHandler(
                    self.plot_widget, self.structure, self.procedure.result
                )
            else:
                self._probe_handler.set_structure_and_result(
                    self.structure, self.procedure.result
                )
            self._probe_handler.enable()
            self.statusBar().showMessage(
                "Modo Sonda activado — clic en el lienzo para medir σ, ε", 4000
            )
        else:
            if self._probe_handler is not None:
                self._probe_handler.disable()
            self.statusBar().showMessage("Modo Sonda desactivado", 2000)

    # ---------------- Acciones de archivo ----------------
    def open_example(self):
        """Alias retrocompat — abre cualquier .json compatible."""
        self.open_project()

    # ---------------- Proyecto: nuevo / abrir / guardar ----------------
    def new_project(self) -> None:
        """Estructura vacía."""
        self.structure = Structure()
        self.procedure = None
        self._current_project_path = None
        self.steps_list.clear()
        if hasattr(self, "canvas_editor"):
            self.canvas_editor.set_structure(self.structure)
        self._build_input_panel()
        self.setWindowTitle("MEF Q4 — Trabajo de Tesis — (nuevo proyecto)")
        self.statusBar().showMessage(
            "Nuevo proyecto vacío. Ve al editor gráfico para crear nodos y elementos."
        )

    def open_project(self) -> None:
        path_str, _ = QFileDialog.getOpenFileName(
            self, "Abrir proyecto", str(Path.cwd()), "Proyecto MEF (*.json)"
        )
        if not path_str:
            return
        self._load_project_path(Path(path_str))

    def _load_project_path(self, path: Path) -> None:
        """Carga el proyecto desde la ruta dada."""
        try:
            structure, description = load_project(path)
        except Exception as e:
            QMessageBox.critical(self, "Error al cargar", str(e))
            return
        self.structure = structure
        self._current_project_path = path
        if hasattr(self, "canvas_editor"):
            self.canvas_editor.set_structure(structure)
            if structure.elements:
                el0 = structure.elements[0]
                self.canvas_editor.set_default_properties(
                    el0.E, el0.nu, el0.t, el0.plane_stress
                )
        self._build_input_panel()
        if structure.nodes and structure.elements:
            self.recalculate()
        else:
            self.procedure = None
            self.steps_list.clear()
        self._add_to_recent(path)
        self.setWindowTitle(f"MEF Q4 — Trabajo de Tesis — {path.name}")
        self.statusBar().showMessage(
            f"Proyecto cargado: {path.name} — {len(structure.nodes)} nodos, "
            f"{len(structure.elements)} elemento(s)"
        )

    def save_project(self) -> None:
        """Guarda en la ruta actual o pregunta si no hay."""
        if self._current_project_path is None:
            self.save_project_as()
            return
        try:
            save_project(self.structure, self._current_project_path,
                         description=f"Guardado desde la app")
            self.statusBar().showMessage(
                f"Guardado: {self._current_project_path.name}", 3000
            )
        except Exception as e:
            QMessageBox.critical(self, "Error al guardar", str(e))

    def save_project_as(self) -> None:
        suggested = str(self._current_project_path or "proyecto_mef.json")
        path_str, _ = QFileDialog.getSaveFileName(
            self, "Guardar proyecto como…", suggested, "Proyecto MEF (*.json)"
        )
        if not path_str:
            return
        path = Path(path_str)
        try:
            save_project(self.structure, path,
                         description="Guardado desde la app")
            self._current_project_path = path
            self._add_to_recent(path)
            self.setWindowTitle(f"MEF Q4 — Trabajo de Tesis — {path.name}")
            self.statusBar().showMessage(f"Guardado: {path.name}", 3000)
        except Exception as e:
            QMessageBox.critical(self, "Error al guardar", str(e))

    # ---------------- Manejo de recientes ----------------
    def _read_recent_list(self) -> list[str]:
        if not RECENT_FILES_PATH.exists():
            return []
        try:
            with open(RECENT_FILES_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                return [p for p in data.get("recent", []) if Path(p).exists()]
        except Exception:
            return []

    def _write_recent_list(self, paths: list[str]) -> None:
        try:
            with open(RECENT_FILES_PATH, "w", encoding="utf-8") as f:
                json.dump({"recent": paths}, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _add_to_recent(self, path: Path) -> None:
        recent = self._read_recent_list()
        path_str = str(path.resolve())
        if path_str in recent:
            recent.remove(path_str)
        recent.insert(0, path_str)
        recent = recent[:MAX_RECENT_FILES]
        self._write_recent_list(recent)
        self._refresh_recent_menu()

    def _refresh_recent_menu(self) -> None:
        if not hasattr(self, "recent_menu"):
            return
        self.recent_menu.clear()
        recent = self._read_recent_list()
        if not recent:
            a = QAction("(sin proyectos recientes)", self)
            a.setEnabled(False)
            self.recent_menu.addAction(a)
            return
        for path_str in recent:
            p = Path(path_str)
            a = QAction(p.name, self)
            a.setToolTip(str(p))
            a.triggered.connect(lambda _, pp=p: self._load_project_path(pp))
            self.recent_menu.addAction(a)
        self.recent_menu.addSeparator()
        clear = QAction("Limpiar lista", self)
        clear.triggered.connect(lambda: (self._write_recent_list([]),
                                         self._refresh_recent_menu()))
        self.recent_menu.addAction(clear)

    def export_excel(self):
        if not self.procedure:
            return
        # Preguntar formato
        from PySide6.QtWidgets import QMessageBox as _MB
        msg = _MB(self)
        msg.setIcon(_MB.Question)
        msg.setWindowTitle("Formato Excel")
        msg.setText("¿Qué formato de Excel preferís?")
        msg.setInformativeText(
            "• Por pasos (didáctico): una hoja por cada uno de los 15 pasos del análisis.\n"
            "  Ideal para entender el procedimiento paso a paso.\n\n"
            "• Por categorías (datos): hojas separadas — Datos_entrada / Resultados_nodales /\n"
            "  Esfuerzos_GP / Esfuerzos_esquinas / matrices clave.\n"
            "  Ideal para post-procesar datos en Excel/MATLAB/Python/R."
        )
        btn_steps = msg.addButton("Por pasos (didáctico)", _MB.AcceptRole)
        btn_org = msg.addButton("Por categorías (datos)", _MB.AcceptRole)
        msg.addButton(_MB.Cancel)
        msg.exec()
        clicked = msg.clickedButton()
        if clicked is None or clicked.text() == "Cancel":
            return
        organized = (clicked is btn_org)
        # Sugerir nombre acorde al formato
        suggested = "resultados_mef_categorias.xlsx" if organized else "resultados_mef_pasos.xlsx"
        path_str, _ = QFileDialog.getSaveFileName(
            self, "Guardar Excel", suggested, "Excel (*.xlsx)"
        )
        if not path_str:
            return
        try:
            if organized:
                export_to_excel_organized(self.procedure, Path(path_str))
                kind = "por categorías"
            else:
                export_to_excel(self.procedure, Path(path_str))
                kind = "por pasos"
            QMessageBox.information(
                self, "Exportar Excel",
                f"Archivo guardado ({kind}):\n{path_str}"
            )
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def _on_choose_material(self) -> None:
        """Abre el diálogo de biblioteca de materiales y aplica la selección."""
        from .material_dialog import MaterialDialog
        try:
            cur_E = float(self.E_input.text())
            cur_nu = float(self.nu_input.text())
        except ValueError:
            cur_E, cur_nu = None, None
        d = MaterialDialog(self, current_E=cur_E, current_nu=cur_nu)
        if d.exec() == QDialog.Accepted and d.selected_material is not None:
            mat = d.selected_material
            from .main_window import _fmt_input
            self.E_input.setText(_fmt_input(mat.E))
            self.nu_input.setText(_fmt_input(mat.nu))
            self.statusBar().showMessage(
                f"Material aplicado: {mat.name} — E={mat.E:g} Pa, ν={mat.nu}",
                4000,
            )

    def export_manual_pdf(self):
        """Exporta el manual teórico (PDF estático con la teoría del Q4)."""
        from ..export.manual_pdf import export_manual
        path_str, _ = QFileDialog.getSaveFileName(
            self, "Guardar Manual teórico", "manual_teorico_q4.pdf", "PDF (*.pdf)"
        )
        if not path_str:
            return
        try:
            export_manual(Path(path_str))
            QMessageBox.information(
                self, "Manual generado",
                f"Manual teórico guardado en:\n{path_str}\n\n"
                "Contiene: funciones de forma, cuadratura de Gauss, Jacobiano, "
                "matriz B, matrices D (tensión y deformación plana), K^e, "
                "ensamblaje, post-proceso y referencias bibliográficas."
            )
        except Exception as e:
            QMessageBox.critical(self, "Error al generar manual", str(e))

    def export_pdf(self):
        if not self.procedure or self.structure is None:
            return
        path_str, _ = QFileDialog.getSaveFileName(
            self, "Guardar PDF", "resultados_mef.pdf", "PDF (*.pdf)"
        )
        if path_str:
            try:
                export_to_pdf(self.procedure, self.structure, Path(path_str))
                QMessageBox.information(self, "Exportar PDF",
                                        f"Archivo guardado:\n{path_str}")
            except Exception as e:
                QMessageBox.critical(self, "Error", str(e))
