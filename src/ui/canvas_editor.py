"""Editor gráfico estilo SAP2000 para dibujar nodos y elementos Q4.

Modos:
    - Seleccionar:     click selecciona el nodo/elemento más cercano
    - Agregar nodo:    click crea un nodo nuevo (con snap a grid)
    - Agregar Q4:      4 clicks consecutivos sobre nodos existentes → elemento
    - Apoyo:           click en nodo → diálogo con Fix X / Fix Y
    - Carga:           click en nodo → diálogo con Fx / Fy
    - Borrar:          click selecciona y borra; o tecla DEL sobre selección

Funciones globales:
    - Snap a grid configurable
    - Generador de malla rectangular N×M
    - Zoom con rueda del mouse
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.patches import Polygon
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QButtonGroup, QLabel,
    QSpinBox, QDoubleSpinBox, QComboBox, QGroupBox, QDialog, QFormLayout,
    QCheckBox, QLineEdit, QDialogButtonBox, QMessageBox, QFrame, QGridLayout,
)

from ..fem.node import Node
from ..fem.q4_element import Q4Element
from ..fem.structure import Structure
from .icons import (
    icon, EDITOR_MODE_ICONS, ICON_MESH, ICON_CLEAR,
)
from .theme import Colors
import qtawesome as qta
from PySide6.QtCore import QSize


class Mode(str, Enum):
    SELECT = "Seleccionar"
    ADD_NODE = "Agregar nodo"
    ADD_Q4 = "Agregar Q4"
    ADD_RECT = "Rectángulo"
    APPLY_BC = "Apoyo"
    APPLY_LOAD = "Carga"
    DELETE = "Borrar"


# Mapeo Mode -> (icono qtawesome, tooltip)
MODE_ICONS: dict[Mode, tuple[str, str]] = {
    Mode.SELECT:     (EDITOR_MODE_ICONS["select"],     "Seleccionar (S) — click en un nodo o elemento"),
    Mode.ADD_NODE:   (EDITOR_MODE_ICONS["add_node"],   "Agregar nodo (N) — click en el canvas (snap a grid)"),
    Mode.ADD_Q4:     (EDITOR_MODE_ICONS["add_q4"],     "Agregar Q4 (Q) — 4 clicks en orden CCW desde sup-der"),
    Mode.ADD_RECT:   ("fa5s.square",                   "Rectángulo (R) — click+drag para crear nodos+elemento de un solo gesto"),
    Mode.APPLY_BC:   (EDITOR_MODE_ICONS["apply_bc"],   "Apoyo (A) — click en nodo para asignar restricciones"),
    Mode.APPLY_LOAD: (EDITOR_MODE_ICONS["apply_load"], "Carga (L) — click en nodo para asignar fuerzas"),
    Mode.DELETE:     (EDITOR_MODE_ICONS["delete"],     "Borrar (D o DEL) — click en nodo/elemento"),
}


# =================== Diálogos auxiliares ===================
class RestraintDialog(QDialog):
    def __init__(self, node: Node, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Apoyos — Nodo {node.id + 1}")
        layout = QFormLayout(self)
        self.cb_x = QCheckBox("Restringido en X")
        self.cb_y = QCheckBox("Restringido en Y")
        self.cb_x.setChecked(node.restraint_x)
        self.cb_y.setChecked(node.restraint_y)
        layout.addRow(self.cb_x)
        layout.addRow(self.cb_y)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        layout.addRow(bb)


class LoadDialog(QDialog):
    def __init__(self, node: Node, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Cargas — Nodo {node.id + 1}")
        layout = QFormLayout(self)
        self.fx = QLineEdit(repr(float(node.load_x)))
        self.fy = QLineEdit(repr(float(node.load_y)))
        layout.addRow("Fx (N):", self.fx)
        layout.addRow("Fy (N):", self.fy)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        layout.addRow(bb)


class MeshDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Generar malla rectangular")
        layout = QFormLayout(self)
        self.xmin = QLineEdit("0.0")
        self.xmax = QLineEdit("1.0")
        self.ymin = QLineEdit("0.0")
        self.ymax = QLineEdit("1.0")
        self.nx = QSpinBox(); self.nx.setRange(1, 100); self.nx.setValue(2)
        self.ny = QSpinBox(); self.ny.setRange(1, 100); self.ny.setValue(2)
        layout.addRow("X mín (m):", self.xmin)
        layout.addRow("X máx (m):", self.xmax)
        layout.addRow("Y mín (m):", self.ymin)
        layout.addRow("Y máx (m):", self.ymax)
        layout.addRow("Divisiones en X:", self.nx)
        layout.addRow("Divisiones en Y:", self.ny)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        layout.addRow(bb)


# =================== Editor principal ===================
class CanvasEditor(QWidget):
    """Editor gráfico de la estructura.

    Mantiene una referencia a un `Structure` y emite `structure_changed` cuando
    se modifica para que MainWindow recompute los pasos.
    """

    structure_changed = Signal()

    def __init__(self, structure: Structure | None = None, parent=None):
        super().__init__(parent)
        self.structure = structure or Structure()
        # propiedades por defecto para elementos nuevos
        self.default_E = 2173706.0
        self.default_nu = 0.15
        self.default_t = 0.1
        self.default_plane_stress = True

        # estado de edición
        self.mode = Mode.SELECT
        self.grid_size = 0.25
        self.snap_enabled = True
        self.show_grid = True
        self.show_labels = True
        self._pending_q4_nodes: list[Node] = []
        self._selected_node: Node | None = None
        self._selected_nodes: list[Node] = []   # multi-selección (rubber-band)
        self._hovered_node: Node | None = None
        # Undo/Redo: stacks de snapshots (estado serializado de la Structure)
        self._undo_stack: list[dict] = []
        self._redo_stack: list[dict] = []
        # Drag state
        self._dragging_node: Node | None = None
        self._drag_started: bool = False

        # Tracking de artistas matplotlib para render incremental (estilo SAP)
        # key: id del nodo/elemento → list de artistas (markers, anotaciones, etc.)
        self._artists_per_node: dict[int, list] = {}
        self._artists_per_element: dict[int, list] = {}
        self._overlay_artists: list = []  # hover ring, pending Q4 preview
        self._grid_artists: list = []
        # Estado del modo "Rectángulo": primera esquina + preview
        self._rect_first_corner: tuple[float, float] | None = None
        self._rect_preview_item = None

        self._build_ui()
        self._connect_mpl_events()
        self.redraw()

    # ---------- UI construction (layout dock-based estilo SAP / Abaqus) ----------
    def _build_ui(self) -> None:
        # Imports locales para mantener cohesión visual de bloques
        from PySide6.QtWidgets import QSplitter
        from .workflow_stepper import (WorkflowStepper, STEP_GEOMETRY, STEP_MATERIAL,
                                       STEP_MESH, STEP_SUPPORTS, STEP_LOADS,
                                       STEP_SOLVE, STEP_RESULTS)
        from .model_tree import ModelTreeWidget
        from .property_inspector import PropertyInspector

        main = QVBoxLayout(self)
        main.setContentsMargins(4, 4, 4, 4)
        main.setSpacing(4)

        # ===== Workflow Stepper (barra superior con 7 etapas) =====
        self.workflow = WorkflowStepper()
        self.workflow.step_clicked.connect(self._on_workflow_step_clicked)
        main.addWidget(self.workflow)

        # ===== Toolbar de modos (iconos cuadrados) =====
        toolbar = QHBoxLayout()
        toolbar.setSpacing(4)
        self.mode_buttons: dict[Mode, QPushButton] = {}
        self.btn_group = QButtonGroup(self)
        self.btn_group.setExclusive(True)
        for mode in Mode:
            ic_name, tip = MODE_ICONS[mode]
            btn = QPushButton()
            btn.setIcon(icon(ic_name))
            btn.setIconSize(QSize(22, 22))
            btn.setProperty("toolbar", True)
            btn.setToolTip(f"<b>{mode.value}</b><br>{tip}")
            btn.setCheckable(True)
            btn.clicked.connect(lambda _, m=mode: self.set_mode(m))
            self.mode_buttons[mode] = btn
            self.btn_group.addButton(btn)
            toolbar.addWidget(btn)
        self.mode_buttons[Mode.SELECT].setChecked(True)
        self.mode_buttons[Mode.SELECT].setIcon(
            qta.icon(MODE_ICONS[Mode.SELECT][0], color="white"))
        sep = QFrame(); sep.setFrameShape(QFrame.VLine); sep.setFrameShadow(QFrame.Sunken)
        sep.setStyleSheet("color: #D5DDE5;")
        toolbar.addWidget(sep)
        self.mode_label = QLabel("Modo: Seleccionar")
        self.mode_label.setStyleSheet("color: #1F4E78; font-weight: bold; padding-left: 8px;")
        toolbar.addWidget(self.mode_label)
        toolbar.addStretch()
        main.addLayout(toolbar)

        # ===== Fila de opciones (grid, snap, mallado, limpiar) =====
        opts = QHBoxLayout()
        opts.addWidget(QLabel("Grid:"))
        self.grid_combo = QComboBox()
        for s in ("0.05", "0.1", "0.25", "0.5", "1.0", "Sin grid"):
            self.grid_combo.addItem(s)
        self.grid_combo.setCurrentText("0.25")
        self.grid_combo.currentTextChanged.connect(self._on_grid_changed)
        opts.addWidget(self.grid_combo)
        self.snap_cb = QCheckBox("Snap")
        self.snap_cb.setChecked(True)
        self.snap_cb.toggled.connect(self._on_snap_changed)
        opts.addWidget(self.snap_cb)
        self.labels_cb = QCheckBox("Etiquetas")
        self.labels_cb.setChecked(True)
        self.labels_cb.toggled.connect(self._on_labels_toggled)
        opts.addWidget(self.labels_cb)
        opts.addSpacing(20)
        btn_mesh = QPushButton(" Generar malla N×M…")
        btn_mesh.setIcon(icon(ICON_MESH))
        btn_mesh.setIconSize(QSize(16, 16))
        btn_mesh.clicked.connect(self._on_generate_mesh)
        opts.addWidget(btn_mesh)
        btn_clear = QPushButton(" Limpiar todo")
        btn_clear.setIcon(icon(ICON_CLEAR))
        btn_clear.setIconSize(QSize(16, 16))
        btn_clear.clicked.connect(self._on_clear_all)
        opts.addWidget(btn_clear)

        btn_fit = QPushButton(" Ajustar vista")
        btn_fit.setIcon(icon("fa5s.expand"))
        btn_fit.setIconSize(QSize(16, 16))
        btn_fit.setToolTip(
            "<b>Ajustar vista (Home)</b><br>"
            "Vuelve la cámara al encuadre que abarca toda la estructura.<br>"
            "<br>"
            "Atajos de navegación del canvas:<br>"
            "&middot; Rueda del mouse: zoom centrado en el cursor<br>"
            "&middot; Click izquierdo + arrastrar: pan<br>"
            "&middot; Tecla Home: ajustar vista"
        )
        btn_fit.clicked.connect(self.reset_view)
        opts.addWidget(btn_fit)

        # Input de coordenadas exactas: Crear nodo en X=__ Y=__ [+]
        opts.addSpacing(20)
        from PySide6.QtGui import QDoubleValidator as _QDV
        opts.addWidget(QLabel("Nodo exacto:"))
        self.exact_x_input = QLineEdit()
        self.exact_x_input.setPlaceholderText("X")
        self.exact_x_input.setMaximumWidth(70)
        self.exact_x_input.setValidator(_QDV())
        self.exact_x_input.setToolTip(
            "<b>Crear nodo en coordenadas exactas</b><br>"
            "Ingresa el valor de X (en metros) — sin depender del snap.<br>"
            "Acepta decimales, notación científica (1e-3) y negativos."
        )
        opts.addWidget(self.exact_x_input)
        self.exact_y_input = QLineEdit()
        self.exact_y_input.setPlaceholderText("Y")
        self.exact_y_input.setMaximumWidth(70)
        self.exact_y_input.setValidator(_QDV())
        opts.addWidget(self.exact_y_input)
        btn_add_exact = QPushButton()
        btn_add_exact.setIcon(icon("fa5s.plus"))
        btn_add_exact.setIconSize(QSize(14, 14))
        btn_add_exact.setMaximumWidth(36)
        btn_add_exact.setToolTip(
            "Crear nodo en las coordenadas X, Y exactas indicadas a la izquierda. "
            "Atajo: Enter en cualquiera de los campos."
        )
        btn_add_exact.clicked.connect(self._on_add_exact_node)
        opts.addWidget(btn_add_exact)
        # Enter en cualquier campo dispara el botón
        self.exact_x_input.returnPressed.connect(self._on_add_exact_node)
        self.exact_y_input.returnPressed.connect(self._on_add_exact_node)
        opts.addStretch()
        main.addLayout(opts)

        # ===== Separador =====
        line = QFrame(); line.setFrameShape(QFrame.HLine); line.setFrameShadow(QFrame.Sunken)
        main.addWidget(line)

        # ===== Layout principal: 3 paneles (ModelTree | Canvas | Inspector) =====
        body_splitter = QSplitter(Qt.Horizontal)
        body_splitter.setHandleWidth(6)
        body_splitter.setChildrenCollapsible(False)

        # --- Panel izquierdo: Model Tree ---
        self.model_tree = ModelTreeWidget()
        self.model_tree.set_structure(self.structure)
        self.model_tree.item_selected.connect(self._on_tree_item_selected)
        self.model_tree.item_double_clicked.connect(self._on_tree_item_double_clicked)
        self.model_tree.item_delete_requested.connect(self._on_tree_item_delete)
        body_splitter.addWidget(self.model_tree)

        # --- Panel central: canvas PyQtGraph (60+ fps, pan/zoom suaves built-in) ---
        from .pg_canvas import PgCanvas
        canvas_container = QWidget()
        cc_layout = QVBoxLayout(canvas_container)
        cc_layout.setContentsMargins(0, 0, 0, 0)
        self.pg_canvas = PgCanvas()
        self.pg_canvas.set_grid_size(self.grid_size)
        self.pg_canvas.set_grid_visible(self.show_grid)
        # Conectar eventos del mouse / teclado
        self.pg_canvas.signals.left_click.connect(self._on_pg_left_click)
        self.pg_canvas.signals.left_double_click.connect(self._on_pg_left_double_click)
        self.pg_canvas.signals.right_click.connect(self._on_pg_right_click)
        self.pg_canvas.signals.mouse_move.connect(self._on_pg_mouse_move)
        self.pg_canvas.signals.mouse_leave.connect(self._on_pg_mouse_leave)
        self.pg_canvas.signals.key_delete.connect(self._on_pg_key_delete)
        self.pg_canvas.signals.key_home.connect(self.reset_view)
        self.pg_canvas.signals.key_escape.connect(self._on_pg_key_escape)
        self.pg_canvas.signals.rubber_band_finished.connect(self._on_pg_rubber_band)
        cc_layout.addWidget(self.pg_canvas)
        body_splitter.addWidget(canvas_container)
        # Aliases para mantener compatibilidad con código viejo que usa self.canvas
        self.canvas = self.pg_canvas
        # Estado del pan (pyqtgraph lo maneja built-in; estos atributos quedan
        # por compatibilidad con código viejo que los chequea)
        self._pan_active = False
        self._pan_start_data = None

        # --- Panel derecho: Property Inspector ---
        self.inspector = PropertyInspector()
        self.inspector.value_changed.connect(self._on_inspector_value_changed)
        self.inspector.show_empty()
        body_splitter.addWidget(self.inspector)

        # Proporciones iniciales: tree 18% · canvas 60% · inspector 22%
        body_splitter.setStretchFactor(0, 2)
        body_splitter.setStretchFactor(1, 6)
        body_splitter.setStretchFactor(2, 2)
        body_splitter.setSizes([220, 700, 260])

        main.addWidget(body_splitter, 1)

        # ===== Status bar enriquecido =====
        status_bar = QFrame()
        status_bar.setStyleSheet(
            "background:#F7F9FB; border-top:1px solid #D5DDE5;"
        )
        sb_layout = QHBoxLayout(status_bar)
        sb_layout.setContentsMargins(10, 4, 10, 4)
        sb_layout.setSpacing(16)

        def _seg(label_html: str, style: str = "") -> QLabel:
            lbl = QLabel(label_html)
            lbl.setStyleSheet(f"color:#1F4E78; {style}")
            return lbl

        self.status_cursor = _seg("Cursor: —", "font-family: Consolas, monospace;")
        self.status_count = _seg("Nodos: 0 · Elementos: 0", "font-weight: 600;")
        self.status_mode = _seg("Modo: Seleccionar")
        self.status_hint = _seg("Click en el canvas para empezar",
                                 "color:#6C757D; font-style: italic;")

        sb_layout.addWidget(self.status_cursor)
        sb_sep1 = QLabel("|"); sb_sep1.setStyleSheet("color: #D5DDE5;")
        sb_layout.addWidget(sb_sep1)
        sb_layout.addWidget(self.status_count)
        sb_sep2 = QLabel("|"); sb_sep2.setStyleSheet("color: #D5DDE5;")
        sb_layout.addWidget(sb_sep2)
        sb_layout.addWidget(self.status_mode)
        sb_sep3 = QLabel("|"); sb_sep3.setStyleSheet("color: #D5DDE5;")
        sb_layout.addWidget(sb_sep3)
        sb_layout.addWidget(self.status_hint, 1)

        main.addWidget(status_bar)

        # Compatibilidad con código viejo que escribía a self.status.setText()
        class _StatusFacade:
            def __init__(self, hint_label):
                self.hint = hint_label
            def setText(self, txt):
                # Si el texto sigue el patrón "Modo: X  |  hint", extraer solo el hint
                if " | " in txt and txt.startswith("Modo:"):
                    parts = txt.split(" | ", 1)
                    if len(parts) == 2:
                        self.hint.setText(parts[1].lstrip())
                        return
                self.hint.setText(txt)
        self.status = _StatusFacade(self.status_hint)

    def _connect_mpl_events(self) -> None:
        """Atajos de teclado Ctrl+Z / Ctrl+Y a nivel del widget.

        Los eventos del mouse ahora son manejados por PgCanvas (señales
        conectadas en _build_ui), reemplazando los mpl_connect anteriores.
        """
        from PySide6.QtGui import QShortcut, QKeySequence
        QShortcut(QKeySequence("Ctrl+Z"), self, activated=self.undo)
        QShortcut(QKeySequence("Ctrl+Y"), self, activated=self.redo)
        QShortcut(QKeySequence("Ctrl+Shift+Z"), self, activated=self.redo)
        # Selección múltiple
        QShortcut(QKeySequence("Ctrl+A"), self, activated=self._select_all_nodes)
        # Atajos de teclado para cambiar modo rápido
        QShortcut(QKeySequence("S"), self, activated=lambda: self.set_mode(Mode.SELECT))
        QShortcut(QKeySequence("N"), self, activated=lambda: self.set_mode(Mode.ADD_NODE))
        QShortcut(QKeySequence("Q"), self, activated=lambda: self.set_mode(Mode.ADD_Q4))
        QShortcut(QKeySequence("R"), self, activated=lambda: self.set_mode(Mode.ADD_RECT))
        QShortcut(QKeySequence("A"), self, activated=lambda: self.set_mode(Mode.APPLY_BC))
        QShortcut(QKeySequence("L"), self, activated=lambda: self.set_mode(Mode.APPLY_LOAD))
        QShortcut(QKeySequence("D"), self, activated=lambda: self.set_mode(Mode.DELETE))

    # ---------- Undo / Redo ----------
    def _snapshot(self) -> dict:
        """Devuelve una serialización del estado actual de la estructura."""
        return {
            "nodes": [
                dict(id=n.id, x=n.x, y=n.y,
                     rx=n.restraint_x, ry=n.restraint_y,
                     fx=n.load_x, fy=n.load_y)
                for n in self.structure.nodes
            ],
            "elements": [
                dict(id=el.id, nodes=[n.id for n in el.nodes],
                     E=el.E, nu=el.nu, t=el.t, ps=el.plane_stress)
                for el in self.structure.elements
            ],
        }

    def _restore(self, snap: dict) -> None:
        """Restaura la estructura desde un snapshot."""
        self.structure.nodes.clear()
        self.structure.elements.clear()
        node_by_id: dict[int, Node] = {}
        for nd in snap["nodes"]:
            n = Node(id=nd["id"], x=nd["x"], y=nd["y"],
                     restraint_x=nd["rx"], restraint_y=nd["ry"],
                     load_x=nd["fx"], load_y=nd["fy"])
            self.structure.add_node(n)
            node_by_id[n.id] = n
        for el in snap["elements"]:
            self.structure.add_element(Q4Element(
                id=el["id"],
                nodes=[node_by_id[i] for i in el["nodes"]],
                E=el["E"], nu=el["nu"], t=el["t"],
                plane_stress=el["ps"],
            ))

    def _push_undo(self) -> None:
        """Llamar ANTES de cada modificación. Pushea snapshot al undo_stack."""
        self._undo_stack.append(self._snapshot())
        # Limitar tamaño del historial
        if len(self._undo_stack) > 50:
            self._undo_stack.pop(0)
        self._redo_stack.clear()

    def undo(self) -> None:
        if not self._undo_stack:
            self.status_hint.setText("Nada que deshacer")
            return
        self._redo_stack.append(self._snapshot())
        self._restore(self._undo_stack.pop())
        self._pending_q4_nodes = []
        self.structure_changed.emit()
        self.redraw()
        self.status_hint.setText("↶ Deshecho")

    def redo(self) -> None:
        if not self._redo_stack:
            self.status_hint.setText("Nada que rehacer")
            return
        self._undo_stack.append(self._snapshot())
        self._restore(self._redo_stack.pop())
        self._pending_q4_nodes = []
        self.structure_changed.emit()
        self.redraw()
        self.status_hint.setText("↷ Rehecho")

    # ---------- Public API ----------
    def set_structure(self, structure: Structure) -> None:
        self.structure = structure
        self._pending_q4_nodes = []
        self._selected_node = None
        if hasattr(self, "model_tree") and self.model_tree is not None:
            self.model_tree.set_structure(structure)
        if hasattr(self, "inspector") and self.inspector is not None:
            self.inspector.show_empty()
        self.redraw()

    def set_default_properties(self, E: float, nu: float, t: float,
                               plane_stress: bool) -> None:
        self.default_E = E
        self.default_nu = nu
        self.default_t = t
        self.default_plane_stress = plane_stress

    # ---------- Handlers del WorkflowStepper / ModelTree / Inspector ----------
    def _on_workflow_step_clicked(self, step_idx: int) -> None:
        """El usuario clickeó una etapa del workflow: cambiar modo correspondiente."""
        from .workflow_stepper import (STEP_GEOMETRY, STEP_MATERIAL, STEP_MESH,
                                       STEP_SUPPORTS, STEP_LOADS, STEP_SOLVE,
                                       STEP_RESULTS)
        # Mapeo etapa → modo del editor
        mode_map = {
            STEP_GEOMETRY: Mode.ADD_NODE,
            STEP_SUPPORTS: Mode.APPLY_BC,
            STEP_LOADS:    Mode.APPLY_LOAD,
        }
        self.workflow.set_current_step(step_idx)
        if step_idx in mode_map:
            self.set_mode(mode_map[step_idx])
        elif step_idx == STEP_MESH:
            self._on_generate_mesh()
        elif step_idx == STEP_SOLVE:
            # Forzar recálculo desde MainWindow
            self.structure_changed.emit()
            self.status_hint.setText("Solver invocado — ver pestaña Paso a paso")
        elif step_idx == STEP_RESULTS:
            self.status_hint.setText(
                "Ve a la pestaña 'Paso a paso' para ver desplazamientos, "
                "esfuerzos y reacciones (pasos 12, 14, 15)."
            )
        elif step_idx == STEP_MATERIAL:
            self.set_mode(Mode.SELECT)
            self.status_hint.setText(
                "Selecciona un elemento del Model Tree para editar E, ν, t"
            )

    def _on_tree_item_selected(self, category: str, obj_id: int) -> None:
        """Sync entre ModelTree y canvas/inspector."""
        if category == "node":
            node = next((n for n in self.structure.nodes if n.id == obj_id), None)
            if node is not None:
                self._selected_node = node
                self.inspector.show_node(node)
                self._render_overlay()
        elif category == "element":
            el = next((e for e in self.structure.elements if e.id == obj_id), None)
            if el is not None:
                self.inspector.show_element(el)
        elif category in ("support", "load"):
            # Para apoyos/cargas, navegar al nodo correspondiente
            node = next((n for n in self.structure.nodes if n.id == obj_id), None)
            if node is not None:
                self._selected_node = node
                self.inspector.show_node(node)
                self._render_overlay()
        else:
            self.inspector.show_empty()

    def _on_tree_item_double_clicked(self, category: str, obj_id: int) -> None:
        """Doble click abre el diálogo de edición apropiado."""
        if category in ("node", "support"):
            node = next((n for n in self.structure.nodes if n.id == obj_id), None)
            if node is not None:
                self._open_bc_dialog(node)
        elif category == "load":
            node = next((n for n in self.structure.nodes if n.id == obj_id), None)
            if node is not None:
                self._open_load_dialog(node)

    def _on_tree_item_delete(self, category: str, obj_id: int) -> None:
        """Tecla DEL en el árbol borra el objeto."""
        if category == "node":
            node = next((n for n in self.structure.nodes if n.id == obj_id), None)
            if node is not None:
                self._push_undo()
                self._delete_node(node)
        elif category == "element":
            el = next((e for e in self.structure.elements if e.id == obj_id), None)
            if el is not None:
                self._push_undo()
                self.structure.elements.remove(el)
                if el.id in self._artists_per_element:
                    self._safe_remove(self._artists_per_element[el.id])
                    del self._artists_per_element[el.id]
                self._update_counter()
                self.canvas.draw_idle()
                self.model_tree.refresh()
                self.structure_changed.emit()

    def _on_inspector_value_changed(self) -> None:
        """El usuario editó un valor en el Property Inspector."""
        # Refrescar el canvas del nodo o elemento activo
        if self._selected_node is not None:
            self._refresh_node(self._selected_node)
        # Si tocó un elemento, las propiedades E/ν/t cambiaron — emit para recalc
        self.structure_changed.emit()
        # Refrescar el árbol (los labels pueden haber cambiado)
        self.model_tree.refresh()

    def set_mode(self, mode: Mode) -> None:
        self.mode = mode
        self._pending_q4_nodes = []
        self._hovered_node = None
        self.status_mode.setText(f"Modo: {mode.value}")
        self.status_hint.setText(self._mode_hint())
        self.mode_label.setText(f"Modo: {mode.value}")
        # Cursor según el modo (feedback visual inmediato sobre qué hace click)
        if hasattr(self, "pg_canvas") and self.pg_canvas is not None:
            cursor_map = {
                Mode.SELECT:     Qt.ArrowCursor,
                Mode.ADD_NODE:   Qt.CrossCursor,
                Mode.ADD_Q4:     Qt.CrossCursor,
                Mode.APPLY_BC:   Qt.PointingHandCursor,
                Mode.APPLY_LOAD: Qt.PointingHandCursor,
                Mode.DELETE:     Qt.ForbiddenCursor,
            }
            self.pg_canvas.setCursor(cursor_map.get(mode, Qt.ArrowCursor))
        # Pintar el icono del modo seleccionado en blanco, los demás en navy
        for m, btn in self.mode_buttons.items():
            ic_name, _ = MODE_ICONS[m]
            if m == mode:
                btn.setIcon(qta.icon(ic_name, color="white"))
                btn.setChecked(True)
            else:
                btn.setIcon(icon(ic_name))
        self.redraw()

    def _mode_hint(self) -> str:
        return {
            Mode.SELECT: "Click en un nodo/elemento para seleccionar. Ctrl+arrastrar = multi-selección.",
            Mode.ADD_NODE: f"Click para crear nodo (snap={self.grid_size if self.snap_enabled else 'OFF'})",
            Mode.ADD_Q4: "Click 4 nodos en orden CCW desde esquina sup-der (++,-+,--,+-). Esc cancela.",
            Mode.ADD_RECT: "Click esquina 1, después click esquina 2: crea nodos + elemento. Esc cancela.",
            Mode.APPLY_BC: "Click en un nodo para asignar apoyos. Si hay multi-selección, aplica al grupo.",
            Mode.APPLY_LOAD: "Click en un nodo para asignar cargas. Si hay multi-selección, aplica al grupo.",
            Mode.DELETE: "Click en un nodo/elemento para borrar (o tecla DEL).",
        }[self.mode]

    # ---------- Eventos Matplotlib ----------
    def _on_click(self, event) -> None:
        # Botón derecho (button == 3): iniciar pan
        if event.inaxes == self.ax and event.button == 3:
            self._pan_active = True
            self._pan_start_data = (event.xdata, event.ydata)
            self.canvas.setCursor(Qt.ClosedHandCursor)
            return
        if event.inaxes != self.ax or event.button != 1:
            return
        if event.xdata is None or event.ydata is None:
            return
        x, y = self._snap(event.xdata, event.ydata)

        if self.mode == Mode.ADD_NODE:
            self._push_undo()
            self._add_node(x, y)
        elif self.mode == Mode.ADD_Q4:
            n = self._pick_node(event.xdata, event.ydata)
            if n is not None:
                self._handle_q4_click(n)
        elif self.mode == Mode.APPLY_BC:
            n = self._pick_node(event.xdata, event.ydata)
            if n is not None:
                self._push_undo()
                self._open_bc_dialog(n)
        elif self.mode == Mode.APPLY_LOAD:
            n = self._pick_node(event.xdata, event.ydata)
            if n is not None:
                self._push_undo()
                self._open_load_dialog(n)
        elif self.mode == Mode.DELETE:
            self._push_undo()
            self._delete_at(event.xdata, event.ydata)
        elif self.mode == Mode.SELECT:
            n = self._pick_node(event.xdata, event.ydata)
            self._selected_node = n
            # Iniciar drag si hicimos click sobre un nodo
            if n is not None:
                self._dragging_node = n
                self._drag_started = False
            self.redraw()

    def _on_release(self, event) -> None:
        """Finaliza el drag del nodo o el pan si estaban en curso."""
        # Pan finalizado
        if self._pan_active:
            self._pan_active = False
            self._pan_start_data = None
            self.canvas.unsetCursor()
            return
        if self._dragging_node is not None:
            if self._drag_started:
                # El nodo realmente se movió: emitir cambio
                self.structure_changed.emit()
                self.status_hint.setText(
                    f"Nodo {self._dragging_node.id + 1} movido a "
                    f"({self._dragging_node.x:.4g}, {self._dragging_node.y:.4g})"
                )
            self._dragging_node = None
            self._drag_started = False

    def _on_mouse_move(self, event) -> None:
        # Pan en curso: trasladar los límites del axes
        if self._pan_active and self._pan_start_data is not None:
            if event.inaxes == self.ax and event.xdata is not None:
                dx = self._pan_start_data[0] - event.xdata
                dy = self._pan_start_data[1] - event.ydata
                xlim = self.ax.get_xlim()
                ylim = self.ax.get_ylim()
                self.ax.set_xlim(xlim[0] + dx, xlim[1] + dx)
                self.ax.set_ylim(ylim[0] + dy, ylim[1] + dy)
                self._safe_remove(self._grid_artists)
                self._grid_artists.clear()
                self._render_grid()
                self.canvas.draw_idle()
            return
        if event.inaxes != self.ax or event.xdata is None:
            self.status_cursor.setText("Cursor: —")
            return
        x, y = self._snap(event.xdata, event.ydata)
        self.status_cursor.setText(f"Cursor: ({x:.4g}, {y:.4g})")

        # Drag: si estamos arrastrando un nodo en modo Select, mover y redibujar
        if self._dragging_node is not None and self.mode == Mode.SELECT:
            if not self._drag_started:
                if abs(self._dragging_node.x - x) > 1e-12 or abs(self._dragging_node.y - y) > 1e-12:
                    self._push_undo()
                    self._drag_started = True
            self._dragging_node.x = float(x)
            self._dragging_node.y = float(y)
            # Refresh incremental: solo el nodo movido (y sus elementos),
            # no se redibuja el resto del canvas.
            self._refresh_node(self._dragging_node)
            self._render_overlay()
            return

        # Hover highlight: si el modo usa nodos, marcar el nodo más cercano
        if self.mode in (Mode.ADD_Q4, Mode.APPLY_BC, Mode.APPLY_LOAD,
                         Mode.DELETE, Mode.SELECT):
            new_hover = self._pick_node(event.xdata, event.ydata)
        else:
            new_hover = None
        if new_hover is not self._hovered_node:
            self._hovered_node = new_hover
            # Solo se refresca la capa de overlay (anillo del hover),
            # no se redibujan nodos ni elementos.
            self._render_overlay()

    def _on_scroll(self, event) -> None:
        """Zoom con rueda del mouse, centrado en el cursor."""
        if event.inaxes != self.ax:
            return
        scale = 1.2 if event.button == "down" else 1 / 1.2
        x, y = event.xdata, event.ydata
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        self.ax.set_xlim([x + (xl - x) * scale for xl in xlim])
        self.ax.set_ylim([y + (yl - y) * scale for yl in ylim])
        # Refrescar el grid con los nuevos límites (sin re-renderizar nodos/elementos)
        self._safe_remove(self._grid_artists)
        self._grid_artists.clear()
        self._render_grid()
        self.canvas.draw_idle()

    def _on_key(self, event) -> None:
        if event.key in ("delete", "backspace") and self._selected_node is not None:
            self._delete_node(self._selected_node)
            self._selected_node = None
        elif event.key == "home":
            self.reset_view()

    # ---------- Snap a grid ----------
    def _snap(self, x: float, y: float) -> tuple[float, float]:
        if not self.snap_enabled or self.grid_size <= 0:
            return x, y
        g = self.grid_size
        return round(x / g) * g, round(y / g) * g

    def _on_grid_changed(self, txt: str) -> None:
        if txt == "Sin grid":
            self.show_grid = False
        else:
            self.show_grid = True
            try:
                self.grid_size = float(txt)
            except ValueError:
                self.grid_size = 0.25
        # Sincronizar con pg_canvas y refrescar grid sin tocar nodos/elementos
        if hasattr(self, "pg_canvas") and self.pg_canvas is not None:
            self.pg_canvas.set_grid_size(self.grid_size)
            self.pg_canvas.set_grid_visible(self.show_grid)

    def _on_snap_changed(self, v: bool) -> None:
        self.snap_enabled = v
        self.status_hint.setText(
            f"Snap a grid {self.grid_size}: {'ACTIVADO' if v else 'desactivado'}"
        )

    def _on_add_exact_node(self) -> None:
        """Crea un nodo en coordenadas exactas (X, Y) escritas por el usuario."""
        try:
            x = float(self.exact_x_input.text().strip())
            y = float(self.exact_y_input.text().strip())
        except ValueError:
            self.status_hint.setText(
                "Ingresá valores numéricos válidos en X e Y antes de presionar +"
            )
            return
        self._push_undo()
        self._add_node(x, y)
        # Limpiar para permitir crear el siguiente rápido
        self.exact_x_input.clear()
        self.exact_y_input.clear()
        self.exact_x_input.setFocus()
        self.status_hint.setText(
            f"Nodo creado en ({x:g}, {y:g}) — precisión exacta, sin snap"
        )

    def _on_labels_toggled(self, v: bool) -> None:
        """Toggle Etiquetas: oculta/muestra los nombres N1, E1, valores de cargas."""
        self.show_labels = v
        if hasattr(self, "pg_canvas") and self.pg_canvas is not None:
            self.pg_canvas.show_labels = v
        # Re-render completo (las etiquetas son hijas de los nodos/elementos)
        self.redraw(fit_view=False)

    # ---------- Operaciones sobre la estructura ----------
    def _next_node_id(self) -> int:
        if not self.structure.nodes:
            return 0
        return max(n.id for n in self.structure.nodes) + 1

    def _next_element_id(self) -> int:
        if not self.structure.elements:
            return 0
        return max(e.id for e in self.structure.elements) + 1

    def _add_node(self, x: float, y: float) -> Node:
        # Evitar duplicados muy cerca
        for n in self.structure.nodes:
            if abs(n.x - x) < 1e-9 and abs(n.y - y) < 1e-9:
                self.status.setText(f"Ya existe un nodo en ({x:.4g}, {y:.4g})")
                return n
        n = Node(id=self._next_node_id(), x=float(x), y=float(y))
        self.structure.add_node(n)
        # Incremental: añadir solo este nodo al canvas, no redibujar todo.
        self._add_node_incremental(n)
        self.structure_changed.emit()
        return n

    def _pick_node(self, x: float, y: float, radius: float | None = None) -> Node | None:
        """Devuelve el nodo más cercano dentro de un radio razonable."""
        if not self.structure.nodes:
            return None
        if radius is None:
            xlim = self.ax.get_xlim()
            radius = (xlim[1] - xlim[0]) * 0.03  # 3% del ancho visible
        best, best_d = None, radius
        for n in self.structure.nodes:
            d = np.hypot(n.x - x, n.y - y)
            if d < best_d:
                best, best_d = n, d
        return best

    def _pick_element(self, x: float, y: float) -> Q4Element | None:
        """Devuelve el primer elemento que contenga el punto (x,y)."""
        for el in self.structure.elements:
            coords = el.coords
            if self._point_in_polygon(x, y, coords):
                return el
        return None

    @staticmethod
    def _point_in_polygon(x: float, y: float, poly: np.ndarray) -> bool:
        """Ray casting algorithm."""
        n = len(poly)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = poly[i]
            xj, yj = poly[j]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-30) + xi):
                inside = not inside
            j = i
        return inside

    def _handle_q4_click(self, n: Node) -> None:
        if n in self._pending_q4_nodes:
            return
        self._pending_q4_nodes.append(n)
        self.status.setText(
            f"Q4: nodo {n.id + 1} añadido  ({len(self._pending_q4_nodes)}/4)"
        )
        if len(self._pending_q4_nodes) == 4:
            self._create_q4()
        else:
            # Incremental: solo actualizar el overlay (preview Q4 parcial)
            self._render_overlay()

    # ---------- Modo "Rectángulo" (click + click para crear Q4 completo) ----------
    def _draw_rect_preview(self, corner0: tuple[float, float],
                           corner1: tuple[float, float]) -> None:
        """Dibuja un rectángulo punteado entre las 2 esquinas (preview)."""
        import pyqtgraph as pg
        from PySide6.QtCore import Qt
        self._clear_rect_preview()
        x0, y0 = corner0
        x1, y1 = corner1
        xs = [x0, x1, x1, x0, x0]
        ys = [y0, y0, y1, y1, y0]
        preview = pg.PlotDataItem(
            x=xs, y=ys,
            pen=pg.mkPen("#ff9800", width=2, style=Qt.DashLine),
        )
        preview.setZValue(20)
        self.pg_canvas.addItem(preview)
        self._rect_preview_item = preview

    def _clear_rect_preview(self) -> None:
        if self._rect_preview_item is not None:
            try:
                self.pg_canvas.removeItem(self._rect_preview_item)
            except Exception:
                pass
            self._rect_preview_item = None

    def _create_rectangle(self, x0: float, y0: float,
                           x1: float, y1: float) -> None:
        """Crea 4 nodos + 1 elemento Q4 a partir de 2 esquinas opuestas.

        Convención de nodos del Q4 (CCW desde sup-der): ++, -+, --, +-.
        Se calcula correctamente independiente del orden de click del usuario.
        """
        self._push_undo()
        # Normalizar: xmin, xmax, ymin, ymax
        xmin, xmax = min(x0, x1), max(x0, x1)
        ymin, ymax = min(y0, y1), max(y0, y1)
        # Crear 4 nodos en orden de la convención
        positions = [
            (xmax, ymax),  # N1  (++)
            (xmin, ymax),  # N2  (-+)
            (xmin, ymin),  # N3  (--)
            (xmax, ymin),  # N4  (+-)
        ]
        new_nodes = []
        for px, py in positions:
            # Reusar nodo si ya existe en esa posición
            existing = next(
                (n for n in self.structure.nodes
                 if abs(n.x - px) < 1e-9 and abs(n.y - py) < 1e-9),
                None,
            )
            if existing is not None:
                new_nodes.append(existing)
            else:
                n = Node(id=self._next_node_id(), x=float(px), y=float(py))
                self.structure.add_node(n)
                self._add_node_incremental(n)
                new_nodes.append(n)
        # Crear el elemento Q4
        try:
            el = Q4Element(
                id=self._next_element_id(),
                nodes=new_nodes,
                E=self.default_E, nu=self.default_nu,
                t=self.default_t, plane_stress=self.default_plane_stress,
            )
            self.structure.add_element(el)
            self._add_element_incremental(el)
            self.status_hint.setText(
                f"Rectángulo {xmax - xmin:g} × {ymax - ymin:g} m creado "
                f"(E{el.id + 1})"
            )
            self.structure_changed.emit()
        except Exception as e:
            QMessageBox.critical(self, "Error al crear elemento", str(e))

    def _create_q4(self) -> None:
        self._push_undo()
        try:
            el = Q4Element(
                id=self._next_element_id(),
                nodes=list(self._pending_q4_nodes),
                E=self.default_E, nu=self.default_nu,
                t=self.default_t, plane_stress=self.default_plane_stress,
            )
            # Verificar orientación (Jacobiano positivo en el centro)
            _, detJ = el.jacobian(0.0, 0.0)
            if detJ <= 0:
                QMessageBox.warning(
                    self, "Orden de nodos incorrecto",
                    "El Jacobiano del elemento es ≤ 0.  Los 4 nodos deben "
                    "ingresarse en orden CCW desde la esquina superior derecha "
                    "(convención ++, -+, --, +-)."
                )
                self._pending_q4_nodes = []
                self.redraw()
                return
            self.structure.add_element(el)
            self._add_element_incremental(el)
            self.status.setText(
                f"Elemento Q4 #{el.id + 1} creado con nodos "
                + ", ".join(str(n.id + 1) for n in el.nodes)
            )
        except Exception as e:
            QMessageBox.critical(self, "Error al crear elemento", str(e))
        finally:
            self._pending_q4_nodes = []
            self.structure_changed.emit()
            self._render_overlay()   # limpia el preview parcial

    def _delete_at(self, x: float, y: float) -> None:
        # Primero busca nodo cercano
        n = self._pick_node(x, y)
        if n is not None:
            self._delete_node(n)
            return
        el = self._pick_element(x, y)
        if el is not None:
            self.structure.elements.remove(el)
            self.structure_changed.emit()
            self.status.setText(f"Elemento {el.id + 1} borrado")
            self.redraw()

    def _delete_node(self, n: Node) -> None:
        # Borra el nodo y cualquier elemento que lo use
        used_by = [el for el in self.structure.elements if n in el.nodes]
        if used_by and not self._confirm_node_delete(n, used_by):
            return
        for el in used_by:
            self.structure.elements.remove(el)
        self.structure.nodes.remove(n)
        # Reindexar IDs para mantenerlos consecutivos
        for i, nd in enumerate(self.structure.nodes):
            nd.id = i
        for i, el in enumerate(self.structure.elements):
            el.id = i
        self.structure_changed.emit()
        self.status.setText(f"Nodo borrado (y {len(used_by)} elemento(s) asociado(s))")
        if hasattr(self, "model_tree") and self.model_tree is not None:
            self.model_tree.refresh()
        if hasattr(self, "inspector") and self.inspector is not None:
            self.inspector.show_empty()
        self.redraw()

    def _confirm_node_delete(self, n: Node, used_by: list[Q4Element]) -> bool:
        r = QMessageBox.question(
            self, "Confirmar borrado",
            f"El nodo {n.id + 1} es usado por {len(used_by)} elemento(s). "
            f"Borrarlo también borrará esos elementos. ¿Continuar?"
        )
        return r == QMessageBox.Yes

    def _open_bc_dialog(self, n: Node) -> None:
        d = RestraintDialog(n, self)
        if d.exec() == QDialog.Accepted:
            n.restraint_x = d.cb_x.isChecked()
            n.restraint_y = d.cb_y.isChecked()
            self.structure_changed.emit()
            self._refresh_node(n)   # incremental

    def _open_load_dialog(self, n: Node) -> None:
        d = LoadDialog(n, self)
        if d.exec() == QDialog.Accepted:
            try:
                n.load_x = float(d.fx.text() or 0)
                n.load_y = float(d.fy.text() or 0)
            except ValueError:
                QMessageBox.critical(self, "Error", "Carga inválida")
                return
            self.structure_changed.emit()
            self._refresh_node(n)   # incremental

    def _on_clear_all(self) -> None:
        r = QMessageBox.question(self, "Limpiar", "¿Borrar TODOS los nodos y elementos?")
        if r != QMessageBox.Yes:
            return
        self._push_undo()
        self.structure.nodes.clear()
        self.structure.elements.clear()
        self._pending_q4_nodes = []
        self._selected_node = None
        self.structure_changed.emit()
        self.redraw()

    # ---------- Mallado automático ----------
    def _on_generate_mesh(self) -> None:
        d = MeshDialog(self)
        if d.exec() != QDialog.Accepted:
            return
        try:
            xmin = float(d.xmin.text())
            xmax = float(d.xmax.text())
            ymin = float(d.ymin.text())
            ymax = float(d.ymax.text())
            nx = d.nx.value()
            ny = d.ny.value()
        except ValueError:
            QMessageBox.critical(self, "Error", "Coordenadas inválidas")
            return
        if xmax <= xmin or ymax <= ymin:
            QMessageBox.critical(self, "Error", "x_max debe > x_min y y_max > y_min")
            return
        self._generate_rect_mesh(xmin, xmax, ymin, ymax, nx, ny)

    def _generate_rect_mesh(self, xmin, xmax, ymin, ymax, nx, ny,
                            animate: bool = True) -> None:
        """Genera una malla de nx × ny elementos Q4 con nodos compartidos.

        Si animate=True, los nodos y elementos se añaden progresivamente
        al canvas con un QTimer (estilo SAP). Si False, se añaden todos
        instantáneamente (más rápido para mallas grandes).
        """
        from PySide6.QtCore import QTimer

        self._push_undo()
        self.structure.nodes.clear()
        self.structure.elements.clear()
        self._clear_all_artists()
        self.ax.clear()
        self.ax.set_aspect("equal")
        self.ax.set_xlabel("X (m)"); self.ax.set_ylabel("Y (m)")

        dx = (xmax - xmin) / nx
        dy = (ymax - ymin) / ny

        # Pre-calcular todos los nodos y elementos en memoria
        nodes_queue: list[Node] = []
        node_at: dict[tuple[int, int], Node] = {}
        nid = 0
        for j in range(ny + 1):
            for i in range(nx + 1):
                n = Node(id=nid, x=xmin + i * dx, y=ymin + j * dy)
                nodes_queue.append(n)
                node_at[(i, j)] = n
                nid += 1

        elements_queue: list[Q4Element] = []
        eid = 0
        for j in range(ny):
            for i in range(nx):
                n1 = node_at[(i + 1, j + 1)]   # (++)
                n2 = node_at[(i,     j + 1)]   # (-+)
                n3 = node_at[(i,     j)]       # (--)
                n4 = node_at[(i + 1, j)]       # (+-)
                el = Q4Element(
                    id=eid, nodes=[n1, n2, n3, n4],
                    E=self.default_E, nu=self.default_nu,
                    t=self.default_t, plane_stress=self.default_plane_stress,
                )
                elements_queue.append(el)
                eid += 1

        # Pre-ajustar la vista para que la animación se vea con los límites correctos
        xs = [n.x for n in nodes_queue]
        ys = [n.y for n in nodes_queue]
        margin = max(max(xs) - min(xs), max(ys) - min(ys), 1.0) * 0.15
        self.ax.set_xlim(min(xs) - margin, max(xs) + margin)
        self.ax.set_ylim(min(ys) - margin, max(ys) + margin)
        self._render_grid()
        self.canvas.draw_idle()

        if not animate:
            # Modo no-animado: añade todo en bloque
            for n in nodes_queue:
                self.structure.add_node(n)
                self._render_node(n)
            for el in elements_queue:
                self.structure.add_element(el)
                self._render_element(el)
            self._update_counter()
            self.structure_changed.emit()
            self.canvas.draw_idle()
            self.status.setText(
                f"Malla {nx}×{ny}: {len(nodes_queue)} nodos, "
                f"{len(elements_queue)} elementos"
            )
            return

        # Modo animado: encolar la inserción con QTimer
        total = len(nodes_queue) + len(elements_queue)
        # Velocidad: total ≈ 1.5 s, mínimo 15 ms/tick, máximo 80 ms/tick
        delay = max(15, min(80, int(1500 / max(total, 1))))
        # Si la malla es enorme, añadir varios items por tick
        items_per_tick = max(1, total // 80)

        self._anim_nodes = nodes_queue
        self._anim_elements = elements_queue
        self._anim_step = 0
        self._anim_total = total
        self._anim_items_per_tick = items_per_tick
        self._anim_target_dims = (nx, ny)

        if hasattr(self, "_anim_timer") and self._anim_timer is not None:
            self._anim_timer.stop()
        self._anim_timer = QTimer(self)
        self._anim_timer.timeout.connect(self._anim_mesh_tick)
        self._anim_timer.start(delay)

    def _anim_mesh_tick(self) -> None:
        """Callback del QTimer: añade `items_per_tick` items al canvas."""
        for _ in range(self._anim_items_per_tick):
            step = self._anim_step
            n_nodes = len(self._anim_nodes)
            if step < n_nodes:
                n = self._anim_nodes[step]
                self.structure.add_node(n)
                self._render_node(n)
            elif step < self._anim_total:
                el = self._anim_elements[step - n_nodes]
                self.structure.add_element(el)
                self._render_element(el)
            else:
                # Terminado
                self._anim_timer.stop()
                self._anim_timer = None
                self._update_counter()
                self.structure_changed.emit()
                nx, ny = self._anim_target_dims
                self.status.setText(
                    f"Malla {nx}×{ny}: {len(self.structure.nodes)} nodos, "
                    f"{len(self.structure.elements)} elementos"
                )
                self.canvas.draw_idle()
                return
            self._anim_step += 1
            # Actualizar el contador en vivo
            self._update_counter()
        self.canvas.draw_idle()

    # ---------- Renderizado ----------
    def _fit_view(self) -> None:
        if hasattr(self, "pg_canvas") and self.pg_canvas is not None:
            self.pg_canvas.fit_to_structure(self.structure)

    # ---------- Handlers del PgCanvas (sustituyen a los mpl_connect) ----------
    def _on_pg_left_click(self, x: float, y: float) -> None:
        """Click izquierdo en el canvas → ejecuta la acción del modo activo."""
        xs, ys = self._snap(x, y)
        if self.mode == Mode.ADD_NODE:
            self._push_undo()
            self._add_node(xs, ys)
        elif self.mode == Mode.ADD_RECT:
            if self._rect_first_corner is None:
                # Primer click: marcar esquina inicial
                self._rect_first_corner = (xs, ys)
                self.status_hint.setText(
                    f"Esquina 1: ({xs:g}, {ys:g}). Click en la esquina opuesta "
                    f"para crear el rectángulo. Esc cancela."
                )
            else:
                # Segundo click: crear el Q4
                x0, y0 = self._rect_first_corner
                self._rect_first_corner = None
                self._clear_rect_preview()
                if abs(xs - x0) < 1e-9 or abs(ys - y0) < 1e-9:
                    self.status_hint.setText("Rectángulo degenerado (área cero), cancelado")
                    return
                self._create_rectangle(x0, y0, xs, ys)
        elif self.mode == Mode.ADD_Q4:
            n = self._pick_node(x, y)
            if n is not None:
                self._handle_q4_click(n)
        elif self.mode == Mode.APPLY_BC:
            n = self._pick_node(x, y)
            if n is not None:
                self._push_undo()
                self._open_bc_dialog(n)
        elif self.mode == Mode.APPLY_LOAD:
            n = self._pick_node(x, y)
            if n is not None:
                self._push_undo()
                self._open_load_dialog(n)
        elif self.mode == Mode.DELETE:
            self._push_undo()
            self._delete_at(x, y)
        elif self.mode == Mode.SELECT:
            n = self._pick_node(x, y)
            self._selected_node = n
            if n is not None:
                # Sync con ModelTree / Inspector
                if hasattr(self, "inspector") and self.inspector is not None:
                    self.inspector.show_node(n)
            self._render_overlay()

    def _on_pg_right_click(self, x: float, y: float) -> None:
        """Click derecho: menú contextual sobre el objeto bajo el cursor."""
        from PySide6.QtWidgets import QMenu
        from PySide6.QtGui import QCursor
        node = self._pick_node(x, y)
        menu = QMenu(self)
        if node is not None:
            menu.addAction(f"Nodo {node.id + 1}").setEnabled(False)
            menu.addSeparator()
            menu.addAction("Editar coords / cargas / apoyos…",
                           lambda n=node: self._open_node_edit(n))
            menu.addAction("Aplicar / editar apoyo…",
                           lambda n=node: (self._push_undo(), self._open_bc_dialog(n)))
            menu.addAction("Aplicar / editar carga…",
                           lambda n=node: (self._push_undo(), self._open_load_dialog(n)))
            menu.addSeparator()
            act_del = menu.addAction("Borrar nodo")
            act_del.triggered.connect(
                lambda _, n=node: (self._push_undo(), self._delete_node(n))
            )
        else:
            el = self._pick_element(x, y)
            if el is not None:
                menu.addAction(f"Elemento E{el.id + 1}").setEnabled(False)
                menu.addSeparator()
                act_del = menu.addAction("Borrar elemento")
                def _del_el(_=None, e=el):
                    self._push_undo()
                    self.structure.elements.remove(e)
                    if hasattr(self.pg_canvas, "unrender_element"):
                        self.pg_canvas.unrender_element(e.id)
                    self._update_counter()
                    if hasattr(self, "model_tree") and self.model_tree is not None:
                        self.model_tree.refresh()
                    self.structure_changed.emit()
                act_del.triggered.connect(_del_el)
            else:
                # Click en vacío: opciones globales
                menu.addAction("Agregar nodo aquí",
                               lambda xx=x, yy=y: (self._push_undo(),
                                                   self._add_node(*self._snap(xx, yy))))
                menu.addAction("Ajustar vista (Home)", self.reset_view)
        menu.exec(QCursor.pos())

    def _on_pg_left_double_click(self, x: float, y: float) -> None:
        """Doble-click izquierdo: abre el diálogo de edición del nodo."""
        n = self._pick_node(x, y)
        if n is not None:
            self._open_node_edit(n)

    def _on_pg_rubber_band(self, xmin: float, ymin: float,
                          xmax: float, ymax: float) -> None:
        """El usuario terminó un rubber-band: seleccionar todos los nodos adentro."""
        if not self.structure or not self.structure.nodes:
            return
        # Si el rectángulo es muy pequeño (no fue un drag real), ignorar
        if (xmax - xmin) < 1e-9 or (ymax - ymin) < 1e-9:
            return
        # Recolectar nodos dentro del rectángulo
        selected = [n for n in self.structure.nodes
                    if xmin <= n.x <= xmax and ymin <= n.y <= ymax]
        if not selected:
            self.status_hint.setText("Rubber-band: ningún nodo dentro del área")
            self._selected_nodes = []
            self._render_overlay()
            return
        self._selected_nodes = selected
        self._selected_node = selected[0]   # sync con el single-select API
        # Mostrar en el Inspector el primero seleccionado
        if hasattr(self, "inspector") and self.inspector is not None:
            self.inspector.show_node(selected[0])
        self.status_hint.setText(
            f"Multi-selección: {len(selected)} nodo(s). "
            f"Ctrl+A selecciona todos. Apoyo/Carga los aplica a todos."
        )
        # Si el modo es APPLY_BC o APPLY_LOAD, aplicar al grupo
        if self.mode == Mode.APPLY_BC:
            self._push_undo()
            self._apply_bc_to_multi(selected)
        elif self.mode == Mode.APPLY_LOAD:
            self._push_undo()
            self._apply_load_to_multi(selected)
        else:
            # Solo destacar visualmente
            self._render_overlay_multi()

    def _render_overlay_multi(self) -> None:
        """Dibuja marcadores rosa sobre los nodos multi-seleccionados."""
        # Aprovecha el overlay existente para mostrar el primero;
        # el resto se marcan con scatter adicional
        if hasattr(self, "pg_canvas") and self.pg_canvas is not None:
            self.pg_canvas.render_overlay(
                hovered=self._hovered_node,
                selected=self._selected_node,
                pending_q4=self._pending_q4_nodes,
            )
            # Sobreponer marker para los demás nodos seleccionados
            import pyqtgraph as pg
            for n in self._selected_nodes:
                if n is self._selected_node:
                    continue
                mk = pg.ScatterPlotItem(
                    pos=[(n.x, n.y)], size=11,
                    brush=pg.mkBrush("#ff0066"),
                    pen=pg.mkPen("#ff0066"),
                )
                mk.setZValue(15)
                self.pg_canvas.addItem(mk)
                self.pg_canvas._overlay_artists.append(mk)

    def _apply_bc_to_multi(self, nodes: list[Node]) -> None:
        """Aplica restricciones a un grupo de nodos vía diálogo único."""
        d = RestraintDialog(nodes[0], self)
        d.setWindowTitle(f"Apoyos — {len(nodes)} nodos seleccionados")
        if d.exec() == QDialog.Accepted:
            for n in nodes:
                n.restraint_x = d.cb_x.isChecked()
                n.restraint_y = d.cb_y.isChecked()
                self._refresh_node(n)
            if hasattr(self, "model_tree") and self.model_tree is not None:
                self.model_tree.refresh()
            self.structure_changed.emit()
            self.status_hint.setText(
                f"Apoyos aplicados a {len(nodes)} nodo(s)"
            )

    def _apply_load_to_multi(self, nodes: list[Node]) -> None:
        """Aplica cargas a un grupo de nodos vía diálogo único."""
        d = LoadDialog(nodes[0], self)
        d.setWindowTitle(f"Cargas — {len(nodes)} nodos seleccionados")
        if d.exec() == QDialog.Accepted:
            try:
                fx = float(d.fx.text() or 0)
                fy = float(d.fy.text() or 0)
            except ValueError:
                QMessageBox.critical(self, "Error", "Carga inválida")
                return
            for n in nodes:
                n.load_x = fx
                n.load_y = fy
                self._refresh_node(n)
            if hasattr(self, "model_tree") and self.model_tree is not None:
                self.model_tree.refresh()
            self.structure_changed.emit()
            self.status_hint.setText(
                f"Cargas aplicadas a {len(nodes)} nodo(s)"
            )

    def _on_pg_key_escape(self) -> None:
        """Esc: cancela la acción actual (Q4 parcial, rectángulo, etc.)."""
        cancelled = False
        if self._pending_q4_nodes:
            self._pending_q4_nodes = []
            cancelled = True
        if self._rect_first_corner is not None:
            self._rect_first_corner = None
            self._clear_rect_preview()
            cancelled = True
        if self._selected_node is not None:
            self._selected_node = None
            cancelled = True
        if self._selected_nodes:
            self._selected_nodes = []
            cancelled = True
        if cancelled:
            self._render_overlay()
            self.status_hint.setText("Acción cancelada")

    def _select_all_nodes(self) -> None:
        """Ctrl+A: seleccionar todos los nodos del modelo."""
        if not self.structure or not self.structure.nodes:
            return
        self._selected_nodes = list(self.structure.nodes)
        self._selected_node = self._selected_nodes[0]
        self._render_overlay_multi()
        self.status_hint.setText(
            f"Seleccionados {len(self._selected_nodes)} nodos. "
            f"Ahora podés aplicar Apoyo o Carga al grupo."
        )

    def _open_node_edit(self, n: Node) -> None:
        """Diálogo unificado para editar TODO sobre un nodo: coords + BC + cargas."""
        from PySide6.QtWidgets import (QDialog, QFormLayout, QLineEdit, QCheckBox,
                                       QDialogButtonBox, QVBoxLayout, QLabel)
        d = QDialog(self)
        d.setWindowTitle(f"Editar Nodo {n.id + 1}")
        lay = QVBoxLayout(d)
        title = QLabel(f"<b>Nodo {n.id + 1}</b>")
        title.setStyleSheet("color:#1F4E78; font-size:11pt;")
        lay.addWidget(title)
        form = QFormLayout()
        ex = QLineEdit(repr(float(n.x)))
        ey = QLineEdit(repr(float(n.y)))
        cb_x = QCheckBox("Restringir en X"); cb_x.setChecked(n.restraint_x)
        cb_y = QCheckBox("Restringir en Y"); cb_y.setChecked(n.restraint_y)
        fx = QLineEdit(repr(float(n.load_x)))
        fy = QLineEdit(repr(float(n.load_y)))
        form.addRow("x (m):", ex)
        form.addRow("y (m):", ey)
        form.addRow(cb_x)
        form.addRow(cb_y)
        form.addRow("Fx (N):", fx)
        form.addRow("Fy (N):", fy)
        lay.addLayout(form)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(d.accept)
        bb.rejected.connect(d.reject)
        lay.addWidget(bb)
        if d.exec() == QDialog.Accepted:
            try:
                self._push_undo()
                n.x = float(ex.text())
                n.y = float(ey.text())
                n.restraint_x = cb_x.isChecked()
                n.restraint_y = cb_y.isChecked()
                n.load_x = float(fx.text() or 0)
                n.load_y = float(fy.text() or 0)
                self._refresh_node(n)
                if hasattr(self, "model_tree") and self.model_tree is not None:
                    self.model_tree.refresh()
                if hasattr(self, "inspector") and self.inspector is not None:
                    self.inspector.show_node(n)
                self.structure_changed.emit()
            except ValueError as e:
                QMessageBox.critical(self, "Error", f"Valor inválido: {e}")

    def _on_pg_mouse_move(self, x: float, y: float) -> None:
        """Mouse move: snap + hover highlight + tooltip flotante + preview rect."""
        from PySide6.QtWidgets import QToolTip
        from PySide6.QtGui import QCursor
        xs, ys = self._snap(x, y)
        self.status_cursor.setText(f"Cursor: ({xs:.4g}, {ys:.4g})")

        # Preview del rectángulo cuando hay primera esquina marcada
        if self.mode == Mode.ADD_RECT and self._rect_first_corner is not None:
            self._draw_rect_preview(self._rect_first_corner, (xs, ys))
        if self.mode in (Mode.ADD_Q4, Mode.APPLY_BC, Mode.APPLY_LOAD,
                         Mode.DELETE, Mode.SELECT):
            new_hover = self._pick_node(x, y)
        else:
            new_hover = None
        if new_hover is not self._hovered_node:
            self._hovered_node = new_hover
            self._render_overlay()
        # Tooltip flotante con info del nodo bajo el cursor
        if new_hover is not None:
            n = new_hover
            lines = [f"<b>Nodo {n.id + 1}</b>",
                     f"x = {n.x:g}   y = {n.y:g}"]
            if n.restraint_x or n.restraint_y:
                fixed = []
                if n.restraint_x:
                    fixed.append("X")
                if n.restraint_y:
                    fixed.append("Y")
                lines.append("Fijo en: " + ", ".join(fixed))
            if n.load_x or n.load_y:
                lines.append(f"F = ({n.load_x:g}, {n.load_y:g}) N")
            QToolTip.showText(QCursor.pos(), "<br>".join(lines), self.pg_canvas)
        else:
            QToolTip.hideText()

    def _on_pg_mouse_leave(self) -> None:
        self.status_cursor.setText("Cursor: —")
        if self._hovered_node is not None:
            self._hovered_node = None
            self._render_overlay()

    def _on_pg_key_delete(self) -> None:
        if self._selected_node is not None:
            self._push_undo()
            self._delete_node(self._selected_node)
            self._selected_node = None

    # ---------- Renderizado incremental (delegan al PgCanvas) ----------
    def _update_counter(self) -> None:
        if hasattr(self, "status_count"):
            n_nodes = len(self.structure.nodes) if self.structure else 0
            n_elems = len(self.structure.elements) if self.structure else 0
            self.status_count.setText(f"Nodos: {n_nodes} · Elementos: {n_elems}")

    @staticmethod
    def _safe_remove(artists: list) -> None:
        """Compat: remueve artistas (de matplotlib, ya casi sin uso en PgCanvas)."""
        for a in artists:
            try:
                a.remove()
            except (NotImplementedError, ValueError, AttributeError):
                pass

    def _clear_all_artists(self) -> None:
        # Delegación al PgCanvas
        if hasattr(self, "pg_canvas") and self.pg_canvas is not None:
            self.pg_canvas.clear_all()
        # Limpiar también el tracking del editor (espejo)
        self._artists_per_node.clear()
        self._artists_per_element.clear()
        self._overlay_artists.clear()
        self._grid_artists.clear()

    def _render_grid(self) -> None:
        # Delegación al PgCanvas
        if hasattr(self, "pg_canvas") and self.pg_canvas is not None:
            self.pg_canvas.set_grid_size(self.grid_size)
            self.pg_canvas.set_grid_visible(self.show_grid)
            self.pg_canvas._refresh_grid()

    def _render_node(self, n: Node) -> list:
        """Delegación a PgCanvas (rendering 50+ fps).

        El código de matplotlib debajo está deshabilitado; PgCanvas hace el trabajo.
        """
        if hasattr(self, "pg_canvas") and self.pg_canvas is not None:
            self.pg_canvas.render_node(n)
            return []
        # ===== Código matplotlib (fallback) =====
        artists = []
        line, = self.ax.plot(n.x, n.y, "o", color="black", markersize=7, zorder=5)
        artists.append(line)
        if self.show_labels:
            ann = self.ax.annotate(
                f"N{n.id + 1}", (n.x, n.y),
                textcoords="offset points", xytext=(7, 7),
                fontsize=9, fontweight="bold")
            artists.append(ann)
        # Apoyos
        if n.restraint_x and n.restraint_y:
            ln, = self.ax.plot(n.x, n.y, "s", color="green", markersize=18,
                               markerfacecolor="none", markeredgewidth=2, zorder=4)
            artists.append(ln)
        elif n.restraint_x or n.restraint_y:
            ln, = self.ax.plot(n.x, n.y, "^", color="green", markersize=15,
                               markerfacecolor="none", markeredgewidth=2, zorder=4)
            artists.append(ln)
        # Cargas
        if n.load_x or n.load_y:
            norm = np.hypot(n.load_x, n.load_y)
            xlim = self.ax.get_xlim()
            size = max((xlim[1] - xlim[0]) * 0.08, 0.1)
            dx_arrow = (n.load_x / norm) * size if norm else 0
            dy_arrow = (n.load_y / norm) * size if norm else 0
            ann_arrow = self.ax.annotate(
                "", xy=(n.x + dx_arrow, n.y + dy_arrow), xytext=(n.x, n.y),
                arrowprops=dict(arrowstyle="->", color="red", lw=2), zorder=6)
            artists.append(ann_arrow)
            if self.show_labels:
                ann_lbl = self.ax.annotate(
                    f"({n.load_x:g}, {n.load_y:g}) N",
                    (n.x + dx_arrow, n.y + dy_arrow),
                    fontsize=8, color="red")
                artists.append(ann_lbl)
        self._artists_per_node[n.id] = artists
        return artists

    def _render_element(self, el: Q4Element) -> list:
        """Delegación a PgCanvas."""
        if hasattr(self, "pg_canvas") and self.pg_canvas is not None:
            self.pg_canvas.render_element(el)
            return []
        # ===== Código matplotlib (fallback) =====
        artists = []
        xy = [(n.x, n.y) for n in el.nodes]
        poly = Polygon(xy, closed=True, facecolor="#cfe2f3",
                       edgecolor="#1f77b4", alpha=0.7, linewidth=1.5)
        self.ax.add_patch(poly)
        artists.append(poly)
        if self.show_labels:
            cx = np.mean([n.x for n in el.nodes])
            cy = np.mean([n.y for n in el.nodes])
            ann = self.ax.annotate(
                f"E{el.id + 1}", (cx, cy), color="#1f77b4",
                ha="center", va="center", fontsize=9, fontweight="bold")
            artists.append(ann)
        self._artists_per_element[el.id] = artists
        return artists

    def _render_overlay(self) -> None:
        """Render no-persistente: hover ring + preview Q4 pendiente + selected."""
        if hasattr(self, "pg_canvas") and self.pg_canvas is not None:
            self.pg_canvas.render_overlay(
                hovered=self._hovered_node,
                selected=self._selected_node,
                pending_q4=self._pending_q4_nodes,
            )
            return
        # ===== Código matplotlib (fallback) =====
        self._safe_remove(self._overlay_artists)
        self._overlay_artists.clear()

        # Live preview del Q4 parcial
        if len(self._pending_q4_nodes) >= 2:
            xy_pending = [(n.x, n.y) for n in self._pending_q4_nodes]
            if len(self._pending_q4_nodes) >= 3:
                xs = [p[0] for p in xy_pending + [xy_pending[0]]]
                ys = [p[1] for p in xy_pending + [xy_pending[0]]]
            else:
                xs = [p[0] for p in xy_pending]
                ys = [p[1] for p in xy_pending]
            ln, = self.ax.plot(xs, ys, "--", color="#ff9800",
                                linewidth=2, alpha=0.7, zorder=3)
            self._overlay_artists.append(ln)
        # Marcadores naranjas + número de orden en pending Q4
        for idx, n in enumerate(self._pending_q4_nodes, start=1):
            mk, = self.ax.plot(n.x, n.y, "o", color="#ff9800",
                                markersize=12, zorder=6)
            self._overlay_artists.append(mk)
            tx = self.ax.annotate(
                f"{idx}", (n.x, n.y),
                color="white", fontsize=8, fontweight="bold",
                ha="center", va="center", zorder=7)
            self._overlay_artists.append(tx)
        # Selected
        if self._selected_node is not None:
            mk, = self.ax.plot(self._selected_node.x, self._selected_node.y,
                                "o", color="#ff0066", markersize=11, zorder=6)
            self._overlay_artists.append(mk)
        # Hover ring
        if self._hovered_node is not None and self._hovered_node not in self._pending_q4_nodes:
            mk, = self.ax.plot(self._hovered_node.x, self._hovered_node.y,
                                "o", color="none",
                                markeredgecolor="#1F4E78", markeredgewidth=2,
                                markersize=18, zorder=6)
            self._overlay_artists.append(mk)
        self.canvas.draw_idle()

    def _add_node_incremental(self, n: Node) -> None:
        """Añade UN nodo al canvas sin redibujar todo."""
        self._render_node(n)
        self._update_counter()
        # PgCanvas no necesita draw_idle: ya actualiza al añadir items
        if hasattr(self, "model_tree") and self.model_tree is not None:
            self.model_tree.refresh()

    def _add_element_incremental(self, el: Q4Element) -> None:
        """Añade UN elemento al canvas sin redibujar todo."""
        self._render_element(el)
        self._update_counter()
        if hasattr(self, "model_tree") and self.model_tree is not None:
            self.model_tree.refresh()

    def _refresh_node(self, n: Node) -> None:
        """Re-renderiza un nodo (uso: tras cambiar BC, carga, o mover)."""
        if hasattr(self, "pg_canvas") and self.pg_canvas is not None:
            self.pg_canvas.render_node(n)  # internamente quita y vuelve a dibujar
            # Y los elementos que lo contienen
            for el in self.structure.elements:
                if n in el.nodes:
                    self.pg_canvas.render_element(el)
            return
        # ===== Código matplotlib (fallback) =====
        if n.id in self._artists_per_node:
            self._safe_remove(self._artists_per_node[n.id])
            del self._artists_per_node[n.id]
        self._render_node(n)
        for el in self.structure.elements:
            if n in el.nodes:
                if el.id in self._artists_per_element:
                    self._safe_remove(self._artists_per_element[el.id])
                    del self._artists_per_element[el.id]
                self._render_element(el)
        self.canvas.draw_idle()

    def redraw(self, fit_view: bool = True) -> None:
        """Render COMPLETO desde cero. Usar tras undo o set_structure."""
        self._update_counter()
        if hasattr(self, "pg_canvas") and self.pg_canvas is not None:
            self.pg_canvas.render_all(self.structure)
            self.pg_canvas.render_overlay(
                hovered=self._hovered_node,
                selected=self._selected_node,
                pending_q4=self._pending_q4_nodes,
            )
            if fit_view:
                self.pg_canvas.fit_to_structure(self.structure)
            return
        # Fallback matplotlib eliminado

    def reset_view(self) -> None:
        """Ajusta la vista a la estructura completa (botón 'zoom to fit')."""
        if hasattr(self, "pg_canvas") and self.pg_canvas is not None:
            self.pg_canvas.fit_to_structure(self.structure)
            self.pg_canvas._refresh_grid()
