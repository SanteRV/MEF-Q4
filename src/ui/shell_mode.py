"""Modo LÁMINA (flat shell) — pestaña autocontenida del aplicativo.

Para qué sirve: es la interfaz del modelo SHELL (cap. 01.01.04 del documento
teórico). El usuario define una lámina rectangular a×b, su malla, material,
carga transversal, carga en el plano y tipo de apoyo; el programa resuelve
con el elemento flat shell de 20 GDL (membrana + flexión desacopladas) y
muestra:

    - la deformada en una vista 3D, con los desplazamientos en el plano
      (u, v) y fuera del plano (w) superpuestos,
    - un resumen con u_max, w_max, momentos y esfuerzos de la fibra
      superior e inferior (membrana + flexión),
    - las matrices didácticas del primer elemento: los dos bloques de la
      ec. 1.4.10, la K^e completa de 20×20 y la B de 6×20 (ec. 1.4.11).

Arquitectura: familia paralela a los modos plane y placa — este módulo NO
toca el flujo de los otros dos; la ventana principal solo agrega la pestaña.
"""
from __future__ import annotations
import time
from typing import Optional

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import (
    QColor, QVector3D, QFont, QCursor, QStandardItemModel, QStandardItem,
)
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QFormLayout, QGroupBox, QLineEdit,
    QComboBox, QPushButton, QLabel, QSlider, QTabWidget, QTableView,
    QMessageBox, QApplication, QHeaderView,
)
import pyqtgraph.opengl as gl

from ..fem.node_shell import NodeShell
from ..fem.shell_element import ShellElement
from ..fem.structure_shell import StructureShell
from ..fem.solver_shell import solve_shell, FEMResultShell


# ---------------------------------------------------------------- helpers
def _fmt(v: float) -> str:
    """Números con precisión completa; ceros residuales como 0."""
    if abs(v) < 1e-13:
        return "0"
    return f"{v:.15g}"


def _matrix_model(arr: np.ndarray) -> QStandardItemModel:
    """Convierte una matriz numpy en un modelo de tabla (display :.15g)."""
    arr = np.atleast_2d(np.asarray(arr, dtype=float))
    rows, cols = arr.shape
    model = QStandardItemModel(rows, cols)
    model.setHorizontalHeaderLabels([str(c + 1) for c in range(cols)])
    model.setVerticalHeaderLabels([str(r + 1) for r in range(rows)])
    for r in range(rows):
        for c in range(cols):
            it = QStandardItem(_fmt(float(arr[r, c])))
            it.setEditable(False)
            it.setTextAlignment(Qt.AlignmentFlag.AlignRight
                                | Qt.AlignmentFlag.AlignVCenter)
            model.setItem(r, c, it)
    return model


# ---------------------------------------------------------------- vista 3D
class ShellCanvas3D(gl.GLViewWidget):
    """Vista 3D de la lámina deformada.

    A diferencia del modo placa, aquí la superficie se dibuja en la posición
    (x + u, y + v, w): el flat shell mueve los nodos también DENTRO del
    plano, y verlo es lo que distingue el comportamiento de membrana del de
    flexión.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setBackgroundColor(QColor("#FAFAFC"))
        self.setCameraPosition(distance=3.0, elevation=32, azimuth=-60)
        self._dyn_items: list = []

        grid = gl.GLGridItem()
        grid.setSize(x=4, y=4)
        grid.setSpacing(x=0.5, y=0.5)
        grid.setColor((180, 180, 188, 130))
        self.addItem(grid)
        font = QFont("Arial", 11)
        font.setBold(True)
        for end, color, rgba, lbl in [
            ((1.0, 0, 0), (0.90, 0.18, 0.18, 1), (230, 46, 46, 255), "X"),
            ((0, 1.0, 0), (0.18, 0.70, 0.18, 1), (40, 165, 40, 255), "Y"),
            ((0, 0, 0.5), (0.18, 0.32, 0.92, 1), (46, 82, 235, 255), "Z"),
        ]:
            ln = gl.GLLinePlotItem(
                pos=np.array([[0, 0, 0], list(end)], dtype=np.float64),
                color=color, width=2.5, antialias=True)
            ln.setGLOptions("opaque")
            self.addItem(ln)
            try:
                t = gl.GLTextItem(pos=(end[0] * 1.12, end[1] * 1.12,
                                       end[2] * 1.2 + (0.04 if lbl == "Z" else 0)),
                                  text=lbl, color=rgba, font=font)
                self.addItem(t)
            except Exception:
                pass

    def show_solution(self, X: np.ndarray, Y: np.ndarray, W: np.ndarray,
                      U: np.ndarray, V: np.ndarray, scale: float,
                      x_lim: tuple[float, float],
                      y_lim: tuple[float, float]) -> None:
        """Dibuja la lámina deformada amplificada por 'scale'.

        X, Y son las coordenadas sin deformar (nx+1, ny+1); U, V, W los
        desplazamientos nodales en las tres direcciones.
        """
        for it in self._dyn_items:
            try:
                self.removeItem(it)
            except Exception:
                pass
        self._dyn_items.clear()

        Xd = X + U * scale
        Yd = Y + V * scale
        Zd = W * scale

        # Color por desplazamiento total: azul = descenso, rojo = ascenso
        w_abs = float(np.max(np.abs(W))) or 1.0
        norm = W / w_abs
        colors = np.zeros((*W.shape, 4))
        neg = norm < 0
        colors[..., 0] = np.where(neg, 0.55 * (1 + norm), 0.55 + 0.45 * norm)
        colors[..., 1] = 0.60 * (1 - np.abs(norm))
        colors[..., 2] = np.where(neg, 0.55 + 0.45 * (-norm), 0.55 * (1 - norm))
        colors[..., 3] = 1.0

        # GLSurfacePlotItem exige grillas regulares; con desplazamiento en el
        # plano la malla deja de serlo, asi que se dibuja como mallado de
        # lineas sobre la posicion deformada real.
        nx, ny = W.shape
        caras = []
        for i in range(nx):
            for j in range(ny - 1):
                caras.append([Xd[i, j], Yd[i, j], Zd[i, j]])
                caras.append([Xd[i, j + 1], Yd[i, j + 1], Zd[i, j + 1]])
        for j in range(ny):
            for i in range(nx - 1):
                caras.append([Xd[i, j], Yd[i, j], Zd[i, j]])
                caras.append([Xd[i + 1, j], Yd[i + 1, j], Zd[i + 1, j]])
        ln = gl.GLLinePlotItem(pos=np.array(caras),
                               color=(0.20, 0.30, 0.55, 0.85),
                               width=1.4, antialias=True, mode="lines")
        self.addItem(ln)
        self._dyn_items.append(ln)

        pts = gl.GLScatterPlotItem(
            pos=np.column_stack([Xd.ravel(), Yd.ravel(), Zd.ravel()]),
            color=colors.reshape(-1, 4), size=6.0, pxMode=True)
        self.addItem(pts)
        self._dyn_items.append(pts)

        # Contorno sin deformar (referencia en z = 0)
        x0, x1 = x_lim
        y0, y1 = y_lim
        borde = np.array([
            [x0, y0, 0], [x1, y0, 0], [x1, y1, 0], [x0, y1, 0], [x0, y0, 0],
        ])
        ref = gl.GLLinePlotItem(pos=borde, color=(0.35, 0.38, 0.42, 0.9),
                                width=2.0, antialias=True)
        self.addItem(ref)
        self._dyn_items.append(ref)

        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        size = max(x1 - x0, y1 - y0)
        self.opts["center"] = QVector3D(cx, cy, 0.0)
        self.setCameraPosition(distance=max(size * 2.2, 1.5),
                               elevation=32, azimuth=-60)
        self.update()


# ---------------------------------------------------------------- widget
class ShellModeWidget(QWidget):
    """Pestaña "Lámina (shell)": entradas, solución, resumen y matrices."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._resultado: Optional[FEMResultShell] = None
        self._structure: Optional[StructureShell] = None
        self._grids: Optional[tuple] = None      # (X, Y, U, V, W)
        self._lims: tuple = ((0.0, 1.0), (0.0, 1.0))
        self._escala_auto: float = 1.0
        self._build_ui()

    # ------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        left = QWidget()
        left.setMaximumWidth(400)
        col = QVBoxLayout(left)
        col.setSpacing(8)

        gb = QGroupBox("Lámina rectangular (flat shell, 20 GDL por elemento)")
        form = QFormLayout(gb)
        self.ed_a = QLineEdit("2.0")
        self.ed_b = QLineEdit("1.0")
        self.ed_t = QLineEdit("0.01")
        self.ed_E = QLineEdit("2.1e11")
        self.ed_nu = QLineEdit("0.3")
        self.ed_nx = QLineEdit("8")
        self.ed_ny = QLineEdit("4")
        form.addRow("Lado a en X (m):", self.ed_a)
        form.addRow("Lado b en Y (m):", self.ed_b)
        form.addRow("Espesor t (m):", self.ed_t)
        form.addRow("E (Pa):", self.ed_E)
        form.addRow("ν (Poisson):", self.ed_nu)
        form.addRow("Malla Nx:", self.ed_nx)
        form.addRow("Malla Ny:", self.ed_ny)
        col.addWidget(gb)

        gb_carga = QGroupBox("Cargas")
        fc = QFormLayout(gb_carga)
        self.ed_q = QLineEdit("-1000.0")
        self.ed_fx = QLineEdit("0.0")
        self.ed_fy = QLineEdit("0.0")
        fc.addRow("q transversal (N/m², − hacia abajo):", self.ed_q)
        fc.addRow("Fx total en el borde x = a (N):", self.ed_fx)
        fc.addRow("Fy total en el borde x = a (N):", self.ed_fy)
        self.cmb_carga = QComboBox()
        self.cmb_carga.addItems([
            "Directo a los nodos, q·A/4 (ec. 1.3.19)",
            "Vigas en 1 dirección, luz en Y (ec. 1.3.20)",
            "Vigas en 1 dirección, luz en X (ec. 1.3.20)",
        ])
        fc.addRow("Reparto de q:", self.cmb_carga)
        self.cmb_bc = QComboBox()
        self.cmb_bc.addItems([
            "Simplemente apoyada (4 bordes)",
            "Empotrada (4 bordes)",
            "Voladizo (empotrada en x = 0)",
        ])
        fc.addRow("Apoyo:", self.cmb_bc)
        col.addWidget(gb_carga)

        self.btn_calc = QPushButton("Calcular lámina")
        self.btn_calc.setProperty("primary", True)
        self.btn_calc.clicked.connect(self.calculate)
        col.addWidget(self.btn_calc)

        self.lbl_status = QLabel(
            "Defina la lámina y presione Calcular. El flat shell superpone "
            "el comportamiento de membrana (plane) con el de flexión (plate)."
        )
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setStyleSheet("color: #555; font-size: 11px;")
        col.addWidget(self.lbl_status)

        self.tabs_res = QTabWidget()
        self.tbl_resumen = QTableView()
        self.tbl_resumen.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)
        self.tabs_res.addTab(self.tbl_resumen, "Resumen")

        # Resultados nodales: los 5 GDL resueltos y la reacción del apoyo
        self.tbl_nodos = QTableView()
        self.tbl_nodos.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents)
        self.tabs_res.addTab(self.tbl_nodos, "Nodos")

        mat_tab = QWidget()
        mv = QVBoxLayout(mat_tab)
        self.cmb_matriz = QComboBox()
        self.cmb_matriz.addItems([
            "K^e (20×20) del elemento 1 — orden nodal",
            "Bloque K_plane (8×8) — ec. 1.4.10",
            "Bloque K_plate (12×12) — ec. 1.4.10",
            "B (6×20) en el centro, z = t/2 — ec. 1.4.11",
            "D (6×6) por bloques",
        ])
        self.cmb_matriz.currentIndexChanged.connect(self._update_matrix_view)
        mv.addWidget(self.cmb_matriz)
        self.tbl_matriz = QTableView()
        mv.addWidget(self.tbl_matriz, 1)
        self.tabs_res.addTab(mat_tab, "Matrices")
        col.addWidget(self.tabs_res, 1)

        root.addWidget(left, 0)

        right = QWidget()
        rcol = QVBoxLayout(right)
        self.canvas = ShellCanvas3D()
        rcol.addWidget(self.canvas, 1)
        fila = QHBoxLayout()
        fila.addWidget(QLabel("Escala de la deformada:"))
        self.sl_escala = QSlider(Qt.Orientation.Horizontal)
        self.sl_escala.setRange(1, 300)
        self.sl_escala.setValue(100)
        self.sl_escala.valueChanged.connect(self._redraw)
        fila.addWidget(self.sl_escala, 1)
        self.lbl_escala = QLabel("x auto")
        fila.addWidget(self.lbl_escala)
        rcol.addLayout(fila)
        root.addWidget(right, 1)

    # ------------------------------------------------------------ cálculo
    def _leer_float(self, ed: QLineEdit, nombre: str) -> float:
        try:
            return float(ed.text().strip())
        except ValueError:
            raise ValueError(f"Valor inválido en '{nombre}': {ed.text()!r}")

    def calculate(self) -> None:
        """Construye la malla, resuelve y actualiza vista, resumen y matrices."""
        try:
            a = self._leer_float(self.ed_a, "Lado a")
            b = self._leer_float(self.ed_b, "Lado b")
            t = self._leer_float(self.ed_t, "Espesor t")
            E = self._leer_float(self.ed_E, "E")
            nu = self._leer_float(self.ed_nu, "ν")
            q = self._leer_float(self.ed_q, "q transversal")
            fx = self._leer_float(self.ed_fx, "Fx")
            fy = self._leer_float(self.ed_fy, "Fy")
            nx = int(self._leer_float(self.ed_nx, "Nx"))
            ny = int(self._leer_float(self.ed_ny, "Ny"))
            if a <= 0 or b <= 0 or t <= 0 or E <= 0 or nx < 1 or ny < 1:
                raise ValueError("a, b, t, E deben ser > 0 y Nx, Ny >= 1.")
            if not (0 <= nu < 0.5):
                raise ValueError("ν debe estar en [0, 0.5).")
        except ValueError as e:
            QMessageBox.critical(self, "Datos inválidos", str(e))
            return

        self.lbl_status.setText("Calculando...")
        QApplication.setOverrideCursor(QCursor(Qt.CursorShape.WaitCursor))
        QApplication.processEvents()
        t0 = time.perf_counter()
        try:
            s = self._make_mesh(a, b, nx, ny, E, nu, t)
            self._apply_bc(s, a, b, self.cmb_bc.currentIndex())
            self._apply_edge_load(s, a, fx, fy)
            caso = ("nodos", "vigas_y", "vigas_x")[self.cmb_carga.currentIndex()]
            res = solve_shell(s, q_uniform=q, load_case=caso)
        except Exception as e:
            self.lbl_status.setText("Error en el cálculo.")
            QMessageBox.critical(self, "Error en el cálculo", str(e))
            return
        finally:
            QApplication.restoreOverrideCursor()
        dt = time.perf_counter() - t0

        self._resultado = res
        self._structure = s

        # Grillas de desplazamiento: nodo (i, j) -> id = j*(nx+1)+i
        X = np.zeros((nx + 1, ny + 1))
        Y = np.zeros((nx + 1, ny + 1))
        U = np.zeros((nx + 1, ny + 1))
        V = np.zeros((nx + 1, ny + 1))
        W = np.zeros((nx + 1, ny + 1))
        for j in range(ny + 1):
            for i in range(nx + 1):
                n = s.nodes[j * (nx + 1) + i]
                d = res.displacements[list(n.dofs)]
                X[i, j], Y[i, j] = n.x, n.y
                U[i, j], V[i, j], W[i, j] = d[0], d[1], d[2]
        self._grids = (X, Y, U, V, W)
        self._lims = ((0.0, a), (0.0, b))

        d_max = float(np.max(np.abs(np.stack([U, V, W]))))
        self._escala_auto = (0.2 * max(a, b) / d_max) if d_max > 0 else 1.0
        self._redraw()

        # ----- resumen -----
        w_max = float(W.flat[np.argmax(np.abs(W))])
        u_max = float(np.max(np.abs(U)))
        v_max = float(np.max(np.abs(V)))
        s_top = max(float(np.max(np.abs(er.stress_total_top)))
                    for er in res.elements)
        s_bot = max(float(np.max(np.abs(er.stress_total_bottom)))
                    for er in res.elements)
        s_mem = max(float(np.max(np.abs(er.stresses_center[:3])))
                    for er in res.elements)
        s_flex = max(float(np.max(np.abs(er.stresses_center[3:])))
                     for er in res.elements)
        M_max = max(float(np.max(np.abs(er.moments_center)))
                    for er in res.elements)
        t_txt = f"{dt * 1000:.0f} ms" if dt < 1.0 else f"{dt:.2f} s"
        filas = [
            ("w máximo (fuera del plano) (m)", _fmt(w_max)),
            ("|u| máximo (membrana en X) (m)", _fmt(u_max)),
            ("|v| máximo (membrana en Y) (m)", _fmt(v_max)),
            ("|σ| máximo de membrana (Pa)", _fmt(s_mem)),
            ("|σ| máximo de flexión en z = t/2 (Pa)", _fmt(s_flex)),
            ("|σ| total fibra superior (Pa)", _fmt(s_top)),
            ("|σ| total fibra inferior (Pa)", _fmt(s_bot)),
            ("Momento máximo |M| centro (N·m/m)", _fmt(M_max)),
            ("Elementos / nodos / GDL",
             f"{len(s.elements)} / {len(s.nodes)} / {s.n_dofs}"),
            ("Tiempo de cálculo", t_txt),
        ]
        model = QStandardItemModel(len(filas), 2)
        model.setHorizontalHeaderLabels(["Magnitud", "Valor"])
        for r, (k, v) in enumerate(filas):
            it_k, it_v = QStandardItem(k), QStandardItem(v)
            it_k.setEditable(False)
            it_v.setEditable(False)
            model.setItem(r, 0, it_k)
            model.setItem(r, 1, it_v)
        self.tbl_resumen.setModel(model)
        self._fill_node_table(s, res)

        self._update_matrix_view()
        self.lbl_status.setText(
            f"Cálculo completado en {t_txt} — {len(s.elements)} elementos, "
            f"{s.n_dofs} GDL (5 por nodo)."
        )

    def _fill_node_table(self, s: StructureShell, res: FEMResultShell) -> None:
        """Tabla nodo a nodo: los 5 GDL resueltos y la reacción del apoyo."""
        cols = ["Nodo", "x (m)", "y (m)", "u (m)", "v (m)", "w (m)",
                "θx (rad)", "θy (rad)", "Apoyo", "Rx (N)", "Ry (N)", "Rz (N)"]
        etiquetas = ("u", "v", "w", "θx", "θy")
        model = QStandardItemModel(len(s.nodes), len(cols))
        model.setHorizontalHeaderLabels(cols)
        for r, n in enumerate(s.nodes):
            d = res.displacements[list(n.dofs)]
            rc = res.reactions[list(n.dofs)]
            apoyado = any(n.restraints)
            valores = [
                str(n.id + 1), _fmt(n.x), _fmt(n.y),
                _fmt(d[0]), _fmt(d[1]), _fmt(d[2]), _fmt(d[3]), _fmt(d[4]),
                " ".join(e for e, on in zip(etiquetas, n.restraints)
                         if on) or "—",
                _fmt(rc[0]) if apoyado else "—",
                _fmt(rc[1]) if apoyado else "—",
                _fmt(rc[2]) if apoyado else "—",
            ]
            for c, txt in enumerate(valores):
                it = QStandardItem(txt)
                it.setEditable(False)
                if c > 0:
                    it.setTextAlignment(Qt.AlignmentFlag.AlignRight
                                        | Qt.AlignmentFlag.AlignVCenter)
                model.setItem(r, c, it)
        self.tbl_nodos.setModel(model)

    # ------------------------------------------------------------ helpers
    @staticmethod
    def _make_mesh(a: float, b: float, nx: int, ny: int,
                   E: float, nu: float, t: float) -> StructureShell:
        """Malla nx×ny de elementos shell, nodos CCW desde inferior-izquierda."""
        s = StructureShell()
        dx, dy = a / nx, b / ny
        at: dict[tuple[int, int], NodeShell] = {}
        nid = 0
        for j in range(ny + 1):
            for i in range(nx + 1):
                n = NodeShell(id=nid, x=i * dx, y=j * dy)
                s.add_node(n)
                at[(i, j)] = n
                nid += 1
        eid = 0
        for j in range(ny):
            for i in range(nx):
                s.add_element(ShellElement(
                    id=eid,
                    nodes=[at[(i, j)], at[(i + 1, j)],
                           at[(i + 1, j + 1)], at[(i, j + 1)]],
                    E=E, nu=nu, t=t))
                eid += 1
        return s

    @staticmethod
    def _apply_bc(s: StructureShell, a: float, b: float, tipo: int) -> None:
        """Aplica el apoyo elegido.

        tipo 0 = simplemente apoyada en los 4 bordes (w = 0 y el giro
                 alrededor del eje del borde);
        tipo 1 = empotrada en los 4 bordes;
        tipo 2 = voladizo empotrado en x = 0.

        En los tres casos se restringe también la membrana (u, v) en los
        bordes apoyados: sin ello el bloque de membrana queda sin sujeción
        y K_ff resulta singular.
        """
        tol = 1e-9
        for n in s.nodes:
            en_x0 = abs(n.x) < tol
            en_xa = abs(n.x - a) < tol
            en_y0 = abs(n.y) < tol
            en_yb = abs(n.y - b) < tol
            if tipo == 2:
                if en_x0:
                    n.restraint_u = n.restraint_v = n.restraint_w = True
                    n.restraint_rx = n.restraint_ry = True
                continue
            if not (en_x0 or en_xa or en_y0 or en_yb):
                continue
            n.restraint_u = n.restraint_v = True
            n.restraint_w = True
            if tipo == 1:
                n.restraint_rx = n.restraint_ry = True
            else:
                if en_x0 or en_xa:
                    n.restraint_rx = True
                if en_y0 or en_yb:
                    n.restraint_ry = True

    @staticmethod
    def _apply_edge_load(s: StructureShell, a: float,
                         fx: float, fy: float) -> None:
        """Reparte la carga en el plano por igual entre los nodos de x = a."""
        if fx == 0.0 and fy == 0.0:
            return
        borde = [n for n in s.nodes if abs(n.x - a) < 1e-9]
        if not borde:
            return
        for n in borde:
            n.load_fx += fx / len(borde)
            n.load_fy += fy / len(borde)

    def _redraw(self) -> None:
        """Redibuja con la escala del slider (en % de la automática)."""
        if self._grids is None:
            return
        X, Y, U, V, W = self._grids
        escala = self._escala_auto * (self.sl_escala.value() / 100.0)
        self.lbl_escala.setText(f"x {escala:.0f}")
        self.canvas.show_solution(X, Y, W, U, V, escala,
                                  self._lims[0], self._lims[1])

    def _update_matrix_view(self) -> None:
        """Muestra la matriz didáctica elegida del primer elemento."""
        if self._resultado is None or self._structure is None:
            return
        el = self._structure.elements[0]
        idx = self.cmb_matriz.currentIndex()
        if idx == 0:
            arr = el.stiffness_matrix()
        elif idx == 1:
            arr = el.stiffness_blocks()[0]
        elif idx == 2:
            arr = el.stiffness_blocks()[1]
        elif idx == 3:
            xs = [n.x for n in el.nodes]
            ys = [n.y for n in el.nodes]
            arr = el.B_matrix((min(xs) + max(xs)) / 2.0,
                              (min(ys) + max(ys)) / 2.0, el.t / 2.0)
        else:
            arr = el.D_matrix()
        self.tbl_matriz.setModel(_matrix_model(arr))
        self.tbl_matriz.resizeColumnsToContents()
