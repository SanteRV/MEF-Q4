"""Modo PLACA (flexión) — pestaña autocontenida del aplicativo.

Para qué sirve: es la interfaz del modelo PLATE (cap. 01.01.03 del
documento teórico). El usuario define una placa rectangular a×b, su malla,
material, carga transversal y tipo de apoyo; el programa resuelve con el
elemento rectangular de 12 GDL (Kirchhoff) y muestra:

    - la DEFORMADA w en una vista 3D (superficie coloreada, con escala),
    - un resumen con w_max, comparación con la solución analítica de
      Timoshenko (cuando la placa es cuadrada), momentos y tiempos,
    - las matrices didácticas del primer elemento: C (ec. 3.14),
      K^e (ec. 3.18) y Q en el centro (ec. 3.16).

Arquitectura: familia paralela al modo plane — este módulo NO toca el
flujo Q4 existente; la ventana principal solo agrega la pestaña.
"""
from __future__ import annotations
import time
from typing import Optional

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QVector3D, QFont, QCursor, QStandardItemModel, QStandardItem
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QFormLayout, QGroupBox, QLineEdit,
    QComboBox, QPushButton, QLabel, QSlider, QTabWidget, QTableView,
    QMessageBox, QApplication, QHeaderView,
)
import pyqtgraph.opengl as gl

from ..fem.node_plate import NodePlate
from ..fem.plate_element import PlateElement
from ..fem.structure_plate import StructurePlate
from ..fem.solver_plate import solve_plate, FEMResultPlate


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
class PlateCanvas3D(gl.GLViewWidget):
    """Vista 3D de la deformada w de la placa (superficie coloreada).

    Para qué sirve: en el modo placa el desplazamiento SÍ sale del plano
    (w en Z), así que la vista 3D muestra la física real: la placa
    hundiéndose bajo la carga. Colores: azul = mayor descenso, rojo =
    mayor ascenso, gris = sin desplazamiento.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setBackgroundColor(QColor("#FAFAFC"))
        self.setCameraPosition(distance=3.0, elevation=32, azimuth=-60)
        self._dyn_items: list = []

        # Grilla base y ejes etiquetados (X rojo, Y verde, Z azul)
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

    def show_solution(self, xs: np.ndarray, ys: np.ndarray,
                      W: np.ndarray, scale: float) -> None:
        """Dibuja la superficie deformada w (amplificada por 'scale').

        xs (nx+1,), ys (ny+1,), W (nx+1, ny+1) con W[i, j] = w en el nodo
        de la columna i (x) y la fila j (y).
        """
        for it in self._dyn_items:
            try:
                self.removeItem(it)
            except Exception:
                pass
        self._dyn_items.clear()

        Z = W * scale

        # Colores por valor de w: azul (minimo) -> gris (0) -> rojo (maximo)
        w_abs = float(np.max(np.abs(W))) or 1.0
        norm = W / w_abs                       # -1 .. 1
        colors = np.zeros((*W.shape, 4))
        neg = norm < 0
        # tono azul proporcional al descenso, rojo al ascenso
        colors[..., 0] = np.where(neg, 0.55 * (1 + norm), 0.55 + 0.45 * norm)
        colors[..., 1] = 0.60 * (1 - np.abs(norm))
        colors[..., 2] = np.where(neg, 0.55 + 0.45 * (-norm), 0.55 * (1 - norm))
        colors[..., 3] = 1.0

        surf = gl.GLSurfacePlotItem(x=xs, y=ys, z=Z, colors=colors,
                                    shader=None, smooth=True)
        surf.setGLOptions("opaque")
        self.addItem(surf)
        self._dyn_items.append(surf)

        # Malla de lineas sobre la superficie (para ver los elementos)
        segs = []
        for i in range(len(xs)):
            for j in range(len(ys) - 1):
                segs.append([xs[i], ys[j], Z[i, j]])
                segs.append([xs[i], ys[j + 1], Z[i, j + 1]])
        for j in range(len(ys)):
            for i in range(len(xs) - 1):
                segs.append([xs[i], ys[j], Z[i, j]])
                segs.append([xs[i + 1], ys[j], Z[i + 1, j]])
        ln = gl.GLLinePlotItem(pos=np.array(segs), color=(0.25, 0.28, 0.33, 0.55),
                               width=1.0, antialias=True, mode="lines")
        self.addItem(ln)
        self._dyn_items.append(ln)

        # Contorno de la placa SIN deformar (referencia en z = 0)
        x0, x1 = float(xs[0]), float(xs[-1])
        y0, y1 = float(ys[0]), float(ys[-1])
        borde = np.array([
            [x0, y0, 0], [x1, y0, 0], [x1, y1, 0], [x0, y1, 0], [x0, y0, 0],
        ])
        ref = gl.GLLinePlotItem(pos=borde, color=(0.35, 0.38, 0.42, 0.9),
                                width=2.0, antialias=True)
        self.addItem(ref)
        self._dyn_items.append(ref)

        # Encuadre
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        size = max(x1 - x0, y1 - y0)
        self.opts["center"] = QVector3D(cx, cy, 0.0)
        self.setCameraPosition(distance=max(size * 2.2, 1.5),
                               elevation=32, azimuth=-60)
        self.update()


# ---------------------------------------------------------------- widget
class PlateModeWidget(QWidget):
    """Pestaña "Placa (flexión)": entradas, solución, resumen y matrices."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._resultado: Optional[FEMResultPlate] = None
        self._W: Optional[np.ndarray] = None       # w por nodo, grilla (nx+1, ny+1)
        self._xs: Optional[np.ndarray] = None
        self._ys: Optional[np.ndarray] = None
        self._escala_auto: float = 1.0
        self._build_ui()

    # ------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        # ===== Panel izquierdo: entradas + resultados =====
        left = QWidget()
        left.setMaximumWidth(380)
        col = QVBoxLayout(left)
        col.setSpacing(8)

        gb_geo = QGroupBox("Placa rectangular (Kirchhoff, 12 GDL por elemento)")
        form = QFormLayout(gb_geo)
        self.ed_a = QLineEdit("1.0")
        self.ed_b = QLineEdit("1.0")
        self.ed_t = QLineEdit("0.01")
        self.ed_E = QLineEdit("2.1e11")
        self.ed_nu = QLineEdit("0.3")
        self.ed_nx = QLineEdit("8")
        self.ed_ny = QLineEdit("8")
        self.ed_q = QLineEdit("-1000.0")
        form.addRow("Lado a en X (m):", self.ed_a)
        form.addRow("Lado b en Y (m):", self.ed_b)
        form.addRow("Espesor t (m):", self.ed_t)
        form.addRow("E (Pa):", self.ed_E)
        form.addRow("ν (Poisson):", self.ed_nu)
        form.addRow("Malla Nx:", self.ed_nx)
        form.addRow("Malla Ny:", self.ed_ny)
        form.addRow("Carga q (N/m², − hacia abajo):", self.ed_q)
        self.cmb_bc = QComboBox()
        self.cmb_bc.addItems([
            "Simplemente apoyada (4 bordes)",
            "Empotrada (4 bordes)",
        ])
        form.addRow("Apoyo:", self.cmb_bc)
        col.addWidget(gb_geo)

        self.btn_calc = QPushButton("Calcular placa")
        self.btn_calc.setProperty("primary", True)
        self.btn_calc.clicked.connect(self.calculate)
        col.addWidget(self.btn_calc)

        self.lbl_status = QLabel("Defina la placa y presione Calcular.")
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setStyleSheet("color: #555; font-size: 11px;")
        col.addWidget(self.lbl_status)

        # Resumen + matrices en pestañas
        self.tabs_res = QTabWidget()
        self.tbl_resumen = QTableView()
        self.tbl_resumen.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)
        self.tabs_res.addTab(self.tbl_resumen, "Resumen")

        mat_tab = QWidget()
        mv = QVBoxLayout(mat_tab)
        self.cmb_matriz = QComboBox()
        self.cmb_matriz.addItems([
            "C (12×12) del elemento 1 (ec. 3.14)",
            "K^e (12×12) del elemento 1 (ec. 3.18)",
            "Q (3×12) en el centro del elemento 1 (ec. 3.16)",
        ])
        self.cmb_matriz.currentIndexChanged.connect(self._update_matrix_view)
        mv.addWidget(self.cmb_matriz)
        self.tbl_matriz = QTableView()
        mv.addWidget(self.tbl_matriz, 1)
        self.tabs_res.addTab(mat_tab, "Matrices")
        col.addWidget(self.tabs_res, 1)

        root.addWidget(left, 0)

        # ===== Panel derecho: vista 3D + escala =====
        right = QWidget()
        rcol = QVBoxLayout(right)
        self.canvas = PlateCanvas3D()
        rcol.addWidget(self.canvas, 1)
        fila = QHBoxLayout()
        fila.addWidget(QLabel("Escala de la deformada:"))
        self.sl_escala = QSlider(Qt.Orientation.Horizontal)
        self.sl_escala.setRange(1, 300)      # % de la escala automatica
        self.sl_escala.setValue(100)
        self.sl_escala.valueChanged.connect(self._redraw_surface)
        fila.addWidget(self.sl_escala, 1)
        self.lbl_escala = QLabel("x auto")
        fila.addWidget(self.lbl_escala)
        rcol.addLayout(fila)
        root.addWidget(right, 1)

    # ------------------------------------------------------------ calculo
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
            q = self._leer_float(self.ed_q, "Carga q")
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
            res = solve_plate(s, q_uniform=q)
        except Exception as e:
            self.lbl_status.setText("Error en el cálculo.")
            QMessageBox.critical(self, "Error en el cálculo", str(e))
            return
        finally:
            QApplication.restoreOverrideCursor()
        dt = time.perf_counter() - t0

        self._resultado = res
        self._structure = s

        # Grilla de w para la superficie 3D: nodo (i, j) -> id = j*(nx+1)+i
        xs = np.linspace(0.0, a, nx + 1)
        ys = np.linspace(0.0, b, ny + 1)
        W = np.zeros((nx + 1, ny + 1))
        for j in range(ny + 1):
            for i in range(nx + 1):
                nid = j * (nx + 1) + i
                W[i, j] = res.displacements[3 * nid]
        self._xs, self._ys, self._W = xs, ys, W

        # Escala automatica: el maximo |w| se dibuja como 20 % del lado mayor
        w_max_abs = float(np.max(np.abs(W)))
        self._escala_auto = (0.2 * max(a, b) / w_max_abs) if w_max_abs > 0 else 1.0
        self._redraw_surface()

        # ----- resumen -----
        idx_max = np.unravel_index(np.argmax(np.abs(W)), W.shape)
        w_max = float(W[idx_max])
        x_max, y_max = float(xs[idx_max[0]]), float(ys[idx_max[1]])

        # Solucion analitica de Timoshenko (solo placa CUADRADA con q uniforme)
        D_flex = E * t ** 3 / (12.0 * (1.0 - nu * nu))
        alfa = None
        if abs(a - b) < 1e-12:
            alfa = 0.00406 if self.cmb_bc.currentIndex() == 0 else 0.00126
        filas = [
            ("w máximo FEM (m)", _fmt(w_max)),
            ("Posición de w máximo (x, y) m", f"({_fmt(x_max)}, {_fmt(y_max)})"),
        ]
        if alfa is not None:
            w_ref = alfa * q * a ** 4 / D_flex
            err = abs(w_max - w_ref) / abs(w_ref) * 100.0 if w_ref != 0 else 0.0
            filas += [
                ("w analítico (Timoshenko) (m)", _fmt(w_ref)),
                ("Error vs analítico", f"{err:.2f} %"),
            ]
        else:
            filas.append(("w analítico", "— (solo placa cuadrada)"))
        M_max = max(float(np.max(np.abs(er.moments_center)))
                    for er in res.elements)
        t_txt = f"{dt * 1000:.0f} ms" if dt < 1.0 else f"{dt:.2f} s"
        filas += [
            ("Momento máximo |M| centro (N·m/m)", _fmt(M_max)),
            ("Rigidez a flexión D (N·m)", _fmt(D_flex)),
            ("Elementos / GDL", f"{len(s.elements)} / {s.n_dofs}"),
            ("Tiempo de cálculo", t_txt),
        ]
        model = QStandardItemModel(len(filas), 2)
        model.setHorizontalHeaderLabels(["Magnitud", "Valor"])
        for r, (k, v) in enumerate(filas):
            it_k = QStandardItem(k)
            it_v = QStandardItem(v)
            it_k.setEditable(False)
            it_v.setEditable(False)
            model.setItem(r, 0, it_k)
            model.setItem(r, 1, it_v)
        self.tbl_resumen.setModel(model)

        self._update_matrix_view()
        self.lbl_status.setText(
            f"Cálculo completado en {t_txt} — "
            f"{len(s.elements)} elementos, {s.n_dofs} GDL."
        )

    # ------------------------------------------------------------ helpers
    @staticmethod
    def _make_mesh(a: float, b: float, nx: int, ny: int,
                   E: float, nu: float, t: float) -> StructurePlate:
        """Malla nx×ny de elementos plate, nodos CCW desde inferior-izquierda."""
        s = StructurePlate()
        dx, dy = a / nx, b / ny
        node_at: dict[tuple[int, int], NodePlate] = {}
        nid = 0
        for j in range(ny + 1):
            for i in range(nx + 1):
                n = NodePlate(id=nid, x=i * dx, y=j * dy)
                s.add_node(n)
                node_at[(i, j)] = n
                nid += 1
        eid = 0
        for j in range(ny):
            for i in range(nx):
                s.add_element(PlateElement(
                    id=eid,
                    nodes=[node_at[(i, j)], node_at[(i + 1, j)],
                           node_at[(i + 1, j + 1)], node_at[(i, j + 1)]],
                    E=E, nu=nu, t=t))
                eid += 1
        return s

    @staticmethod
    def _apply_bc(s: StructurePlate, a: float, b: float, tipo: int) -> None:
        """Aplica el apoyo elegido en los 4 bordes.

        tipo 0 = simplemente apoyada 'dura' (w = 0; el giro alrededor del
                 eje normal al borde tambien es 0 porque w = 0 a lo largo
                 del borde recto).
        tipo 1 = empotrada (w = θx = θy = 0).
        """
        tol = 1e-9
        for n in s.nodes:
            en_x0 = abs(n.x - 0.0) < tol
            en_xa = abs(n.x - a) < tol
            en_y0 = abs(n.y - 0.0) < tol
            en_yb = abs(n.y - b) < tol
            if not (en_x0 or en_xa or en_y0 or en_yb):
                continue
            n.restraint_w = True
            if tipo == 1:
                n.restraint_rx = True
                n.restraint_ry = True
            else:
                if en_x0 or en_xa:
                    n.restraint_rx = True
                if en_y0 or en_yb:
                    n.restraint_ry = True

    def _redraw_surface(self) -> None:
        """Redibuja la superficie con la escala del slider (en % de la auto)."""
        if self._W is None:
            return
        factor = self.sl_escala.value() / 100.0
        escala = self._escala_auto * factor
        self.lbl_escala.setText(f"x {escala:.0f}")
        self.canvas.show_solution(self._xs, self._ys, self._W, escala)

    def _update_matrix_view(self) -> None:
        """Muestra la matriz didáctica elegida del primer elemento."""
        if self._resultado is None or not getattr(self, "_structure", None):
            return
        el = self._structure.elements[0]
        idx = self.cmb_matriz.currentIndex()
        if idx == 0:
            arr = el.C_matrix()
        elif idx == 1:
            arr = el.stiffness_matrix()
        else:
            xs = [n.x for n in el.nodes]
            ys = [n.y for n in el.nodes]
            from ..fem.plate_element import Q_matrix
            arr = Q_matrix((min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0)
        self.tbl_matriz.setModel(_matrix_model(arr))
        self.tbl_matriz.resizeColumnsToContents()
