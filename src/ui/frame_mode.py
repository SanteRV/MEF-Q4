"""Modo PÓRTICO (frame 3D) — pestaña autocontenida del aplicativo.

Para qué sirve: es la interfaz del modelo FRAME tridimensional (cap. 01.02
del documento teórico). El usuario elige una tipología, sus dimensiones, la
sección y las cargas; el programa resuelve con el elemento de 12 GDL y
muestra:

    - la estructura y su deformada en una vista 3D,
    - los diagramas de fuerzas internas del elemento seleccionado
      (axial, cortantes, torsor y flectores), construidos con la
      superposición S(x) = S_nodos(x) + Σ S_carga del cap. 01.02.01.010,
    - tablas de desplazamientos, reacciones y fuerzas de extremo,
    - las matrices didácticas: k en el SCL, T (ec. 2.3.1) y K = Tᵀ·k·T.

Arquitectura: familia paralela a los modos plane, placa y lámina.
"""
from __future__ import annotations
import time
from typing import Optional

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtGui import (
    QColor, QVector3D, QFont, QCursor, QStandardItemModel, QStandardItem,
)
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QFormLayout, QGroupBox, QLineEdit,
    QComboBox, QPushButton, QLabel, QSlider, QTabWidget, QTableView,
    QMessageBox, QApplication, QHeaderView, QSplitter,
)
import pyqtgraph.opengl as gl

from ..fem.frame_element import FrameElement, FrameLoad
from ..fem.node_frame import NodeFrame
from ..fem.solver_frame import solve_frame, FEMResultFrame
from ..fem.structure_frame import StructureFrame


# ---------------------------------------------------------------- helpers
def _fmt(v: float) -> str:
    if abs(v) < 1e-13:
        return "0"
    return f"{v:.15g}"


def _table_model(headers: list[str], filas: list[list[str]]) -> QStandardItemModel:
    model = QStandardItemModel(len(filas), len(headers))
    model.setHorizontalHeaderLabels(headers)
    for r, fila in enumerate(filas):
        for c, txt in enumerate(fila):
            it = QStandardItem(txt)
            it.setEditable(False)
            if c > 0:
                it.setTextAlignment(Qt.AlignmentFlag.AlignRight
                                    | Qt.AlignmentFlag.AlignVCenter)
            model.setItem(r, c, it)
    return model


def _matrix_model(arr: np.ndarray) -> QStandardItemModel:
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
class FrameCanvas3D(gl.GLViewWidget):
    """Vista 3D de la estructura y su deformada."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setBackgroundColor(QColor("#FAFAFC"))
        self.setCameraPosition(distance=12.0, elevation=22, azimuth=-70)
        self._dyn_items: list = []

        grid = gl.GLGridItem()
        grid.setSize(x=12, y=12)
        grid.setSpacing(x=1, y=1)
        grid.setColor((180, 180, 188, 120))
        self.addItem(grid)
        font = QFont("Arial", 11)
        font.setBold(True)
        for end, color, rgba, lbl in [
            ((1.5, 0, 0), (0.90, 0.18, 0.18, 1), (230, 46, 46, 255), "X"),
            ((0, 1.5, 0), (0.18, 0.70, 0.18, 1), (40, 165, 40, 255), "Y"),
            ((0, 0, 1.5), (0.18, 0.32, 0.92, 1), (46, 82, 235, 255), "Z"),
        ]:
            ln = gl.GLLinePlotItem(
                pos=np.array([[0, 0, 0], list(end)], dtype=np.float64),
                color=color, width=2.5, antialias=True)
            ln.setGLOptions("opaque")
            self.addItem(ln)
            try:
                self.addItem(gl.GLTextItem(
                    pos=(end[0] * 1.12, end[1] * 1.12, end[2] * 1.12),
                    text=lbl, color=rgba, font=font))
            except Exception:
                pass

    def show_structure(self, s: StructureFrame, u: Optional[np.ndarray],
                       scale: float) -> None:
        """Dibuja la geometría original y, si hay solución, la deformada."""
        for it in self._dyn_items:
            try:
                self.removeItem(it)
            except Exception:
                pass
        self._dyn_items.clear()

        segs = []
        for el in s.elements:
            for n in el.nodes:
                segs.append([n.x, n.y, n.z])
        if segs:
            ln = gl.GLLinePlotItem(pos=np.array(segs),
                                   color=(0.42, 0.45, 0.50, 0.85),
                                   width=2.0, antialias=True, mode="lines")
            self.addItem(ln)
            self._dyn_items.append(ln)

        nodos = np.array([[n.x, n.y, n.z] for n in s.nodes]) if s.nodes else None
        if nodos is not None and len(nodos):
            pts = gl.GLScatterPlotItem(pos=nodos, color=(0.25, 0.28, 0.35, 1.0),
                                       size=8.0, pxMode=True)
            self.addItem(pts)
            self._dyn_items.append(pts)

        if u is not None:
            # La deformada se dibuja con varias estaciones por barra usando
            # la interpolación lineal entre nodos: basta para leer la forma.
            segs_d = []
            for el in s.elements:
                p = []
                for n in el.nodes:
                    d = u[list(n.dofs)][:3] * scale
                    p.append([n.x + d[0], n.y + d[1], n.z + d[2]])
                segs_d.extend(p)
            ln_d = gl.GLLinePlotItem(pos=np.array(segs_d),
                                     color=(0.85, 0.20, 0.20, 1.0),
                                     width=3.0, antialias=True, mode="lines")
            self.addItem(ln_d)
            self._dyn_items.append(ln_d)

        if nodos is not None and len(nodos):
            centro = nodos.mean(axis=0)
            tam = float(np.max(nodos.max(axis=0) - nodos.min(axis=0))) or 1.0
            self.opts["center"] = QVector3D(*centro)
            self.setCameraPosition(distance=tam * 2.4, elevation=22,
                                   azimuth=-70)
        self.update()


# ---------------------------------------------------------------- widget
class FrameModeWidget(QWidget):
    """Pestaña "Pórtico (frame 3D)": entradas, solución, tablas y diagramas."""

    TIPOLOGIAS = [
        "Viga simplemente apoyada",
        "Voladizo",
        "Viga continua de 2 tramos",
        "Pórtico plano (1 vano, 1 nivel)",
        "Pórtico plano (2 vanos, 2 niveles)",
        "Pórtico 3D (1 vano en X e Y)",
    ]

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._resultado: Optional[FEMResultFrame] = None
        self._structure: Optional[StructureFrame] = None
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

        gb_geo = QGroupBox("Estructura (frame 3D, 6 GDL por nodo)")
        fg = QFormLayout(gb_geo)
        self.cmb_tipo = QComboBox()
        self.cmb_tipo.addItems(self.TIPOLOGIAS)
        self.cmb_tipo.currentIndexChanged.connect(self._on_tipo_changed)
        fg.addRow("Tipología:", self.cmb_tipo)
        self.ed_L = QLineEdit("6.0")
        self.ed_H = QLineEdit("3.0")
        fg.addRow("Luz L (m):", self.ed_L)
        self.row_H = self.ed_H
        fg.addRow("Altura H (m):", self.ed_H)
        col.addWidget(gb_geo)

        gb_sec = QGroupBox("Material y sección")
        fs = QFormLayout(gb_sec)
        self.ed_E = QLineEdit("2.1e11")
        self.ed_nu = QLineEdit("0.3")
        self.ed_A = QLineEdit("0.01")
        self.ed_Iy = QLineEdit("8.333e-6")
        self.ed_Iz = QLineEdit("8.333e-6")
        self.ed_J = QLineEdit("1.6e-5")
        fs.addRow("E (Pa):", self.ed_E)
        fs.addRow("ν (para G = E/[2(1+ν)]):", self.ed_nu)
        fs.addRow("A (m²):", self.ed_A)
        fs.addRow("Iy (m⁴):", self.ed_Iy)
        fs.addRow("Iz (m⁴):", self.ed_Iz)
        fs.addRow("J (m⁴):", self.ed_J)
        col.addWidget(gb_sec)

        gb_car = QGroupBox("Cargas")
        fc = QFormLayout(gb_car)
        self.ed_w = QLineEdit("-10000.0")
        self.ed_P = QLineEdit("0.0")
        self.ed_Fh = QLineEdit("0.0")
        fc.addRow("w distribuida en vigas (N/m):", self.ed_w)
        fc.addRow("P puntual en centro de vano (N):", self.ed_P)
        fc.addRow("Fh horizontal en cada nivel (N):", self.ed_Fh)
        self.cmb_apoyo = QComboBox()
        self.cmb_apoyo.addItems(["Empotrado", "Articulado"])
        fc.addRow("Apoyo en la base:", self.cmb_apoyo)
        col.addWidget(gb_car)

        self.btn_calc = QPushButton("Calcular pórtico")
        self.btn_calc.setProperty("primary", True)
        self.btn_calc.clicked.connect(self.calculate)
        col.addWidget(self.btn_calc)

        self.lbl_status = QLabel(
            "Elija una tipología y presione Calcular. Cada nodo aporta 6 GDL: "
            "3 desplazamientos y 3 giros."
        )
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setStyleSheet("color: #555; font-size: 11px;")
        col.addWidget(self.lbl_status)

        self.tabs_res = QTabWidget()
        self.tbl_resumen = QTableView()
        self.tbl_desp = QTableView()
        self.tbl_reac = QTableView()
        self.tbl_fuerzas = QTableView()
        for t, nombre in [(self.tbl_resumen, "Resumen"),
                          (self.tbl_desp, "Desplazamientos"),
                          (self.tbl_reac, "Reacciones"),
                          (self.tbl_fuerzas, "Fuerzas de extremo")]:
            t.horizontalHeader().setSectionResizeMode(
                QHeaderView.ResizeMode.ResizeToContents)
            self.tabs_res.addTab(t, nombre)

        mat_tab = QWidget()
        mv = QVBoxLayout(mat_tab)
        self.cmb_matriz = QComboBox()
        self.cmb_matriz.addItems([
            "k (12×12) en el SCL del elemento seleccionado",
            "T (12×12) matriz de transformación — ec. 2.3.1",
            "K = Tᵀ·k·T (12×12) en el SCG — ec. 2.4.2",
            "Q_f (12×1) cargas de fijación — 01.02.01.07",
        ])
        self.cmb_matriz.currentIndexChanged.connect(self._update_matrix_view)
        mv.addWidget(self.cmb_matriz)
        self.tbl_matriz = QTableView()
        mv.addWidget(self.tbl_matriz, 1)
        self.tabs_res.addTab(mat_tab, "Matrices")
        col.addWidget(self.tabs_res, 1)

        root.addWidget(left, 0)

        # ===== Panel derecho: vista 3D arriba, diagramas abajo =====
        right = QSplitter(Qt.Orientation.Vertical)

        arriba = QWidget()
        rcol = QVBoxLayout(arriba)
        rcol.setContentsMargins(0, 0, 0, 0)
        self.canvas = FrameCanvas3D()
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
        right.addWidget(arriba)

        abajo = QWidget()
        dcol = QVBoxLayout(abajo)
        dcol.setContentsMargins(0, 0, 0, 0)
        fila2 = QHBoxLayout()
        fila2.addWidget(QLabel("Diagramas del elemento:"))
        self.cmb_elem = QComboBox()
        self.cmb_elem.currentIndexChanged.connect(self._update_diagrams)
        fila2.addWidget(self.cmb_elem)
        fila2.addWidget(QLabel("Componente:"))
        self.cmb_comp = QComboBox()
        self.cmb_comp.addItems([
            "Mz — flector en el plano local x-y",
            "My — flector en el plano local x-z",
            "Vy — cortante local y",
            "Vz — cortante local z",
            "P — axial",
            "T — torsor",
        ])
        self.cmb_comp.currentIndexChanged.connect(self._update_diagrams)
        fila2.addWidget(self.cmb_comp)
        fila2.addStretch()
        dcol.addLayout(fila2)
        self.figure = Figure(figsize=(6, 2.6), tight_layout=True)
        self.mpl = FigureCanvasQTAgg(self.figure)
        dcol.addWidget(self.mpl, 1)
        right.addWidget(abajo)
        right.setSizes([500, 260])

        root.addWidget(right, 1)
        self._on_tipo_changed()

    def _on_tipo_changed(self) -> None:
        """La altura solo tiene sentido en las tipologías con columnas."""
        con_altura = self.cmb_tipo.currentIndex() >= 3
        self.ed_H.setEnabled(con_altura)

    # ------------------------------------------------------------ cálculo
    def _leer_float(self, ed: QLineEdit, nombre: str) -> float:
        try:
            return float(ed.text().strip())
        except ValueError:
            raise ValueError(f"Valor inválido en '{nombre}': {ed.text()!r}")

    def calculate(self) -> None:
        try:
            L = self._leer_float(self.ed_L, "Luz L")
            H = self._leer_float(self.ed_H, "Altura H")
            E = self._leer_float(self.ed_E, "E")
            nu = self._leer_float(self.ed_nu, "ν")
            A = self._leer_float(self.ed_A, "A")
            Iy = self._leer_float(self.ed_Iy, "Iy")
            Iz = self._leer_float(self.ed_Iz, "Iz")
            Jt = self._leer_float(self.ed_J, "J")
            w = self._leer_float(self.ed_w, "w")
            P = self._leer_float(self.ed_P, "P")
            Fh = self._leer_float(self.ed_Fh, "Fh")
            if min(L, H, E, A, Iy, Iz, Jt) <= 0:
                raise ValueError("L, H, E, A, Iy, Iz y J deben ser > 0.")
            if not (-1.0 < nu < 0.5):
                raise ValueError("ν debe estar en (−1, 0.5).")
        except ValueError as e:
            QMessageBox.critical(self, "Datos inválidos", str(e))
            return

        self.lbl_status.setText("Calculando...")
        QApplication.setOverrideCursor(QCursor(Qt.CursorShape.WaitCursor))
        QApplication.processEvents()
        t0 = time.perf_counter()
        try:
            s = self._build(self.cmb_tipo.currentIndex(), L, H,
                            E, nu, A, Iy, Iz, Jt, w, P, Fh,
                            self.cmb_apoyo.currentIndex() == 0)
            res = solve_frame(s)
        except Exception as e:
            self.lbl_status.setText("Error en el cálculo.")
            QMessageBox.critical(self, "Error en el cálculo", str(e))
            return
        finally:
            QApplication.restoreOverrideCursor()
        dt = time.perf_counter() - t0

        self._structure, self._resultado = s, res

        d_max = max(
            float(np.max(np.abs(res.displacements[list(n.dofs)][:3])))
            for n in s.nodes) if s.nodes else 0.0
        tam = 1.0
        if s.nodes:
            P3 = np.array([[n.x, n.y, n.z] for n in s.nodes])
            tam = float(np.max(P3.max(axis=0) - P3.min(axis=0))) or 1.0
        self._escala_auto = (0.15 * tam / d_max) if d_max > 0 else 1.0

        self.cmb_elem.blockSignals(True)
        self.cmb_elem.clear()
        self.cmb_elem.addItems([f"Elemento {el.id + 1} "
                                f"(N{el.nodes[0].id + 1}–N{el.nodes[1].id + 1})"
                                for el in s.elements])
        self.cmb_elem.blockSignals(False)

        self._redraw()
        self._fill_tables(dt)
        self._update_matrix_view()
        self._update_diagrams()

        t_txt = f"{dt * 1000:.0f} ms" if dt < 1.0 else f"{dt:.2f} s"
        self.lbl_status.setText(
            f"Cálculo completado en {t_txt} — {len(s.elements)} elementos, "
            f"{len(s.nodes)} nodos, {s.n_dofs} GDL.")

    # ------------------------------------------------------------ modelos
    @staticmethod
    def _build(tipo: int, L: float, H: float, E: float, nu: float,
               A: float, Iy: float, Iz: float, Jt: float,
               w: float, P: float, Fh: float,
               empotrado: bool) -> StructureFrame:
        """Arma una de las tipologías con sus apoyos y cargas.

        Los modelos planos se resuelven en el plano XZ (Z vertical), que es
        el criterio de ejes del resto del aplicativo; los GDL que salen de
        ese plano se restringen para que K_ff no quede singular.
        """
        G = E / (2.0 * (1.0 + nu))
        s = StructureFrame()

        def nodo(x, y, z) -> NodeFrame:
            n = NodeFrame(len(s.nodes), x, y, z)
            s.add_node(n)
            return n

        def barra(ni, nj, cargas=None) -> FrameElement:
            el = FrameElement(len(s.elements), [ni, nj], E=E, G=G, A=A,
                              Iy=Iy, Iz=Iz, J=Jt, loads=list(cargas or []))
            s.add_element(el)
            return el

        # Carga transversal sobre las vigas: para una barra horizontal el
        # eje local z apunta hacia +Z, así que la carga vertical va en "z".
        def cargas_viga(largo):
            c = []
            if w != 0.0:
                c.append(FrameLoad("distribuida", w, eje="z"))
            if P != 0.0:
                c.append(FrameLoad("puntual", P, a=largo / 2.0, eje="z"))
            return c

        if tipo == 0:       # viga simplemente apoyada (2 tramos para ver el centro)
            n0 = nodo(0, 0, 0)
            n1 = nodo(L / 2, 0, 0)
            n2 = nodo(L, 0, 0)
            barra(n0, n1, cargas_viga(L / 2))
            barra(n1, n2, cargas_viga(L / 2))
            n0.articular()
            n2.restraint_uy = n2.restraint_uz = True
            n0.restraint_rx = n2.restraint_rx = True
        elif tipo == 1:     # voladizo
            n0 = nodo(0, 0, 0)
            n1 = nodo(L, 0, 0)
            barra(n0, n1, cargas_viga(L))
            n0.empotrar()
        elif tipo == 2:     # viga continua de 2 tramos
            n0 = nodo(0, 0, 0)
            n1 = nodo(L, 0, 0)
            n2 = nodo(2 * L, 0, 0)
            barra(n0, n1, cargas_viga(L))
            barra(n1, n2, cargas_viga(L))
            for n in (n0, n1, n2):
                n.restraint_uy = n.restraint_uz = n.restraint_rx = True
            n0.restraint_ux = True
        elif tipo == 3:     # pórtico plano de 1 vano y 1 nivel
            b0 = nodo(0, 0, 0)
            b1 = nodo(L, 0, 0)
            t0 = nodo(0, 0, H)
            t1 = nodo(L, 0, H)
            barra(b0, t0)
            barra(t0, t1, cargas_viga(L))
            barra(b1, t1)
            for n in (b0, b1):
                n.empotrar() if empotrado else n.articular()
            t0.load_fx += Fh
        elif tipo == 4:     # pórtico plano de 2 vanos y 2 niveles
            cols = [0.0, L, 2 * L]
            niveles = [0.0, H, 2 * H]
            grid = {}
            for j, z in enumerate(niveles):
                for i, x in enumerate(cols):
                    grid[(i, j)] = nodo(x, 0, z)
            for i in range(len(cols)):
                for j in range(len(niveles) - 1):
                    barra(grid[(i, j)], grid[(i, j + 1)])
            for j in range(1, len(niveles)):
                for i in range(len(cols) - 1):
                    barra(grid[(i, j)], grid[(i + 1, j)], cargas_viga(L))
            for i in range(len(cols)):
                n = grid[(i, 0)]
                n.empotrar() if empotrado else n.articular()
            for j in range(1, len(niveles)):
                grid[(0, j)].load_fx += Fh
        else:               # pórtico 3D de 1 vano en X e Y
            base, techo = {}, {}
            for i in (0, 1):
                for j in (0, 1):
                    base[(i, j)] = nodo(i * L, j * L, 0)
            for i in (0, 1):
                for j in (0, 1):
                    techo[(i, j)] = nodo(i * L, j * L, H)
            for k in base:
                barra(base[k], techo[k])
            for j in (0, 1):
                barra(techo[(0, j)], techo[(1, j)], cargas_viga(L))
            for i in (0, 1):
                barra(techo[(i, 0)], techo[(i, 1)], cargas_viga(L))
            for n in base.values():
                n.empotrar() if empotrado else n.articular()
            techo[(0, 0)].load_fx += Fh

        # Los modelos planos viven en XZ: se bloquea lo que sale del plano
        if tipo <= 4:
            for n in s.nodes:
                n.restraint_uy = True
                n.restraint_rx = True
                n.restraint_rz = True
        return s

    # ------------------------------------------------------------ salidas
    def _fill_tables(self, dt: float) -> None:
        s, res = self._structure, self._resultado
        if s is None or res is None:
            return

        d_max, nodo_max = 0.0, None
        for n in s.nodes:
            d = float(np.max(np.abs(res.displacements[list(n.dofs)][:3])))
            if d > d_max:
                d_max, nodo_max = d, n
        M_max = max((er.moment_z_max for er in res.elements), default=0.0)
        My_max = max((er.moment_y_max for er in res.elements), default=0.0)
        V_max = max((max(er.shear_y_max, er.shear_z_max)
                     for er in res.elements), default=0.0)
        N_max = max((er.axial_max for er in res.elements), default=0.0)
        T_max = max((er.torsion_max for er in res.elements), default=0.0)
        t_txt = f"{dt * 1000:.0f} ms" if dt < 1.0 else f"{dt:.2f} s"
        self.tbl_resumen.setModel(_table_model(
            ["Magnitud", "Valor"],
            [
                ["Desplazamiento máximo (m)", _fmt(d_max)],
                ["Nodo del máximo",
                 f"N{nodo_max.id + 1}" if nodo_max else "—"],
                ["|Mz| máximo (N·m)", _fmt(M_max)],
                ["|My| máximo (N·m)", _fmt(My_max)],
                ["|V| máximo (N)", _fmt(V_max)],
                ["|N| axial máximo (N)", _fmt(N_max)],
                ["|T| torsor máximo (N·m)", _fmt(T_max)],
                ["Elementos / nodos / GDL",
                 f"{len(s.elements)} / {len(s.nodes)} / {s.n_dofs}"],
                ["Tiempo de cálculo", t_txt],
            ]))

        self.tbl_desp.setModel(_table_model(
            ["Nodo", "ux (m)", "uy (m)", "uz (m)",
             "rx (rad)", "ry (rad)", "rz (rad)"],
            [[f"N{n.id + 1}"] + [_fmt(v) for v in res.displacements[list(n.dofs)]]
             for n in s.nodes]))

        filas_r = []
        for n in s.nodes:
            if not any(n.restraints):
                continue
            filas_r.append([f"N{n.id + 1}"]
                           + [_fmt(v) for v in res.reactions[list(n.dofs)]])
        self.tbl_reac.setModel(_table_model(
            ["Nodo", "Rx (N)", "Ry (N)", "Rz (N)",
             "Mx (N·m)", "My (N·m)", "Mz (N·m)"], filas_r))

        etiquetas = ["N", "Vy", "Vz", "T", "My", "Mz"]
        filas_f = []
        for el, er in zip(s.elements, res.elements):
            Q = er.end_forces
            filas_f.append([f"E{el.id + 1} — nodo i"]
                           + [_fmt(v) for v in Q[:6]])
            filas_f.append([f"E{el.id + 1} — nodo j"]
                           + [_fmt(v) for v in Q[6:]])
        self.tbl_fuerzas.setModel(_table_model(
            ["Elemento / extremo"] + etiquetas, filas_f))

    def _redraw(self) -> None:
        if self._structure is None:
            return
        escala = self._escala_auto * (self.sl_escala.value() / 100.0)
        self.lbl_escala.setText(f"x {escala:.0f}")
        u = self._resultado.displacements if self._resultado else None
        self.canvas.show_structure(self._structure, u, escala)

    def _elemento_actual(self) -> Optional[FrameElement]:
        if self._structure is None:
            return None
        i = self.cmb_elem.currentIndex()
        if 0 <= i < len(self._structure.elements):
            return self._structure.elements[i]
        return None

    def _update_matrix_view(self) -> None:
        el = self._elemento_actual()
        if el is None:
            return
        idx = self.cmb_matriz.currentIndex()
        if idx == 0:
            arr = el.stiffness_local()
        elif idx == 1:
            arr = el.transformation_matrix()
        elif idx == 2:
            arr = el.stiffness_matrix()
        else:
            arr = el.fixed_end_forces().reshape(-1, 1)
        self.tbl_matriz.setModel(_matrix_model(arr))
        self.tbl_matriz.resizeColumnsToContents()

    def _update_diagrams(self) -> None:
        el = self._elemento_actual()
        self.figure.clear()
        if el is None or self._resultado is None:
            self.mpl.draw()
            return
        comp = ["Mz", "My", "Vy", "Vz", "P", "T"][self.cmb_comp.currentIndex()]
        unidad = "N·m" if comp in ("Mz", "My", "T") else "N"
        v_e = self._resultado.displacements[el.global_dofs()]
        x, vals = el.diagram(comp, v_e)

        ax = self.figure.add_subplot(111)
        ax.plot(x, vals, "-", color="#1F4E78", lw=2)
        ax.fill_between(x, vals, 0, color="#1F4E78", alpha=0.15)
        ax.axhline(0, color="#333", lw=1)
        ax.set_xlabel("x a lo largo del elemento (m)")
        ax.set_ylabel(f"{comp} ({unidad})")
        ax.set_title(f"Elemento {el.id + 1} — diagrama de {comp}")
        ax.grid(True, alpha=0.3)
        self.mpl.draw()

        self._update_matrix_view()
