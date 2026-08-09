"""Diálogo de convergencia MEF — cap. 01.01.08 del documento teórico.

Tiene dos pestañas, una por cada mitad del capítulo:

    Refinamiento de malla (01.01.08.01): resuelve el mismo problema con
    mallas progresivamente más finas y compara σ_max, u_max y el error
    relativo en función del número de GDL.

    Criterios de la formulación (01.01.08.02 a 01.01.08.05): cuerpo rígido,
    deformación constante, autovalores de K^e y prueba del parche. Son
    propiedades del elemento, no del modelo dibujado, y por eso se ejecutan
    sobre elementos y parches de referencia.
"""
from __future__ import annotations
import time
from copy import deepcopy

import numpy as np
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QSpinBox,
    QTableView, QHeaderView, QGroupBox, QMessageBox, QFileDialog, QDialogButtonBox,
    QFormLayout, QCheckBox, QLineEdit, QTabWidget, QWidget,
)
from PySide6.QtGui import QStandardItemModel, QStandardItem, QColor

from ..fem.node import Node
from ..fem.q4_element import Q4Element
from ..fem.structure import Structure
from ..fem.solver import solve
from ..fem import verification


def _generate_rect_mesh(xmin, xmax, ymin, ymax, nx, ny,
                        E, nu, t, plane_stress) -> Structure:
    """Misma lógica que CanvasEditor — generar Nx*Ny elementos Q4."""
    s = Structure()
    dx = (xmax - xmin) / nx
    dy = (ymax - ymin) / ny
    node_at: dict[tuple[int, int], Node] = {}
    node_id = 0
    for j in range(ny + 1):
        for i in range(nx + 1):
            n = Node(id=node_id, x=xmin + i * dx, y=ymin + j * dy)
            s.add_node(n)
            node_at[(i, j)] = n
            node_id += 1
    el_id = 0
    for j in range(ny):
        for i in range(nx):
            n1 = node_at[(i,     j)]       # (--)
            n2 = node_at[(i + 1, j)]       # (+-)
            n3 = node_at[(i + 1, j + 1)]   # (++)
            n4 = node_at[(i,     j + 1)]   # (-+)
            s.add_element(Q4Element(
                id=el_id, nodes=[n1, n2, n3, n4],
                E=E, nu=nu, t=t, plane_stress=plane_stress,
            ))
            el_id += 1
    return s


def _apply_boundary_left_right(s: Structure, xmin: float, xmax: float,
                               total_fx: float, total_fy: float) -> None:
    """Restringe el borde izquierdo y aplica carga distribuida en el borde derecho."""
    left = [n for n in s.nodes if abs(n.x - xmin) < 1e-9]
    right = [n for n in s.nodes if abs(n.x - xmax) < 1e-9]
    for n in left:
        n.restraint_x = True
        n.restraint_y = True
    for n in right:
        n.load_x = total_fx / len(right)
        n.load_y = total_fy / len(right)


class ConvergenceDialog(QDialog):
    """Diálogo que ejecuta el estudio de convergencia."""

    def __init__(self, parent=None, base_structure: Structure | None = None):
        super().__init__(parent)
        self.setWindowTitle("Estudio de convergencia MEF")
        self.resize(1100, 700)

        # Propiedades a usar (tomadas del proyecto actual si existe)
        if base_structure and base_structure.elements:
            el0 = base_structure.elements[0]
            self.E = el0.E
            self.nu = el0.nu
            self.t = el0.t
            self.plane_stress = el0.plane_stress
        else:
            self.E, self.nu, self.t = 2173706.0, 0.15, 0.1
            self.plane_stress = True

        # Geometría base (placa cuadrada por defecto, igual al demo: 1×1)
        self.xmin, self.xmax = -0.5, 0.5
        self.ymin, self.ymax = -0.5, 0.5
        self.total_fx, self.total_fy = 100.0, 100.0

        # Niveles de refinamiento por defecto
        self.refinements = [1, 2, 4, 8, 16]

        self.df_results: pd.DataFrame | None = None

        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        self.tabs = QTabWidget()
        root.addWidget(self.tabs, 1)
        self.tabs.addTab(self._build_mesh_tab(), "Refinamiento de malla")
        self.tabs.addTab(self._build_criteria_tab(),
                         "Criterios de la formulación")
        bb = QDialogButtonBox(QDialogButtonBox.Close)
        bb.rejected.connect(self.reject)
        root.addWidget(bb)

    # ------------------------------------------------ pestaña 1: malla
    def _build_mesh_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        # ===== Parámetros de la corrida =====
        params = QGroupBox("Configuración del estudio (cap. 01.01.08.01)")
        form = QFormLayout(params)

        # Dimensión de la placa
        self.x_input = QLineEdit(f"{self.xmin:g} ; {self.xmax:g}")
        self.y_input = QLineEdit(f"{self.ymin:g} ; {self.ymax:g}")
        self.fx_input = QLineEdit(f"{self.total_fx:g}")
        self.fy_input = QLineEdit(f"{self.total_fy:g}")
        form.addRow("Rango X (xmin ; xmax) [m]:", self.x_input)
        form.addRow("Rango Y (ymin ; ymax) [m]:", self.y_input)
        form.addRow("Carga TOTAL en borde derecho Fx [N]:", self.fx_input)
        form.addRow("Carga TOTAL en borde derecho Fy [N]:", self.fy_input)

        # Niveles
        self.refinements_input = QLineEdit(",".join(str(n) for n in self.refinements))
        form.addRow("Refinamientos (Nx=Ny, separados por coma):", self.refinements_input)

        # Botón ejecutar
        run_row = QHBoxLayout()
        self.btn_run = QPushButton("Ejecutar estudio")
        self.btn_run.setProperty("primary", True)
        self.btn_run.clicked.connect(self._run_study)
        run_row.addWidget(self.btn_run)
        self.btn_export = QPushButton("Exportar resultados (.csv)")
        self.btn_export.clicked.connect(self._export_csv)
        self.btn_export.setEnabled(False)
        run_row.addWidget(self.btn_export)
        run_row.addStretch()

        layout.addWidget(params)
        layout.addLayout(run_row)

        # ===== Resultados: tabla + gráfica =====
        results_split = QHBoxLayout()

        # Tabla
        self.table = QTableView()
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.setMinimumWidth(450)
        results_split.addWidget(self.table)

        # Gráfica
        self.figure = Figure(figsize=(6, 5), tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        results_split.addWidget(self.canvas, 1)

        layout.addLayout(results_split, 1)
        return tab

    # ------------------------------------------ pestaña 2: criterios
    def _build_criteria_tab(self) -> QWidget:
        """Comprobaciones de los capítulos 01.01.08.02 a 01.01.08.05."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        info = QLabel(
            "El documento exige que la formulación cumpla cuatro criterios "
            "antes de confiar en el refinamiento de malla:\n"
            "  • Cuerpo rígido (01.01.08.02): un movimiento rígido no debe "
            "generar deformaciones — K·q = 0.\n"
            "  • Deformación constante (01.01.08.03): un estado uniforme debe "
            "reproducirse exacto en todo el elemento.\n"
            "  • Compatibilidad (01.01.08.04): el plane es conforme; el plate "
            "de 12 GDL es no conforme y por eso depende de la prueba del "
            "parche con malla rectangular.\n"
            "  • Prueba del parche (01.01.08.05): un parche con nodo interior "
            "sometido a un estado constante en el contorno debe reproducirlo "
            "exactamente. Se verifican antes los autovalores de K^e (deben "
            "aparecer 3 modos rígidos y ningún modo espurio).\n"
            "\n"
            "Son propiedades del ELEMENTO, no del modelo dibujado: se ejecutan "
            "sobre un Q4 distorsionado y un plate rectangular de referencia, "
            "usando el material del proyecto."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #444; font-size: 11px;")
        layout.addWidget(info)

        fila = QHBoxLayout()
        self.btn_criterios = QPushButton("Ejecutar comprobaciones")
        self.btn_criterios.setProperty("primary", True)
        self.btn_criterios.clicked.connect(self._run_criteria)
        fila.addWidget(self.btn_criterios)
        self.lbl_criterios = QLabel("Sin ejecutar.")
        fila.addWidget(self.lbl_criterios, 1)
        layout.addLayout(fila)

        self.table_criterios = QTableView()
        self.table_criterios.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents)
        self.table_criterios.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table_criterios, 1)
        return tab

    def _run_criteria(self):
        from PySide6.QtWidgets import QApplication
        from PySide6.QtGui import QCursor
        from PySide6.QtCore import Qt as _Qt

        self.lbl_criterios.setText("Ejecutando...")
        QApplication.setOverrideCursor(QCursor(_Qt.CursorShape.WaitCursor))
        QApplication.processEvents()
        try:
            checks = verification.run_all(
                E=self.E, nu=self.nu, t=self.t, plane_stress=self.plane_stress)
        except Exception as e:
            self.lbl_criterios.setText("Error al ejecutar las comprobaciones.")
            QMessageBox.critical(self, "Error", str(e))
            return
        finally:
            QApplication.restoreOverrideCursor()

        cols = ["Capítulo", "Comprobación", "Resultado", "Error medido",
                "Tolerancia", "Detalle"]
        model = QStandardItemModel(len(checks), len(cols))
        model.setHorizontalHeaderLabels(cols)
        for r, c in enumerate(checks):
            valores = [
                c.criterio, c.nombre, c.estado,
                f"{c.valor:.3e}", f"{c.tolerancia:.3e}", c.detalle,
            ]
            for col, txt in enumerate(valores):
                item = QStandardItem(txt)
                item.setEditable(False)
                if col == 2:
                    item.setForeground(
                        QColor("#1B7F3B") if c.ok else QColor("#B00020"))
                model.setItem(r, col, item)
        self.table_criterios.setModel(model)

        fallidos = [c for c in checks if not c.ok]
        if fallidos:
            self.lbl_criterios.setText(
                f"{len(checks) - len(fallidos)} de {len(checks)} cumplen — "
                f"{len(fallidos)} sin cumplir.")
        else:
            self.lbl_criterios.setText(
                f"Los {len(checks)} criterios del cap. 01.01.08 se cumplen.")

    def _parse_range(self, txt: str) -> tuple[float, float]:
        parts = txt.replace(",", ";").split(";")
        return float(parts[0]), float(parts[1])

    def _run_study(self):
        try:
            self.xmin, self.xmax = self._parse_range(self.x_input.text())
            self.ymin, self.ymax = self._parse_range(self.y_input.text())
            self.total_fx = float(self.fx_input.text())
            self.total_fy = float(self.fy_input.text())
            refs = [int(s.strip()) for s in self.refinements_input.text().split(",") if s.strip()]
            if not refs:
                raise ValueError("Lista de refinamientos vacía")
            self.refinements = refs
        except Exception as e:
            QMessageBox.critical(self, "Error en parámetros", str(e))
            return

        # Feedback visual INTUITIVO: dialogo de progreso real (el estudio
        # tiene pasos medibles — una malla = un paso) + cursor de espera.
        from PySide6.QtWidgets import QApplication, QProgressDialog
        from PySide6.QtGui import QCursor
        from PySide6.QtCore import Qt as _Qt
        n_mallas = len(self.refinements)
        progress = QProgressDialog(
            "Preparando el estudio...", "", 0, n_mallas, self)
        progress.setWindowTitle("Estudio de convergencia")
        progress.setWindowModality(_Qt.WindowModality.WindowModal)
        progress.setCancelButton(None)     # el estudio no se interrumpe a medias
        progress.setMinimumDuration(0)     # mostrarlo de inmediato
        progress.setValue(0)
        QApplication.setOverrideCursor(QCursor(_Qt.CursorShape.WaitCursor))
        QApplication.processEvents()

        try:
            rows = []
            prev_u_max = None
            prev_sx_max = None
            for idx, n in enumerate(self.refinements):
                progress.setLabelText(
                    f"Resolviendo malla {idx + 1} de {n_mallas} ({n}×{n})...")
                QApplication.processEvents()
                t0 = time.perf_counter()
                s = _generate_rect_mesh(self.xmin, self.xmax, self.ymin, self.ymax,
                                        n, n, self.E, self.nu, self.t, self.plane_stress)
                _apply_boundary_left_right(s, self.xmin, self.xmax,
                                           self.total_fx, self.total_fy)
                res = solve(s)
                dt = time.perf_counter() - t0
                u_max = float(np.max(np.abs(res.displacements)))
                sx_vals = [sig[0] for el in res.elements for sig in el.stresses_at_gauss]
                sx_max = float(max(sx_vals)) if sx_vals else 0.0
                n_dof = 2 * len(s.nodes)
                n_el = len(s.elements)
                rel_err_u = abs(u_max - prev_u_max) / abs(prev_u_max) if prev_u_max else None
                rel_err_s = abs(sx_max - prev_sx_max) / abs(prev_sx_max) if prev_sx_max else None
                rows.append({
                    "Malla": f"{n}×{n}",
                    "Elementos": n_el,
                    "GDL": n_dof,
                    "u_max (m)": u_max,
                    "σx_max (Pa)": sx_max,
                    "Δ u rel (vs anterior)": rel_err_u if rel_err_u is not None else float("nan"),
                    "Δ σx rel (vs anterior)": rel_err_s if rel_err_s is not None else float("nan"),
                    "tiempo (s)": dt,
                })
                prev_u_max, prev_sx_max = u_max, sx_max
                progress.setValue(idx + 1)   # malla terminada -> avanza la barra

            self.df_results = pd.DataFrame(rows)
            self._update_table()
            self._update_plot()
            self.btn_export.setEnabled(True)
        finally:
            # Cerrar el progreso y restaurar el cursor aunque algo falle
            progress.close()
            QApplication.restoreOverrideCursor()

    def _update_table(self):
        df = self.df_results
        model = QStandardItemModel(df.shape[0], df.shape[1])
        model.setHorizontalHeaderLabels([str(c) for c in df.columns])
        for r in range(df.shape[0]):
            for c in range(df.shape[1]):
                v = df.iat[r, c]
                if isinstance(v, float) and not np.isnan(v):
                    txt = f"{v:.6g}"
                elif isinstance(v, float) and np.isnan(v):
                    txt = "—"
                else:
                    txt = str(v)
                item = QStandardItem(txt)
                item.setEditable(False)
                item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                model.setItem(r, c, item)
        self.table.setModel(model)

    def _update_plot(self):
        df = self.df_results
        self.figure.clear()
        ax1 = self.figure.add_subplot(211)
        ax2 = self.figure.add_subplot(212)

        ax1.plot(df["GDL"], df["u_max (m)"], "o-", color="#1F4E78", lw=2)
        ax1.set_xscale("log")
        ax1.set_xlabel("Número de GDL")
        ax1.set_ylabel("u_max (m)")
        ax1.set_title("Convergencia de desplazamiento máximo")
        ax1.grid(True, alpha=0.3)

        ax2.plot(df["GDL"], df["σx_max (Pa)"], "s-", color="#D62728", lw=2)
        ax2.set_xscale("log")
        ax2.set_xlabel("Número de GDL")
        ax2.set_ylabel("σx_max (Pa)")
        ax2.set_title("Convergencia de esfuerzo máximo (σx en GPs)")
        ax2.grid(True, alpha=0.3)
        self.canvas.draw()

    def _export_csv(self):
        if self.df_results is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Exportar resultados", "convergencia.csv", "CSV (*.csv)"
        )
        if path:
            self.df_results.to_csv(path, index=False, encoding="utf-8")
            QMessageBox.information(self, "Exportado",
                                    f"Resultados guardados en:\n{path}")
