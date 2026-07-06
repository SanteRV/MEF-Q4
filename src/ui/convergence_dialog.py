"""Diálogo de estudio de convergencia MEF.

Resuelve el mismo problema con mallas progresivamente más finas y compara
los resultados (σ_max, u_max, error relativo) en función del número de GDL.
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
    QFormLayout, QCheckBox, QLineEdit,
)
from PySide6.QtGui import QStandardItemModel, QStandardItem

from ..fem.node import Node
from ..fem.q4_element import Q4Element
from ..fem.structure import Structure
from ..fem.solver import solve


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
            n1 = node_at[(i + 1, j + 1)]   # (++)
            n2 = node_at[(i,     j + 1)]   # (-+)
            n3 = node_at[(i,     j)]       # (--)
            n4 = node_at[(i + 1, j)]       # (+-)
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
        layout = QVBoxLayout(self)
        # ===== Parámetros de la corrida =====
        params = QGroupBox("Configuración del estudio")
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

        bb = QDialogButtonBox(QDialogButtonBox.Close)
        bb.rejected.connect(self.reject)
        layout.addWidget(bb)

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

        rows = []
        prev_u_max = None
        prev_sx_max = None
        for n in self.refinements:
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

        self.df_results = pd.DataFrame(rows)
        self._update_table()
        self._update_plot()
        self.btn_export.setEnabled(True)

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
