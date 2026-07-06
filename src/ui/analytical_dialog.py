"""Diálogo de comparación MEF vs solución analítica."""
from __future__ import annotations
import numpy as np
import pandas as pd

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
    QGroupBox, QFormLayout, QLineEdit, QSpinBox, QTableView, QHeaderView,
    QDialogButtonBox, QMessageBox, QPlainTextEdit,
)
from PySide6.QtGui import QStandardItemModel, QStandardItem

from ..fem.analytical import ANALYTICAL_CASES
from ..fem.solver import solve


class AnalyticalDialog(QDialog):
    def __init__(self, parent=None, structure=None):
        super().__init__(parent)
        self.setWindowTitle("Comparación MEF vs Solución Analítica")
        self.resize(900, 650)

        # Tomar propiedades del proyecto si existe
        if structure and structure.elements:
            el0 = structure.elements[0]
            self.E, self.nu, self.t = el0.E, el0.nu, el0.t
        else:
            self.E, self.nu, self.t = 2173706.0, 0.15, 0.1

        self._build_ui()
        self._on_case_changed()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # Selector de caso
        sel_box = QGroupBox("Caso analítico")
        sel_layout = QFormLayout(sel_box)
        self.case_combo = QComboBox()
        for k in ANALYTICAL_CASES:
            self.case_combo.addItem(k)
        self.case_combo.currentTextChanged.connect(self._on_case_changed)
        sel_layout.addRow("Problema:", self.case_combo)
        layout.addWidget(sel_box)

        # Parámetros editables
        param_box = QGroupBox("Parámetros del problema")
        pf = QFormLayout(param_box)
        self.E_input = QLineEdit(repr(float(self.E)))
        self.nu_input = QLineEdit(repr(float(self.nu)))
        self.t_input = QLineEdit(repr(float(self.t)))
        self.L_input = QLineEdit("1.0")
        self.H_input = QLineEdit("1.0")
        self.q_input = QLineEdit("1000.0")
        self.nx_input = QSpinBox(); self.nx_input.setRange(1, 50); self.nx_input.setValue(2)
        self.ny_input = QSpinBox(); self.ny_input.setRange(1, 50); self.ny_input.setValue(2)
        pf.addRow("E (Pa):", self.E_input)
        pf.addRow("ν:", self.nu_input)
        pf.addRow("t (m):", self.t_input)
        pf.addRow("L (m):", self.L_input)
        pf.addRow("H (m):", self.H_input)
        pf.addRow("Carga q (Pa) [o τ en corte]:", self.q_input)
        pf.addRow("Refinamiento Nx:", self.nx_input)
        pf.addRow("Refinamiento Ny:", self.ny_input)
        layout.addWidget(param_box)

        # Descripción del caso
        self.desc_text = QPlainTextEdit()
        self.desc_text.setReadOnly(True)
        self.desc_text.setMaximumHeight(140)
        layout.addWidget(self.desc_text)

        # Botón de comparación
        row = QHBoxLayout()
        btn_run = QPushButton("Ejecutar comparación")
        btn_run.setProperty("primary", True)
        btn_run.clicked.connect(self._run)
        row.addWidget(btn_run)
        row.addStretch()
        layout.addLayout(row)

        # Tabla de resultados
        self.table = QTableView()
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        layout.addWidget(self.table, 1)

        bb = QDialogButtonBox(QDialogButtonBox.Close)
        bb.rejected.connect(self.reject)
        layout.addWidget(bb)

    def _on_case_changed(self):
        name = self.case_combo.currentText()
        if name == "Tracción uniforme":
            self.desc_text.setPlainText(
                "Placa rectangular L×H sometida a tracción uniforme q en el "
                "borde derecho.\n"
                "El borde izquierdo está restringido en X y Y.\n\n"
                "Solución teórica (estado uniforme):\n"
                "    σx = q,  σy = 0,  τxy = 0\n"
                "    u_max (extremo derecho) = q·L / E"
            )
        elif name == "Corte puro":
            self.desc_text.setPlainText(
                "Placa rectangular bajo tracciones tangenciales τ en sus 4 bordes.\n"
                "Esquina inferior izquierda fija; borde inferior con roller en Y; "
                "borde izquierdo con roller en X.\n\n"
                "Solución teórica:\n"
                "    σx = σy = 0,  τxy = τ\n"
                "    γxy = τ / G  donde G = E / [2(1+ν)]"
            )

    def _run(self):
        try:
            E = float(self.E_input.text())
            nu = float(self.nu_input.text())
            t = float(self.t_input.text())
            L = float(self.L_input.text())
            H = float(self.H_input.text())
            q = float(self.q_input.text())
            nx = self.nx_input.value()
            ny = self.ny_input.value()
        except Exception as e:
            QMessageBox.critical(self, "Error en parámetros", str(e))
            return

        name = self.case_combo.currentText()
        case_fn = ANALYTICAL_CASES[name]
        try:
            structure, case = case_fn(E, nu, t, L=L, H=H, q=q, nx=nx, ny=ny)
        except TypeError:
            # Algunos casos usan `tau` en vez de `q`
            structure, case = case_fn(E, nu, t, L=L, H=H, tau=q, nx=nx, ny=ny)

        result = solve(structure)
        # Tomar σ en el centro del primer elemento como referencia (estado uniforme)
        sigma_center = (result.elements[0].stresses_at_gauss[0]
                        + result.elements[0].stresses_at_gauss[1]
                        + result.elements[0].stresses_at_gauss[2]
                        + result.elements[0].stresses_at_gauss[3]) / 4.0
        u_max_mef = float(np.max(np.abs(result.displacements)))

        # Construir tabla de comparación
        rows = []
        def add(name_field, mef_val, ana_val):
            if np.isnan(ana_val):
                rel = float("nan")
                txt = "—"
            else:
                rel = (mef_val - ana_val) / ana_val * 100 if ana_val != 0 else float("nan")
                txt = f"{rel:.3f} %" if not np.isnan(rel) else "—"
            rows.append({
                "Magnitud": name_field,
                "MEF": mef_val,
                "Analítica": ana_val,
                "Error relativo": txt,
            })

        add("σx (Pa)",   sigma_center[0], case.sigma_x_expected)
        add("σy (Pa)",   sigma_center[1], case.sigma_y_expected)
        add("τxy (Pa)",  sigma_center[2], case.tau_xy_expected)
        add("u_max (m)", u_max_mef,        case.u_max_expected)

        df = pd.DataFrame(rows)
        self._fill_table(df)

    def _fill_table(self, df: pd.DataFrame):
        model = QStandardItemModel(df.shape[0], df.shape[1])
        model.setHorizontalHeaderLabels([str(c) for c in df.columns])
        for r in range(df.shape[0]):
            for c in range(df.shape[1]):
                v = df.iat[r, c]
                if isinstance(v, float) and not np.isnan(v):
                    txt = f"{v:.6e}"
                elif isinstance(v, float) and np.isnan(v):
                    txt = "—"
                else:
                    txt = str(v)
                item = QStandardItem(txt)
                item.setEditable(False)
                item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                model.setItem(r, c, item)
        self.table.setModel(model)
