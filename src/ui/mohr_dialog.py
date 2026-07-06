"""Diálogo del Círculo de Mohr para un punto (Gauss o esquina) de un elemento Q4.

Dado el estado plano de esfuerzos (σx, σy, τxy) en un punto del proyecto actual,
calcula y grafica el círculo de Mohr, junto con los esfuerzos principales σ1, σ2,
el cortante máximo τ_max y el ángulo principal θp.

Convención de signos para la gráfica:
    Eje horizontal: σ (esfuerzo normal), positivo a la derecha.
    Eje vertical:   τ (esfuerzo cortante), positivo hacia ARRIBA.
    El punto "X" (cara perpendicular a x) se ubica en (σx, -τxy) y el punto "Y"
    en (σy, +τxy). Es la convención académica estándar (ver Hibbeler, Beer).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, QFormLayout,
    QComboBox, QLabel, QTableView, QHeaderView, QDialogButtonBox,
)
from PySide6.QtGui import QStandardItemModel, QStandardItem

from .theme import Colors


# Etiquetas de los 8 puntos por elemento: 4 Gauss + 4 esquinas.
_POINT_LABELS = ["GP1", "GP2", "GP3", "GP4", "N1", "N2", "N3", "N4"]


def _compute_mohr(sx: float, sy: float, txy: float) -> dict:
    """Devuelve las magnitudes derivadas del círculo de Mohr."""
    sigma_avg = (sx + sy) / 2.0
    R = float(np.sqrt(((sx - sy) / 2.0) ** 2 + txy ** 2))
    sigma_1 = sigma_avg + R
    sigma_2 = sigma_avg - R
    tau_max = R
    theta_p_rad = 0.5 * np.arctan2(2.0 * txy, sx - sy)
    theta_p_deg = float(np.degrees(theta_p_rad))
    return {
        "sigma_avg": float(sigma_avg),
        "R": R,
        "sigma_1": float(sigma_1),
        "sigma_2": float(sigma_2),
        "tau_max": float(tau_max),
        "theta_p_rad": float(theta_p_rad),
        "theta_p_deg": theta_p_deg,
    }


class MohrDialog(QDialog):
    """Diálogo interactivo del círculo de Mohr."""

    def __init__(self, parent=None, structure=None, fem_result=None):
        super().__init__(parent)
        self.setWindowTitle("Círculo de Mohr — Estado plano de esfuerzos")
        self.resize(900, 650)

        self.structure = structure
        self.fem_result = fem_result

        self._build_ui()
        self._populate_elements()
        self._on_selection_changed()

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        root = QVBoxLayout(self)

        title = QLabel("Círculo de Mohr en un punto")
        title.setProperty("heading", True)
        title.setStyleSheet(f"color:{Colors.PRIMARY}; font-size:14pt; font-weight:bold;")
        root.addWidget(title)

        main_row = QHBoxLayout()
        root.addLayout(main_row, 1)

        # ===== Columna izquierda: selección + esfuerzos + tabla =====
        left = QVBoxLayout()
        main_row.addLayout(left, 0)

        # Selector
        sel_box = QGroupBox("Selección de punto")
        sel_form = QFormLayout(sel_box)
        self.combo_element = QComboBox()
        self.combo_point = QComboBox()
        for lbl in _POINT_LABELS:
            self.combo_point.addItem(lbl)
        self.combo_element.currentIndexChanged.connect(self._on_selection_changed)
        self.combo_point.currentIndexChanged.connect(self._on_selection_changed)
        sel_form.addRow("Elemento:", self.combo_element)
        sel_form.addRow("Punto:", self.combo_point)
        left.addWidget(sel_box)

        # Esfuerzos en el punto
        stress_box = QGroupBox("Esfuerzos en el punto")
        stress_form = QFormLayout(stress_box)
        self.lbl_sx = QLabel("—")
        self.lbl_sy = QLabel("—")
        self.lbl_txy = QLabel("—")
        for w in (self.lbl_sx, self.lbl_sy, self.lbl_txy):
            w.setStyleSheet("font-family: Consolas, monospace; font-size: 10.5pt;")
        stress_form.addRow("σx:", self.lbl_sx)
        stress_form.addRow("σy:", self.lbl_sy)
        stress_form.addRow("τxy:", self.lbl_txy)
        left.addWidget(stress_box)

        # Tabla resumen de magnitudes derivadas
        res_box = QGroupBox("Magnitudes derivadas")
        res_layout = QVBoxLayout(res_box)
        self.table = QTableView()
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setMinimumWidth(320)
        self.table.setMinimumHeight(220)
        res_layout.addWidget(self.table)
        left.addWidget(res_box, 1)

        # ===== Columna derecha: gráfica =====
        plot_box = QGroupBox("Círculo de Mohr")
        plot_layout = QVBoxLayout(plot_box)
        self.figure = Figure(figsize=(6, 6), tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        plot_layout.addWidget(self.canvas)
        main_row.addWidget(plot_box, 1)

        # ===== Botonera =====
        bb = QDialogButtonBox(QDialogButtonBox.Close)
        bb.rejected.connect(self.reject)
        root.addWidget(bb)

    # ------------------------------------------------------------------ datos
    def _populate_elements(self):
        self.combo_element.blockSignals(True)
        self.combo_element.clear()
        if self.structure is not None:
            for i, el in enumerate(self.structure.elements):
                eid = getattr(el, "id", i)
                self.combo_element.addItem(f"E{eid}", i)
        self.combo_element.blockSignals(False)

    def _current_stress_vector(self) -> np.ndarray | None:
        """Devuelve [σx, σy, τxy] del punto seleccionado, o None si no hay datos."""
        if self.fem_result is None or not self.fem_result.elements:
            return None
        el_idx = self.combo_element.currentData()
        if el_idx is None:
            return None
        el_res = self.fem_result.elements[el_idx]
        pt_idx = self.combo_point.currentIndex()
        if 0 <= pt_idx < 4:
            arr = el_res.stresses_at_gauss[pt_idx]
        elif 4 <= pt_idx < 8:
            arr = el_res.stresses_at_corners[pt_idx - 4]
        else:
            return None
        return np.asarray(arr, dtype=float).flatten()

    # ------------------------------------------------------------------ slots
    def _on_selection_changed(self):
        s = self._current_stress_vector()
        if s is None or s.size < 3:
            self.lbl_sx.setText("—")
            self.lbl_sy.setText("—")
            self.lbl_txy.setText("—")
            self.figure.clear()
            self.canvas.draw()
            self.table.setModel(QStandardItemModel(0, 2))
            return
        sx, sy, txy = float(s[0]), float(s[1]), float(s[2])
        self.lbl_sx.setText(f"{sx:.4g} Pa")
        self.lbl_sy.setText(f"{sy:.4g} Pa")
        self.lbl_txy.setText(f"{txy:.4g} Pa")
        m = _compute_mohr(sx, sy, txy)
        self._update_table(m)
        self._update_plot(sx, sy, txy, m)

    def _update_table(self, m: dict):
        rows = [
            ("σ₁ (principal mayor)", f"{m['sigma_1']:.4g} Pa"),
            ("σ₂ (principal menor)", f"{m['sigma_2']:.4g} Pa"),
            ("τ_max (en el plano)", f"{m['tau_max']:.4g} Pa"),
            ("σ_promedio (centro)", f"{m['sigma_avg']:.4g} Pa"),
            ("R (radio del círculo)", f"{m['R']:.4g} Pa"),
            ("θp (ángulo principal)", f"{m['theta_p_deg']:.4g} °"),
        ]
        model = QStandardItemModel(len(rows), 2)
        model.setHorizontalHeaderLabels(["Magnitud", "Valor"])
        for r, (k, v) in enumerate(rows):
            it_k = QStandardItem(k)
            it_v = QStandardItem(v)
            for it in (it_k, it_v):
                it.setEditable(False)
            it_v.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            model.setItem(r, 0, it_k)
            model.setItem(r, 1, it_v)
        self.table.setModel(model)

    # ------------------------------------------------------------------ plot
    def _update_plot(self, sx: float, sy: float, txy: float, m: dict):
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        cx = m["sigma_avg"]
        R = m["R"]
        s1, s2 = m["sigma_1"], m["sigma_2"]
        tau_max = m["tau_max"]

        # Círculo (paramétrico)
        theta = np.linspace(0.0, 2.0 * np.pi, 360)
        circ_x = cx + R * np.cos(theta)
        circ_y = R * np.sin(theta)
        ax.plot(circ_x, circ_y, color=Colors.PRIMARY, lw=2.0, label="Círculo de Mohr")

        # Punto X: (σx, -τxy); Punto Y: (σy, +τxy). Diámetro entre ambos.
        Xx, Xy = sx, -txy
        Yx, Yy = sy, +txy
        ax.plot([Xx, Yx], [Xy, Yy], color=Colors.TEXT if hasattr(Colors, "TEXT") else "#212529",
                lw=1.2, linestyle="--", label="Estado actual (diámetro)")
        ax.scatter([Xx], [Xy], color=Colors.PRIMARY, s=55, zorder=5)
        ax.scatter([Yx], [Yy], color=Colors.PRIMARY, s=55, zorder=5)
        ax.annotate(f"X (σx, -τxy)\n({sx:.3g}, {-txy:.3g})",
                    xy=(Xx, Xy), xytext=(8, 8), textcoords="offset points",
                    fontsize=8.5, color=Colors.PRIMARY)
        ax.annotate(f"Y (σy, +τxy)\n({sy:.3g}, {txy:.3g})",
                    xy=(Yx, Yy), xytext=(8, -18), textcoords="offset points",
                    fontsize=8.5, color=Colors.PRIMARY)

        # Esfuerzos principales (intersecciones con eje σ)
        ax.scatter([s1, s2], [0, 0], color=Colors.DANGER, s=75, zorder=6,
                   label="σ₁, σ₂ (principales)")
        ax.annotate(f"σ₁ = {s1:.3g}", xy=(s1, 0), xytext=(6, 10),
                    textcoords="offset points", fontsize=9,
                    color=Colors.DANGER, fontweight="bold")
        ax.annotate(f"σ₂ = {s2:.3g}", xy=(s2, 0), xytext=(-6, 10),
                    textcoords="offset points", fontsize=9, ha="right",
                    color=Colors.DANGER, fontweight="bold")

        # Cortante máximo (extremos superiores/inferiores del círculo)
        ax.scatter([cx, cx], [tau_max, -tau_max], color=Colors.SUCCESS,
                   s=55, zorder=5, marker="s", label=f"τ_max = ±{tau_max:.3g}")

        # Centro del círculo
        ax.scatter([cx], [0], color=Colors.PRIMARY, marker="x", s=60, zorder=6)
        ax.annotate(f"C = σ_prom = {cx:.3g}", xy=(cx, 0), xytext=(0, -16),
                    textcoords="offset points", fontsize=9, ha="center",
                    color=Colors.PRIMARY)

        # Ejes principales
        ax.axhline(0, color="#999999", lw=0.7)
        ax.axvline(0, color="#999999", lw=0.7)

        # Estética
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("σ (esfuerzo normal) [Pa]")
        ax.set_ylabel("τ (esfuerzo cortante) [Pa]")
        ax.set_title("Círculo de Mohr — estado plano")
        ax.legend(loc="upper right", fontsize=8.5, framealpha=0.9)

        # Margen visual alrededor del círculo
        pad = max(R * 0.25, 1e-9)
        ax.set_xlim(cx - R - pad, cx + R + pad)
        ax.set_ylim(-R - pad, R + pad)

        self.canvas.draw()
