"""Diálogo de selección de material del catálogo.

Muestra una lista de materiales predefinidos a la izquierda y un panel
detalle a la derecha con propiedades, descripción y comparación frente al
material actual del proyecto.

Uso:
    dlg = MaterialDialog(self, current_E=el.E, current_nu=el.nu)
    if dlg.exec() == QDialog.Accepted and dlg.selected_material:
        mat = dlg.selected_material
        # aplicar mat.E y mat.nu a los inputs de la UI
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem,
    QLabel, QPushButton, QDialogButtonBox, QGroupBox, QFormLayout,
    QFrame,
)

from ..fem.materials import MATERIALS, Material, get_material
from .theme import Colors


class MaterialDialog(QDialog):
    """Diálogo que permite seleccionar un material del catálogo.

    El usuario ve la lista de materiales con sus propiedades. Al elegir uno
    y presionar OK, el diálogo guarda la elección. La app principal puede
    leer `dialog.selected_material` (Material o None).
    """

    def __init__(self, parent=None, current_E: float | None = None,
                 current_nu: float | None = None):
        super().__init__(parent)
        self.setWindowTitle("Seleccionar material")
        self.resize(800, 500)

        self.current_E = current_E
        self.current_nu = current_nu
        self.selected_material: Material | None = None

        self._build_ui()
        # Seleccionar el primer material por defecto
        if self.list_widget.count() > 0:
            self.list_widget.setCurrentRow(0)

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # Encabezado
        header = QLabel("Catálogo de materiales")
        header.setProperty("heading", True)
        header.setStyleSheet(
            f"color: {Colors.PRIMARY}; font-size: 14pt; font-weight: bold;"
        )
        layout.addWidget(header)

        subtitle = QLabel(
            "Elige un material del catálogo para copiar sus propiedades "
            "elásticas (E, ν) al proyecto."
        )
        subtitle.setProperty("muted", True)
        subtitle.setStyleSheet(f"color: {Colors.TEXT_MUTED};")
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

        # ============= Splitter horizontal =============
        body = QHBoxLayout()
        body.setSpacing(12)

        # ---- Lado izquierdo: lista de materiales (40%) ----
        list_box = QGroupBox("Materiales")
        list_layout = QVBoxLayout(list_box)
        list_layout.setContentsMargins(6, 14, 6, 8)

        self.list_widget = QListWidget()
        self.list_widget.setStyleSheet(f"""
            QListWidget {{
                background-color: {Colors.BG};
                border: 1px solid {Colors.BORDER};
                border-radius: 4px;
                padding: 4px;
                outline: none;
            }}
            QListWidget::item {{
                padding: 10px 12px;
                border-bottom: 1px solid {Colors.BG_SUBTLE};
            }}
            QListWidget::item:hover {{
                background-color: {Colors.PRIMARY_LIGHT};
                color: {Colors.PRIMARY};
            }}
            QListWidget::item:selected {{
                background-color: {Colors.PRIMARY};
                color: white;
                font-weight: bold;
            }}
        """)
        for name in MATERIALS:
            item = QListWidgetItem(name)
            item.setData(Qt.UserRole, name)
            self.list_widget.addItem(item)
        self.list_widget.currentItemChanged.connect(self._on_selection_changed)
        list_layout.addWidget(self.list_widget)

        body.addWidget(list_box, 4)

        # ---- Lado derecho: detalle (60%) ----
        detail_box = QGroupBox("Detalle del material")
        detail_layout = QVBoxLayout(detail_box)
        detail_layout.setContentsMargins(10, 14, 10, 10)
        detail_layout.setSpacing(8)

        # Nombre grande
        self.name_label = QLabel("—")
        self.name_label.setStyleSheet(
            f"color: {Colors.PRIMARY}; font-size: 14pt; font-weight: bold;"
        )
        detail_layout.addWidget(self.name_label)

        # Tabla de propiedades (QFormLayout)
        prop_frame = QFrame()
        prop_frame.setStyleSheet(
            f"QFrame {{ background-color: {Colors.BG_SUBTLE}; "
            f"border: 1px solid {Colors.BORDER}; border-radius: 4px; }}"
        )
        prop_layout = QFormLayout(prop_frame)
        prop_layout.setContentsMargins(12, 10, 12, 10)
        prop_layout.setSpacing(6)
        self.E_value = QLabel("—")
        self.nu_value = QLabel("—")
        self.rho_value = QLabel("—")
        for lbl in (self.E_value, self.nu_value, self.rho_value):
            lbl.setStyleSheet(
                f"color: {Colors.TEXT}; font-family: 'Consolas','Courier New',monospace;"
            )
        prop_layout.addRow(self._field_label("E (Módulo de Young):"), self.E_value)
        prop_layout.addRow(self._field_label("ν (Coef. de Poisson):"), self.nu_value)
        prop_layout.addRow(self._field_label("ρ (Densidad):"), self.rho_value)
        detail_layout.addWidget(prop_frame)

        # Descripción
        desc_title = QLabel("Descripción")
        desc_title.setStyleSheet(
            f"color: {Colors.PRIMARY}; font-weight: bold; margin-top: 4px;"
        )
        detail_layout.addWidget(desc_title)

        self.desc_label = QLabel("—")
        self.desc_label.setWordWrap(True)
        self.desc_label.setStyleSheet(
            f"color: {Colors.TEXT}; padding: 6px 4px;"
        )
        self.desc_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        detail_layout.addWidget(self.desc_label, 1)

        # Comparación con material actual
        self.compare_box = QGroupBox("Comparación con material actual")
        compare_layout = QVBoxLayout(self.compare_box)
        compare_layout.setContentsMargins(10, 14, 10, 8)
        compare_layout.setSpacing(4)
        self.compare_current = QLabel("—")
        self.compare_selected = QLabel("—")
        self.compare_delta = QLabel("—")
        for lbl in (self.compare_current, self.compare_selected,
                    self.compare_delta):
            lbl.setStyleSheet(
                f"color: {Colors.TEXT}; font-family: 'Consolas','Courier New',monospace; "
                f"font-size: 9.5pt;"
            )
        self.compare_delta.setStyleSheet(
            f"color: {Colors.PRIMARY}; font-family: 'Consolas','Courier New',monospace; "
            f"font-size: 9.5pt; font-weight: bold;"
        )
        compare_layout.addWidget(self.compare_current)
        compare_layout.addWidget(self.compare_selected)
        compare_layout.addWidget(self.compare_delta)
        detail_layout.addWidget(self.compare_box)

        # Si no hay material actual provisto, ocultar el bloque de comparación
        if self.current_E is None or self.current_nu is None:
            self.compare_box.setVisible(False)

        body.addWidget(detail_box, 6)
        layout.addLayout(body, 1)

        # ============= Botones inferiores =============
        button_row = QHBoxLayout()
        button_row.addStretch()

        self.btn_cancel = QPushButton("Cancelar")
        self.btn_cancel.clicked.connect(self.reject)
        button_row.addWidget(self.btn_cancel)

        self.btn_apply = QPushButton("Aplicar")
        self.btn_apply.setProperty("primary", True)
        self.btn_apply.setDefault(True)
        self.btn_apply.clicked.connect(self._on_apply)
        button_row.addWidget(self.btn_apply)

        layout.addLayout(button_row)

    def _field_label(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(f"color: {Colors.TEXT_MUTED}; font-weight: 600;")
        return lbl

    # ------------------------------------------------------------------ Slots
    def _on_selection_changed(self, current: QListWidgetItem,
                              _previous: QListWidgetItem):
        if current is None:
            return
        name = current.data(Qt.UserRole)
        try:
            mat = get_material(name)
        except KeyError:
            return
        self._update_detail(mat)

    @staticmethod
    def _fmt_E(E: float) -> str:
        """Formato humano para el módulo de elasticidad.

        Usa GPa / MPa / kPa según el orden de magnitud, igual que en
        ingeniería de materiales (Acero 210 GPa, Hormigón 25 GPa, etc.).
        """
        absE = abs(E)
        if absE >= 1e9:
            return f"{E / 1e9:.3g} GPa  ({E:.3e} Pa)"
        if absE >= 1e6:
            return f"{E / 1e6:.3g} MPa  ({E:.3e} Pa)"
        if absE >= 1e3:
            return f"{E / 1e3:.3g} kPa  ({E:.3e} Pa)"
        return f"{E:.3g} Pa"

    def _update_detail(self, mat: Material):
        self.name_label.setText(mat.name)
        self.E_value.setText(self._fmt_E(mat.E))
        self.nu_value.setText(f"{mat.nu:.3f}")
        self.rho_value.setText(f"{mat.density:.0f} kg/m³")
        self.desc_label.setText(mat.description)

        # Comparación
        if self.current_E is not None and self.current_nu is not None:
            self.compare_current.setText(
                f"Material actual:      E = {self._fmt_E(self.current_E)}, "
                f"ν = {self.current_nu:.3f}"
            )
            self.compare_selected.setText(
                f"Material seleccionado: E = {self._fmt_E(mat.E)}, ν = {mat.nu:.3f}"
            )
            dE = mat.E - self.current_E
            dnu = mat.nu - self.current_nu
            rel_E = (dE / self.current_E * 100.0) if self.current_E != 0 else float("nan")
            self.compare_delta.setText(
                f"Cambio:               ΔE = {self._fmt_E(dE)} ({rel_E:+.2f} %),  "
                f"Δν = {dnu:+.3f}"
            )

    def _on_apply(self):
        item = self.list_widget.currentItem()
        if item is None:
            self.reject()
            return
        name = item.data(Qt.UserRole)
        try:
            self.selected_material = get_material(name)
        except KeyError:
            self.selected_material = None
            self.reject()
            return
        self.accept()
