"""Property Inspector — panel lateral contextual de propiedades.

Muestra y permite editar las propiedades del objeto actualmente seleccionado
(un Node o un Q4Element) en el canvas o en el ModelTree. Cuando no hay nada
seleccionado, muestra un placeholder.

Estilo inspirado en el Property Inspector de ANSYS Workbench / VS Code:
título grande, formulario debajo, edición in-place con commit en
editingFinished (Enter o focus-out).

Emite `value_changed` después de cada edición válida para que el host
(MainWindow) pueda repintar el canvas, refrescar resultados, etc.
"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QDoubleValidator
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QLabel, QLineEdit, QCheckBox,
    QComboBox, QSizePolicy,
)

from .theme import Colors


# Replicado de main_window._fmt_input para no acoplarse al módulo host:
# repr(float(v)) garantiza round-trip exacto al editar valores en QLineEdit
# (a diferencia de :g / :.6g que truncan a 6 cifras significativas).
def _fmt_input(val) -> str:
    if not np.isfinite(val):
        return str(val)
    return repr(float(val))


# Estilo base de QLineEdit (heredado del QSS global) vs estilo de error.
_LINEEDIT_OK_STYLE = ""
_LINEEDIT_ERR_STYLE = "border: 1px solid red;"


class PropertyInspector(QWidget):
    """Panel contextual de propiedades del objeto seleccionado.

    Modos:
      - empty:   "Selecciona un objeto"
      - node:    muestra/edita x, y, restricciones, desplazamientos impuestos
                 (U_c de la ec. 1.6.1) y cargas load_x, load_y
      - element: muestra/edita E, nu, t, plane_stress; muestra lista de nodos (read-only)
    """

    # Emitido cuando el usuario edita un valor (después de validación)
    value_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)

        # Referencias al objeto actualmente mostrado. Usadas por los slots de
        # edición para saber qué atributo actualizar. Sólo una de las dos
        # debería estar a no-None en cada momento.
        self._current_node = None
        self._current_element = None

        # Layout raíz: título arriba + formulario debajo.
        self._root_layout = QVBoxLayout(self)
        self._root_layout.setContentsMargins(12, 12, 12, 12)
        self._root_layout.setSpacing(10)

        # Título grande (navy bold) — siempre presente; el contenido cambia
        # según el modo (empty / node / element).
        self._title = QLabel("—")
        self._title.setStyleSheet(
            f"color: {Colors.PRIMARY}; font-size: 14pt; font-weight: bold;"
        )
        self._title.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self._root_layout.addWidget(self._title)

        # Contenedor del formulario: se vacía y se vuelve a poblar cada vez
        # que cambia el modo. Se mantiene el QFormLayout y se limpian sus
        # items para no recrear el widget contenedor.
        self._form_host = QWidget(self)
        self._form_layout = QFormLayout(self._form_host)
        self._form_layout.setContentsMargins(0, 0, 0, 0)
        self._form_layout.setSpacing(6)
        self._root_layout.addWidget(self._form_host)

        # Pie del panel: queda flexible para empujar el formulario hacia
        # arriba cuando hay pocos campos.
        self._root_layout.addStretch(1)

        # Anchura mínima de la columna derecha del formulario.
        self.setMinimumWidth(220)
        self.setStyleSheet(f"background-color: {Colors.BG_SUBTLE};")

        # Arranca vacío.
        self.show_empty()

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------
    def show_empty(self) -> None:
        """Limpia el panel y muestra un placeholder."""
        self._current_node = None
        self._current_element = None
        self._clear_form()
        self._title.setText("—")

        msg = QLabel(
            "Selecciona un objeto del árbol del modelo o del lienzo "
            "para ver sus propiedades."
        )
        msg.setWordWrap(True)
        msg.setStyleSheet(f"color: {Colors.TEXT_MUTED}; font-style: italic;")
        # Se inserta como fila sin label para que ocupe ambas columnas.
        self._form_layout.addRow(msg)

    def show_node(self, node) -> None:
        """Muestra propiedades editables del nodo."""
        self._current_node = node
        self._current_element = None
        self._clear_form()
        self._title.setText(f"Nodo N{node.id + 1}")

        # x, y como QLineEdit con validador de float (cualquier valor finito).
        self._ed_x = self._make_float_edit(
            node.x, tooltip="<b>Coordenada x (m)</b><br>Posición horizontal del nodo."
        )
        self._ed_y = self._make_float_edit(
            node.y, tooltip="<b>Coordenada y (m)</b><br>Posición vertical del nodo."
        )
        self._ed_x.editingFinished.connect(
            lambda: self._commit_node_float("x", self._ed_x, validate=None)
        )
        self._ed_y.editingFinished.connect(
            lambda: self._commit_node_float("y", self._ed_y, validate=None)
        )

        # Restricciones: checkbox por GDL.
        self._cb_rx = QCheckBox()
        self._cb_rx.setChecked(bool(node.restraint_x))
        self._cb_rx.setToolTip(
            "<b>Restricción en X</b><br>"
            "Si está marcada, el desplazamiento horizontal del nodo es "
            "CONOCIDO: vale lo indicado en 'ux impuesto' (0 = apoyo rígido)."
        )
        self._cb_rx.stateChanged.connect(
            lambda _s: self._commit_node_bool("restraint_x", self._cb_rx)
        )

        self._cb_ry = QCheckBox()
        self._cb_ry.setChecked(bool(node.restraint_y))
        self._cb_ry.setToolTip(
            "<b>Restricción en Y</b><br>"
            "Si está marcada, el desplazamiento vertical del nodo es "
            "CONOCIDO: vale lo indicado en 'uy impuesto' (0 = apoyo rígido)."
        )
        self._cb_ry.stateChanged.connect(
            lambda _s: self._commit_node_bool("restraint_y", self._cb_ry)
        )

        # Desplazamientos impuestos (vector U_c de la ec. 1.6.1). Solo tienen
        # efecto en el GDL correspondiente si su restricción está marcada.
        self._ed_ux = self._make_float_edit(
            node.prescribed_x,
            tooltip="<b>ux impuesto (m)</b><br>"
                    "Desplazamiento horizontal CONOCIDO del nodo (vector U_c "
                    "de la ec. 1.6.1). Solo se aplica si la restricción en X "
                    "está marcada. 0 = apoyo rígido; distinto de 0 = "
                    "asentamiento o desplazamiento forzado.",
        )
        self._ed_uy = self._make_float_edit(
            node.prescribed_y,
            tooltip="<b>uy impuesto (m)</b><br>"
                    "Desplazamiento vertical CONOCIDO del nodo (vector U_c de "
                    "la ec. 1.6.1). Solo se aplica si la restricción en Y "
                    "está marcada. 0 = apoyo rígido.",
        )
        self._ed_ux.editingFinished.connect(
            lambda: self._commit_node_float(
                "prescribed_x", self._ed_ux, validate=None)
        )
        self._ed_uy.editingFinished.connect(
            lambda: self._commit_node_float(
                "prescribed_y", self._ed_uy, validate=None)
        )

        # Cargas: cualquier float (positivo o negativo).
        self._ed_fx = self._make_float_edit(
            node.load_x,
            tooltip="<b>Carga puntual Fx (N)</b><br>"
                    "Componente horizontal de la fuerza aplicada en el nodo. "
                    "Positivo: hacia +X.",
        )
        self._ed_fy = self._make_float_edit(
            node.load_y,
            tooltip="<b>Carga puntual Fy (N)</b><br>"
                    "Componente vertical de la fuerza aplicada en el nodo. "
                    "Positivo: hacia +Y.",
        )
        self._ed_fx.editingFinished.connect(
            lambda: self._commit_node_float("load_x", self._ed_fx, validate=None)
        )
        self._ed_fy.editingFinished.connect(
            lambda: self._commit_node_float("load_y", self._ed_fy, validate=None)
        )

        self._form_layout.addRow("x (m):", self._ed_x)
        self._form_layout.addRow("y (m):", self._ed_y)
        self._form_layout.addRow("Restricción X:", self._cb_rx)
        self._form_layout.addRow("Restricción Y:", self._cb_ry)
        self._form_layout.addRow("ux impuesto (m):", self._ed_ux)
        self._form_layout.addRow("uy impuesto (m):", self._ed_uy)
        self._form_layout.addRow("Fx (N):", self._ed_fx)
        self._form_layout.addRow("Fy (N):", self._ed_fy)

    def show_element(self, element) -> None:
        """Muestra propiedades editables del elemento."""
        self._current_node = None
        self._current_element = element
        self._clear_form()
        self._title.setText(f"Elemento E{element.id + 1}")

        # E > 0
        self._ed_E = self._make_float_edit(
            element.E,
            tooltip=(
                "<b>Módulo de elasticidad E (Pa)</b><br>"
                "Rigidez del material. Valores típicos:<br>"
                "• Acero: 2.1 × 10¹¹ Pa<br>"
                "• Hormigón: 2.5 × 10¹⁰ Pa<br>"
                "• Aluminio: 7.0 × 10¹⁰ Pa<br>"
                "Determina la magnitud de los desplazamientos: a mayor E, "
                "estructura más rígida y menores deformaciones."
            ),
        )
        self._ed_E.editingFinished.connect(
            lambda: self._commit_elem_float(
                "E", self._ed_E, validate=lambda v: v > 0.0
            )
        )

        # -1 < nu < 0.5 (catálogo físico permite negativos teóricos, pero
        # nunca llegar a 0.5 porque la matriz constitutiva se vuelve
        # singular en deformación plana incompresible).
        self._ed_nu = self._make_float_edit(
            element.nu,
            tooltip=(
                "<b>Coeficiente de Poisson ν</b><br>"
                "Adimensional, típicamente 0.0 ≤ ν < 0.5. Mide la contracción "
                "lateral relativa cuando se aplica tracción axial.<br>"
                "• Acero: 0.30<br>"
                "• Hormigón: 0.20<br>"
                "• Caucho: ≈ 0.49 (casi incompresible)"
            ),
        )
        self._ed_nu.editingFinished.connect(
            lambda: self._commit_elem_float(
                "nu", self._ed_nu, validate=lambda v: -1.0 < v < 0.5
            )
        )

        # t > 0
        self._ed_t = self._make_float_edit(
            element.t,
            tooltip=(
                "<b>Espesor t (m)</b><br>"
                "Espesor de la placa fuera del plano (en dirección Z).<br>"
                "En tensión plana: el espesor afecta linealmente la matriz K^e "
                "(K^e ∝ t)."
            ),
        )
        self._ed_t.editingFinished.connect(
            lambda: self._commit_elem_float(
                "t", self._ed_t, validate=lambda v: v > 0.0
            )
        )

        # Hipótesis: tensión plana (True) / deformación plana (False).
        self._cb_plane = QComboBox()
        self._cb_plane.addItems(["Tensión plana", "Deformación plana"])
        self._cb_plane.setCurrentIndex(0 if element.plane_stress else 1)
        self._cb_plane.setToolTip(
            "<b>Hipótesis del problema plano</b><br><br>"
            "<b>Tensión plana:</b> σz = τxz = τyz = 0.<br>"
            "Apropiada para <i>placas delgadas</i> cargadas en su plano "
            "(ej. una chapa, una membrana).<br><br>"
            "<b>Deformación plana:</b> εz = γxz = γyz = 0.<br>"
            "Apropiada para <i>sólidos largos</i> cuya sección transversal "
            "no cambia (ej. túneles, presas largas, tuberías)."
        )
        self._cb_plane.currentIndexChanged.connect(self._commit_elem_plane)

        self._form_layout.addRow("E (Pa):", self._ed_E)
        self._form_layout.addRow("ν (Poisson):", self._ed_nu)
        self._form_layout.addRow("t (m):", self._ed_t)
        self._form_layout.addRow("Hipótesis:", self._cb_plane)

        # Lista de nodos del elemento (read-only).
        node_names = ", ".join(f"N{n.id + 1}" for n in element.nodes)
        lbl_nodes = QLabel(f"Nodos: {node_names}")
        lbl_nodes.setStyleSheet(f"color: {Colors.TEXT_MUTED};")
        lbl_nodes.setWordWrap(True)
        self._form_layout.addRow(lbl_nodes)

    # ------------------------------------------------------------------
    # Helpers internos
    # ------------------------------------------------------------------
    def _clear_form(self) -> None:
        """Elimina todos los rows del formulario.

        Se necesita iterar con takeAt(0) porque takeRow() invalida los
        índices de las filas siguientes. setParent(None) basta para que el
        widget se desconecte; el GC lo libera cuando no quedan referencias.
        """
        while self._form_layout.count():
            item = self._form_layout.takeAt(0)
            if item is None:
                break
            w = item.widget()
            if w is not None:
                w.setParent(None)

    def _make_float_edit(self, value: float, *, tooltip: str = "") -> QLineEdit:
        """QLineEdit con validador double y formato de máxima precisión."""
        ed = QLineEdit(_fmt_input(value))
        # StandardNotation acepta -1.23e-4, lo cual coincide con la salida
        # de repr() para valores muy pequeños o muy grandes.
        validator = QDoubleValidator(self)
        validator.setNotation(QDoubleValidator.StandardNotation)
        ed.setValidator(validator)
        ed.setMinimumWidth(140)
        if tooltip:
            ed.setToolTip(tooltip)
        return ed

    # ---- commits Node ----
    def _commit_node_float(self, attr: str, edit: QLineEdit, *, validate):
        if self._current_node is None:
            return
        try:
            value = float(edit.text())
        except ValueError:
            edit.setStyleSheet(_LINEEDIT_ERR_STYLE)
            return
        if validate is not None and not validate(value):
            edit.setStyleSheet(_LINEEDIT_ERR_STYLE)
            return
        edit.setStyleSheet(_LINEEDIT_OK_STYLE)
        setattr(self._current_node, attr, value)
        # Re-normaliza la representación (e.g. "1." -> "1.0") sin perder el
        # round-trip, para que el usuario vea exactamente lo guardado.
        edit.setText(_fmt_input(value))
        self.value_changed.emit()

    def _commit_node_bool(self, attr: str, cb: QCheckBox):
        if self._current_node is None:
            return
        setattr(self._current_node, attr, bool(cb.isChecked()))
        self.value_changed.emit()

    # ---- commits Element ----
    def _commit_elem_float(self, attr: str, edit: QLineEdit, *, validate):
        if self._current_element is None:
            return
        try:
            value = float(edit.text())
        except ValueError:
            edit.setStyleSheet(_LINEEDIT_ERR_STYLE)
            return
        if validate is not None and not validate(value):
            edit.setStyleSheet(_LINEEDIT_ERR_STYLE)
            return
        edit.setStyleSheet(_LINEEDIT_OK_STYLE)
        setattr(self._current_element, attr, value)
        edit.setText(_fmt_input(value))
        self.value_changed.emit()

    def _commit_elem_plane(self, idx: int):
        if self._current_element is None:
            return
        # Índice 0 = Tensión plana (plane_stress=True), 1 = Deformación plana.
        self._current_element.plane_stress = (idx == 0)
        self.value_changed.emit()
