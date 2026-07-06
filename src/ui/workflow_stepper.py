"""Barra horizontal estilo wizard con las 7 etapas del workflow MEF.

Se coloca tipicamente arriba del editor para que el usuario sepa en que
paso esta y pueda saltar a otro haciendo click. Inspirado en la barra de
modulos de Abaqus.

Uso:
    stepper = WorkflowStepper()
    stepper.step_clicked.connect(on_step_change)
    stepper.set_current_step(WorkflowStepper.STEP_MESH)
"""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QPushButton, QButtonGroup, QSizePolicy, QLabel,
)

from .icons import icon
from .theme import Colors


# Constantes publicas: indices de los pasos del workflow.
STEP_GEOMETRY = 0
STEP_MATERIAL = 1
STEP_MESH     = 2
STEP_SUPPORTS = 3
STEP_LOADS    = 4
STEP_SOLVE    = 5
STEP_RESULTS  = 6


# QSS por chip. Tres estados se distinguen por dynamic properties:
#   active="true"    -> step en curso (navy fuerte)
#   completed="true" -> step ya pasado (blanco con borde navy)
#   sin propiedades  -> pendiente (gris subtle)
_CHIP_QSS = f"""
QPushButton {{
    background: {Colors.BG_SUBTLE};
    color: {Colors.TEXT_MUTED};
    border: 1px solid {Colors.BORDER};
    border-radius: 4px;
    padding: 8px 14px;
    font-size: 9.5pt;
    text-align: left;
}}
QPushButton:hover {{
    background: {Colors.PRIMARY_LIGHT};
    color: {Colors.PRIMARY};
    border: 1px solid {Colors.PRIMARY};
}}
QPushButton[active="true"] {{
    background: {Colors.PRIMARY};
    color: white;
    font-weight: bold;
    border: 1px solid {Colors.PRIMARY};
}}
QPushButton[active="true"]:hover {{
    background: {Colors.PRIMARY_HOVER};
    color: white;
    border: 1px solid {Colors.PRIMARY_HOVER};
}}
QPushButton[completed="true"] {{
    background: white;
    color: {Colors.PRIMARY};
    border: 1px solid {Colors.PRIMARY};
    font-weight: 500;
}}
QPushButton[completed="true"]:hover {{
    background: {Colors.PRIMARY_LIGHT};
    color: {Colors.PRIMARY};
    border: 1px solid {Colors.PRIMARY};
}}
"""


# Tooltips por step, explican que se hace en cada etapa.
_TOOLTIPS = [
    "Define los nodos y elementos del modelo",
    "Configura E, nu y espesor del elemento",
    "Genera una malla rectangular automática N x M",
    "Restringe nodos en X y/o Y",
    "Aplica fuerzas nodales Fx, Fy",
    "Recalcula los 15 pasos del MEF con los datos actuales",
    "Visualiza deformada, esfuerzos y reacciones",
]


class WorkflowStepper(QWidget):
    """Barra horizontal de 7 etapas del workflow MEF.

    Visualmente: una fila de "chips" con numero + nombre + icono.
    El chip activo se resalta en navy fuerte; los previos en navy claro
    sobre fondo blanco; los siguientes en gris (pendientes).

    Click en cualquier chip emite step_clicked(index).
    """

    step_clicked = Signal(int)

    # (nombre visible, icono fa5s)
    STEPS = [
        ("Geometría",  "fa5s.draw-polygon"),
        ("Material",   "fa5s.flask"),
        ("Mallado",    "fa5s.th"),
        ("Apoyos",     "fa5s.anchor"),
        ("Cargas",     "fa5s.long-arrow-alt-down"),
        ("Resolver",   "fa5s.calculator"),
        ("Resultados", "fa5s.chart-line"),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)

        # Altura fija para que no se "estire" verticalmente al meterlo en un
        # layout vertical (e.g., arriba de un canvas).
        self.setFixedHeight(44)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        # QButtonGroup con exclusividad: solo un chip checked a la vez.
        # Lo usamos para coherencia visual; el repintado real se hace via
        # dynamic properties (active/completed) en _refresh_styles.
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self._buttons: list[QPushButton] = []
        self._current = 0

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)

        # Construccion de chips + conectores ">" entre ellos.
        n = len(self.STEPS)
        for i, (name, icon_name) in enumerate(self.STEPS):
            btn = self._make_chip(i, name, icon_name)
            self._buttons.append(btn)
            self._group.addButton(btn, i)
            layout.addWidget(btn)

            # Chevron decorativo entre chips (no entre el ultimo y el stretch).
            if i < n - 1:
                sep = QLabel(">")
                sep.setStyleSheet(
                    f"color: {Colors.TEXT_MUTED}; "
                    f"font-size: 11pt; padding: 0 2px;"
                )
                sep.setAlignment(Qt.AlignCenter)
                layout.addWidget(sep)

        # Stretch final: chips quedan alineados a la izquierda y mantienen
        # su tamano natural en lugar de expandirse.
        layout.addStretch(1)

        # Senal unica del grupo: traduce id -> step_clicked.
        self._group.idClicked.connect(self._on_button_clicked)

        # Estado inicial: primer step activo, resto pendientes.
        self.set_current_step(STEP_GEOMETRY)

    # ------------------------------------------------------------------ API

    def set_current_step(self, idx: int) -> None:
        """Marca el step idx como el actual.

        Steps anteriores quedan en estado "completed"; posteriores en
        "pending". No emite la senal (eso solo lo hace el click del usuario).
        """
        if not (0 <= idx < len(self._buttons)):
            return
        self._current = idx
        # Bloqueamos senales del grupo para no disparar idClicked al cambiar
        # el checked programaticamente.
        self._group.blockSignals(True)
        self._buttons[idx].setChecked(True)
        self._group.blockSignals(False)
        self._refresh_styles()

    def current_step(self) -> int:
        """Indice del step activo."""
        return self._current

    # ----------------------------------------------------------- internals

    def _make_chip(self, idx: int, name: str, icon_name: str) -> QPushButton:
        """Crea un chip (QPushButton checkable) con icono + numero + nombre."""
        # Numero antepuesto para que el usuario lea "1 Geometria", etc.
        text = f"{idx + 1}  {name}"
        btn = QPushButton(text, self)
        btn.setCheckable(True)
        btn.setAutoExclusive(False)  # Lo maneja el QButtonGroup.
        btn.setCursor(Qt.PointingHandCursor)
        btn.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        btn.setMinimumHeight(32)
        # Icono inicial en color "muted"; se actualiza por estado en _refresh_styles.
        btn.setIcon(icon(icon_name, color=Colors.TEXT_MUTED))
        btn.setToolTip(_TOOLTIPS[idx])
        btn.setStyleSheet(_CHIP_QSS)
        # Guardamos el nombre del icono para poder recolorearlo segun estado.
        btn.setProperty("_icon_name", icon_name)
        return btn

    def _on_button_clicked(self, idx: int) -> None:
        """Click del usuario: actualiza estado interno y emite la senal."""
        self.set_current_step(idx)
        self.step_clicked.emit(idx)

    def _refresh_styles(self) -> None:
        """Aplica las dynamic properties active/completed y reaplica el QSS.

        Qt no refresca el style si solo cambias una property: hay que
        unpolish + polish para que el selector [active="true"] tome efecto.
        Tambien recoloreamos el icono porque qtawesome renderiza el color
        en el momento de crear el QIcon (no via QSS).
        """
        for i, btn in enumerate(self._buttons):
            if i < self._current:
                state_active = False
                state_completed = True
                ico_color = Colors.PRIMARY
            elif i == self._current:
                state_active = True
                state_completed = False
                ico_color = "white"
            else:
                state_active = False
                state_completed = False
                ico_color = Colors.TEXT_MUTED

            btn.setProperty("active", state_active)
            btn.setProperty("completed", state_completed)

            # Recolorear el icono para que case con el color del texto del chip.
            icon_name = btn.property("_icon_name")
            if icon_name:
                btn.setIcon(icon(icon_name, color=ico_color))

            # Forzar repaint del QSS tras cambio de dynamic property.
            btn.style().unpolish(btn)
            btn.style().polish(btn)
            btn.update()
