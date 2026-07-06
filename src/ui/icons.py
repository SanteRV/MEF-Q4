"""Iconos vectoriales (Font Awesome via qtawesome).

Todos los iconos del mismo set para consistencia visual:
- Color por defecto: navy (Colors.PRIMARY)
- Tamaño por defecto: 18px
- Color cuando el botón está checked: blanco (manejado por QSS)
"""
from __future__ import annotations
from PySide6.QtGui import QIcon
import qtawesome as qta

from .theme import Colors


# Tamaño default de los iconos
ICON_SIZE = 18


def icon(name: str, color: str = Colors.PRIMARY, size: int = ICON_SIZE) -> QIcon:
    """Devuelve un QIcon con el nombre Font Awesome.

    Lista completa en https://fontawesome.com/icons (usar prefix fa5s, fa5r, fa5b)
    """
    return qta.icon(name, color=color)


def icon_with_states(name: str) -> QIcon:
    """Icon que cambia de color cuando el botón está checked/disabled.

    Se usa en los botones del toolbar (checkable):
    - normal:   navy
    - active:   blanco (cuando el botón está marcado)
    - disabled: gris claro
    """
    return qta.icon(
        name,
        color=Colors.PRIMARY,
        color_active=Colors.PRIMARY,
        color_disabled=Colors.BORDER,
        # cuando el botón es checked, Qt usa "on" state; qtawesome lo maneja con on
        on=name,
        color_on="white",
    )


# Iconos del editor (modos)
EDITOR_MODE_ICONS = {
    "select":     "fa5s.mouse-pointer",
    "add_node":   "fa5s.circle",          # nodo = círculo lleno
    "add_q4":     "fa5s.vector-square",   # cuadrado vectorial
    "apply_bc":   "fa5s.anchor",          # ancla = restricción
    "apply_load": "fa5s.long-arrow-alt-down",   # flecha vertical
    "delete":     "fa5s.trash-alt",
}

# Iconos generales
ICON_RECALC      = "fa5s.sync-alt"
ICON_EXPORT      = "fa5s.file-download"
ICON_EXCEL       = "fa5s.file-excel"
ICON_PDF         = "fa5s.file-pdf"
ICON_OPEN        = "fa5s.folder-open"
ICON_SAVE        = "fa5s.save"
ICON_GRID        = "fa5s.th"
ICON_MESH        = "fa5s.border-all"
ICON_CLEAR       = "fa5s.eraser"
ICON_TAB_STEPS   = "fa5s.list-ol"
ICON_TAB_EDITOR  = "fa5s.drafting-compass"
ICON_LOGO        = "fa5s.cube"
