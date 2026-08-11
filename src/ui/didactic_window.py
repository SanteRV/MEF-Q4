"""Ventana secundaria "Modo didáctico" — Corrección 2, punto 7.

Para qué sirve: el desarrollo paso a paso es valioso para uno o dos
elementos, pero estorba cuando el aplicativo se usa para analizar una
estructura completa. Por eso sale del flujo principal y vive aquí, en una
ventana aparte que se abre solo cuando el usuario quiere estudiar la
formulación.

La ventana principal queda dedicada al MODELO (dibujar, asignar, resolver);
esta hospeda las vistas por formulación:

    Plano: paso a paso / editor / vista 3D   (elemento Q4, cap. 01.01.02)
    Placa (flexión)                          (plate 12 GDL, cap. 01.01.03)
    Lámina (shell)                           (flat shell 20 GDL, cap. 01.01.04)
    Pórtico (frame 3D)                       (frame, cap. 01.02)

No duplica lógica: recibe ya construidos los mismos widgets que antes
vivían en las pestañas de la ventana principal.
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QLabel, QMainWindow, QTabWidget, QVBoxLayout, QWidget,
)

from .theme import Colors


class DidacticWindow(QMainWindow):
    """Ventana con las pestañas didácticas por formulación."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Modo didáctico — desarrollo paso a paso")
        self.resize(1280, 860)

        central = QWidget()
        lay = QVBoxLayout(central)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        aviso = QLabel(
            "  Estas vistas explican la formulación elemento por elemento. "
            "Para analizar una estructura completa use la ventana principal "
            "(Modelo)."
        )
        aviso.setWordWrap(True)
        aviso.setStyleSheet(
            f"background:{Colors.BG_SUBTLE}; color:{Colors.TEXT_MUTED};"
            f"border-bottom:1px solid {Colors.BORDER}; padding:7px 12px;"
            "font-size:9.5pt;"
        )
        lay.addWidget(aviso)

        self.tabs = QTabWidget()
        self.tabs.setIconSize(QSize(18, 18))
        lay.addWidget(self.tabs, 1)

        self.setCentralWidget(central)
        self.statusBar().showMessage(
            "Modo didáctico — el análisis de estructuras completas está en "
            "la ventana principal."
        )

    def add_view(self, widget: QWidget, icon: QIcon, titulo: str,
                 tooltip: str = "") -> None:
        """Agrega una vista didáctica ya construida."""
        i = self.tabs.addTab(widget, icon, titulo)
        if tooltip:
            self.tabs.setTabToolTip(i, tooltip)

    def show_view(self, titulo: str) -> None:
        """Trae la ventana al frente mostrando una vista concreta."""
        for i in range(self.tabs.count()):
            if self.tabs.tabText(i) == titulo:
                self.tabs.setCurrentIndex(i)
                break
        self.show()
        self.raise_()
        self.activateWindow()
