"""Punto de entrada del aplicativo MEF Q4."""
from __future__ import annotations
import sys
from PySide6.QtWidgets import QApplication
from .ui.main_window import MainWindow
from .ui.theme import apply_theme


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("MEF Q4 — Trabajo de Tesis")
    app.setOrganizationName("Tesis Ingeniería Civil")
    apply_theme(app)
    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
