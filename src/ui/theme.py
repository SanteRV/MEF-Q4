"""Tema visual académico profesional — paleta navy + blanco.

Uso:
    from src.ui.theme import apply_theme, Colors
    apply_theme(app)              # Aplica QSS global a toda la app
    label.setStyleSheet(f"color:{Colors.PRIMARY};")
"""
from __future__ import annotations
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFont


class Colors:
    """Paleta del tema académico profesional."""
    PRIMARY        = "#1F4E78"   # Navy — encabezados, botones primarios
    PRIMARY_HOVER  = "#2E6BA0"   # Navy más claro — hover
    PRIMARY_DARK   = "#163A5C"   # Navy oscuro — pressed
    PRIMARY_LIGHT  = "#D8E4F0"   # Navy muy claro — fondo selección
    BG             = "#FFFFFF"   # Blanco — fondo principal
    BG_SUBTLE      = "#F7F9FB"   # Gris muy claro — fondo secundario
    BG_ALT         = "#EEF2F6"   # Gris ligeramente más oscuro — filas alternas
    BORDER         = "#D5DDE5"   # Gris claro — bordes
    BORDER_FOCUS   = "#1F4E78"   # Navy — borde con foco
    GRID_LINE      = "#E6ECF2"   # Gris muy claro — líneas internas de tablas
    TEXT           = "#212529"   # Casi negro — texto principal
    TEXT_MUTED     = "#6C757D"   # Gris medio — texto secundario
    SUCCESS        = "#28A745"   # Verde — apoyos, OK
    DANGER         = "#DC3545"   # Rojo — cargas, errores
    WARNING        = "#FFC107"   # Amarillo — advertencias


STYLESHEET = f"""
/* ============= Base ============= */
QMainWindow, QWidget {{
    background-color: {Colors.BG};
    color: {Colors.TEXT};
    font-family: "Segoe UI", "San Francisco", Arial, sans-serif;
    font-size: 10pt;
}}

QLabel {{
    color: {Colors.TEXT};
    /* Un poco de aire vertical: los textos explicativos del aplicativo son
       largos y sin este margen se leen apretados. */
    padding: 1px 0;
}}

QLabel[heading="true"] {{
    color: {Colors.PRIMARY};
    font-size: 16pt;
    font-weight: bold;
}}

/* Texto de ayuda o pie: gris, algo menor, para que no compita con los datos */
QLabel[hint="true"] {{
    color: {Colors.TEXT_MUTED};
    font-size: 9pt;
}}

QLabel[muted="true"] {{
    color: {Colors.TEXT_MUTED};
}}

/* ============= Header banner ============= */
QFrame#headerBanner {{
    background-color: {Colors.PRIMARY};
    border: none;
    min-height: 56px;
    max-height: 56px;
}}

QFrame#headerBanner QLabel {{
    color: white;
    background-color: transparent;
    padding: 0 16px;
}}

QFrame#headerBanner QLabel#headerTitle {{
    font-size: 16pt;
    font-weight: bold;
}}

QFrame#headerBanner QLabel#headerSubtitle {{
    font-size: 9pt;
    color: #B7CCE0;
}}

/* ============= Tabs principales ============= */
QTabWidget::pane {{
    border: 1px solid {Colors.BORDER};
    border-top: none;
    background-color: {Colors.BG};
}}

QTabBar::tab {{
    background-color: {Colors.BG_SUBTLE};
    color: {Colors.TEXT_MUTED};
    padding: 10px 24px;
    border: 1px solid {Colors.BORDER};
    border-bottom: none;
    font-size: 10pt;
    font-weight: 500;
    min-width: 140px;
}}

QTabBar::tab:hover {{
    background-color: {Colors.PRIMARY_LIGHT};
    color: {Colors.PRIMARY};
}}

QTabBar::tab:selected {{
    background-color: {Colors.PRIMARY};
    color: white;
    font-weight: bold;
}}

/* ============= Sub-tabs (modo documento, más compactos) ============= */
QTabWidget[documentMode="true"]::pane {{
    border: 1px solid {Colors.BORDER};
    background-color: {Colors.BG};
}}

QTabWidget[documentMode="true"] QTabBar::tab {{
    background-color: {Colors.BG};
    color: {Colors.TEXT_MUTED};
    padding: 6px 18px;
    border: none;
    border-bottom: 2px solid transparent;
    min-width: 120px;
    font-size: 9.5pt;
    font-weight: 500;
}}

QTabWidget[documentMode="true"] QTabBar::tab:hover {{
    color: {Colors.PRIMARY};
    background-color: {Colors.BG_SUBTLE};
}}

QTabWidget[documentMode="true"] QTabBar::tab:selected {{
    color: {Colors.PRIMARY};
    background-color: {Colors.BG};
    border-bottom: 2px solid {Colors.PRIMARY};
    font-weight: bold;
}}

/* ============= Splitter (handle visible) ============= */
QSplitter::handle {{
    background-color: {Colors.BG_SUBTLE};
}}

QSplitter::handle:horizontal {{
    width: 6px;
    border-left: 1px solid {Colors.BORDER};
    border-right: 1px solid {Colors.BORDER};
}}

QSplitter::handle:vertical {{
    height: 6px;
    border-top: 1px solid {Colors.BORDER};
    border-bottom: 1px solid {Colors.BORDER};
}}

QSplitter::handle:hover {{
    background-color: {Colors.PRIMARY_LIGHT};
}}

QSplitter::handle:pressed {{
    background-color: {Colors.PRIMARY};
}}

/* ============= GroupBox ============= */
QGroupBox {{
    background-color: {Colors.BG};
    border: 1px solid {Colors.BORDER};
    border-radius: 6px;
    margin-top: 12px;
    padding-top: 8px;
    font-weight: 600;
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 8px;
    color: {Colors.PRIMARY};
    background-color: {Colors.BG};
}}

/* ============= Botones ============= */
QPushButton {{
    background-color: {Colors.BG_SUBTLE};
    color: {Colors.TEXT};
    border: 1px solid {Colors.BORDER};
    border-radius: 4px;
    padding: 6px 14px;
    font-weight: 500;
    min-height: 24px;
}}

QPushButton:hover {{
    background-color: {Colors.PRIMARY_LIGHT};
    border-color: {Colors.PRIMARY};
}}

QPushButton:pressed {{
    background-color: {Colors.BORDER};
}}

/* Botón primario (clase) */
QPushButton[primary="true"] {{
    background-color: {Colors.PRIMARY};
    color: white;
    border: 1px solid {Colors.PRIMARY};
    font-weight: bold;
}}

QPushButton[primary="true"]:hover {{
    background-color: {Colors.PRIMARY_HOVER};
    border-color: {Colors.PRIMARY_HOVER};
}}

QPushButton[primary="true"]:pressed {{
    background-color: {Colors.PRIMARY_DARK};
}}

/* Botón de toolbar (cuadrado, para iconos) */
QPushButton[toolbar="true"] {{
    background-color: {Colors.BG};
    border: 1px solid {Colors.BORDER};
    border-radius: 4px;
    padding: 6px;
    min-width: 40px;
    min-height: 40px;
    font-size: 16pt;
}}

QPushButton[toolbar="true"]:hover {{
    background-color: {Colors.PRIMARY_LIGHT};
    border-color: {Colors.PRIMARY};
}}

QPushButton[toolbar="true"]:checked {{
    background-color: {Colors.PRIMARY};
    color: white;
    border-color: {Colors.PRIMARY};
}}

/* ============= Inputs ============= */
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background-color: {Colors.BG};
    border: 1px solid {Colors.BORDER};
    border-radius: 3px;
    padding: 4px 6px;
    color: {Colors.TEXT};
    selection-background-color: {Colors.PRIMARY_LIGHT};
}}

QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
    border-color: {Colors.BORDER_FOCUS};
    outline: none;
}}

QLineEdit:disabled {{
    background-color: {Colors.BG_SUBTLE};
    color: {Colors.TEXT_MUTED};
}}

QComboBox::drop-down {{
    border: none;
    width: 20px;
}}

QComboBox::down-arrow {{
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid {Colors.PRIMARY};
    width: 0;
    height: 0;
}}

/* ============= Lista de pasos ============= */
QListWidget {{
    background-color: {Colors.BG};
    border: 1px solid {Colors.BORDER};
    border-radius: 4px;
    padding: 4px;
    outline: none;
}}

QListWidget::item {{
    padding: 8px 12px;
    border-bottom: 1px solid {Colors.BG_SUBTLE};
    color: {Colors.TEXT};
}}

QListWidget::item:hover {{
    background-color: {Colors.BG_SUBTLE};
}}

QListWidget::item:selected {{
    background-color: {Colors.PRIMARY};
    color: white;
    border-left: 3px solid {Colors.PRIMARY_DARK};
    font-weight: bold;
}}

/* ============= Tablas =============
   Las tablas del aplicativo son casi todas NUMÉRICAS y con muchos dígitos.
   Por eso el cuerpo usa una fuente de ancho fijo: así las cifras quedan
   alineadas en columna y se comparan de un vistazo (un valor 1.23e-05
   debajo de otro 9.87e-05 se lee al instante). Los encabezados conservan
   la tipografía de la interfaz, que es más legible para texto. */
QTableView {{
    background-color: {Colors.BG};
    alternate-background-color: {Colors.BG_SUBTLE};
    border: 1px solid {Colors.BORDER};
    border-radius: 4px;
    gridline-color: {Colors.GRID_LINE};
    selection-background-color: {Colors.PRIMARY_LIGHT};
    selection-color: {Colors.TEXT};
    font-family: "Consolas", "Cascadia Mono", "DejaVu Sans Mono", monospace;
    font-size: 9.5pt;
}}

QTableView::item {{
    padding: 6px 9px;
    border: none;
}}

QTableView::item:selected {{
    background-color: {Colors.PRIMARY_LIGHT};
    color: {Colors.TEXT};
}}

QTableView QTableCornerButton::section {{
    background-color: {Colors.PRIMARY};
    border: none;
}}

QHeaderView {{
    background-color: {Colors.PRIMARY};
    font-family: "Segoe UI", "San Francisco", Arial, sans-serif;
}}

/* Encabezado de columnas: navy sólido, con aire suficiente para respirar */
QHeaderView::section:horizontal {{
    background-color: {Colors.PRIMARY};
    color: white;
    padding: 7px 11px;
    border: none;
    border-right: 1px solid {Colors.PRIMARY_HOVER};
    font-weight: 600;
    font-size: 9.5pt;
}}

/* Encabezado de filas: gris claro, discreto — es solo un índice y no debe
   competir visualmente con los datos */
QHeaderView::section:vertical {{
    background-color: {Colors.BG_ALT};
    color: {Colors.TEXT_MUTED};
    padding: 5px 9px;
    border: none;
    border-right: 1px solid {Colors.BORDER};
    border-bottom: 1px solid {Colors.GRID_LINE};
    font-weight: 500;
    font-size: 9pt;
}}

QHeaderView::section:last {{
    border-right: none;
}}

/* ============= Status bar ============= */
QStatusBar {{
    background-color: {Colors.BG_SUBTLE};
    border-top: 1px solid {Colors.BORDER};
    color: {Colors.TEXT_MUTED};
    padding: 4px;
}}

QStatusBar::item {{
    border: none;
}}

/* ============= Menú ============= */
QMenuBar {{
    background-color: {Colors.BG};
    color: {Colors.TEXT};
    border-bottom: 1px solid {Colors.BORDER};
}}

QMenuBar::item:selected {{
    background-color: {Colors.PRIMARY_LIGHT};
    color: {Colors.PRIMARY};
}}

QMenu {{
    background-color: {Colors.BG};
    border: 1px solid {Colors.BORDER};
    padding: 4px;
}}

QMenu::item {{
    padding: 6px 24px;
}}

QMenu::item:selected {{
    background-color: {Colors.PRIMARY};
    color: white;
}}

/* ============= ScrollArea / ScrollBar ============= */
QScrollArea {{
    border: 1px solid {Colors.BORDER};
    border-radius: 4px;
    background-color: {Colors.BG};
}}

QScrollBar:vertical, QScrollBar:horizontal {{
    background-color: {Colors.BG_SUBTLE};
    border: none;
    width: 10px;
    height: 10px;
}}

QScrollBar::handle {{
    background-color: {Colors.BORDER};
    border-radius: 3px;
    min-height: 30px;
    min-width: 30px;
}}

QScrollBar::handle:hover {{
    background-color: {Colors.TEXT_MUTED};
}}

QScrollBar::add-line, QScrollBar::sub-line {{
    background: none;
    border: none;
    height: 0;
    width: 0;
}}

/* ============= QPlainTextEdit (matrices) ============= */
QPlainTextEdit {{
    background-color: {Colors.BG_SUBTLE};
    border: 1px solid {Colors.BORDER};
    border-radius: 4px;
    color: {Colors.TEXT};
    font-family: "Consolas", "Courier New", monospace;
    font-size: 9.5pt;
    padding: 8px;
}}

/* ============= QCheckBox ============= */
QCheckBox::indicator {{
    width: 16px;
    height: 16px;
    border: 1.5px solid {Colors.BORDER};
    border-radius: 3px;
    background-color: {Colors.BG};
}}

QCheckBox::indicator:hover {{
    border-color: {Colors.PRIMARY};
}}

QCheckBox::indicator:checked {{
    background-color: {Colors.PRIMARY};
    border-color: {Colors.PRIMARY};
    image: none;
}}

/* ============= QSlider ============= */
QSlider::groove:horizontal {{
    border: 1px solid {Colors.BORDER};
    height: 6px;
    border-radius: 3px;
    background: {Colors.BG_SUBTLE};
}}

QSlider::handle:horizontal {{
    background: {Colors.PRIMARY};
    border: none;
    width: 16px;
    margin: -6px 0;
    border-radius: 8px;
}}

QSlider::handle:horizontal:hover {{
    background: {Colors.PRIMARY_HOVER};
}}

QSlider::sub-page:horizontal {{
    background: {Colors.PRIMARY_HOVER};
    border-radius: 3px;
}}

/* ============= QDialog ============= */
QDialog {{
    background-color: {Colors.BG};
}}
"""


def style_table(view) -> None:
    """Ajustes de presentación que el QSS no puede hacer por sí solo.

    Para qué sirve: deja las tablas legibles sin repetir el mismo bloque en
    cada widget — filas alternas, altura de fila cómoda, sin rejilla
    vertical ruidosa y encabezado de filas oculto cuando no aporta nada.
    """
    from PySide6.QtWidgets import QAbstractItemView, QHeaderView

    view.setAlternatingRowColors(True)
    view.setShowGrid(True)
    view.setWordWrap(False)
    view.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    view.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    vh = view.verticalHeader()
    vh.setDefaultSectionSize(26)          # altura de fila cómoda
    vh.setMinimumSectionSize(22)
    hh = view.horizontalHeader()
    hh.setHighlightSections(False)        # el encabezado no cambia al seleccionar
    hh.setDefaultAlignment(
        Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter)


def apply_theme(app: QApplication) -> None:
    """Aplica el tema académico profesional a toda la app."""
    # Fuente base
    font = QFont("Segoe UI", 10)
    app.setFont(font)
    # Stylesheet global
    app.setStyleSheet(STYLESHEET)
