"""Ventana única de modelado — Corrección 2, pasos 1 a 5 y 9.

Para qué sirve: reemplaza el flujo de "una pestaña por formulación" por un
solo ambiente de modelado, como un programa de análisis estructural:

    paso 1  definir materiales y SECCIONES (frame / plane / plate / shell)
    paso 2  definir la GRILLA de coordenadas y dibujar en 3D, con vistas
            rápidas de los planos XY, XZ y YZ
    paso 3  elegir el tipo de elemento y su sección antes de dibujar
    paso 4  asignar apoyos a VARIOS nodos seleccionados a la vez
    paso 5  asignar cargas y desplazamientos prescritos
    paso 9  resolver y consultar reacciones, esfuerzos y momentos

El cálculo lo hace el modelo unificado (src/fem/solver_model.py), que a su
vez reutiliza los cuatro núcleos ya validados.
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QCursor, QStandardItem, QStandardItemModel
from PySide6.QtWidgets import (
    QApplication, QButtonGroup, QComboBox, QDoubleSpinBox, QFormLayout,
    QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMessageBox,
    QPushButton, QSplitter, QTabWidget, QTableView, QVBoxLayout, QWidget,
)

from ..fem.grid import GridSystem
from ..fem.model import DOF_NAMES, Model
from ..fem.sections import AreaSection, FrameSection, default_library
from ..fem.solver_model import solve_model
from .icons import icon
from .model_canvas_3d import Mode, ModelCanvas3D
from .theme import Colors, style_table


def _fmt(v: float) -> str:
    """Número con precisión completa; ceros residuales como 0."""
    if abs(v) < 1e-13:
        return "0"
    return f"{v:.15g}"


def _table_model(headers: list[str], rows: list[list[str]]) -> QStandardItemModel:
    """Modelo de tabla de solo lectura a partir de texto ya formateado."""
    m = QStandardItemModel(len(rows), len(headers))
    m.setHorizontalHeaderLabels(headers)
    for r, fila in enumerate(rows):
        for c, txt in enumerate(fila):
            it = QStandardItem(txt)
            it.setEditable(False)
            if c > 0:
                it.setTextAlignment(Qt.AlignmentFlag.AlignRight
                                    | Qt.AlignmentFlag.AlignVCenter)
            m.setItem(r, c, it)
    return m


class ModelModeWidget(QWidget):
    """Pestaña "Modelo": grilla, dibujo 3D, asignaciones, análisis y resultados."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.model = Model(sections=default_library())
        self.grid = GridSystem.uniform(2, 2, 1, 5.0, 5.0, 3.0)
        self.result = None
        self._build_ui()
        self.canvas.set_grid(self.grid)
        self.canvas.set_model(self.model)
        self._refresh_sections()
        self._refresh_planes()
        self._update_info()

    # ------------------------------------------------------------ interfaz
    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        panel = QWidget()
        panel.setMaximumWidth(340)
        col = QVBoxLayout(panel)
        col.setSpacing(8)

        col.addWidget(self._grp_grid())
        col.addWidget(self._grp_draw())
        col.addWidget(self._grp_assign())
        col.addWidget(self._grp_run())
        col.addStretch(1)
        root.addWidget(panel, 0)

        derecha = QSplitter(Qt.Orientation.Vertical)

        lienzo = QWidget()
        lv = QVBoxLayout(lienzo)
        lv.setContentsMargins(0, 0, 0, 0)
        # El canvas se crea ANTES de la barra de vistas, que se conecta a él
        self.canvas = ModelCanvas3D()
        self.canvas.status_message.connect(self._on_status)
        self.canvas.model_changed.connect(self._on_model_changed)
        self.canvas.selection_changed.connect(self._on_selection)
        lv.addLayout(self._barra_vistas())
        lv.addWidget(self.canvas, 1)
        self.lbl_status = QLabel("Defina la grilla y comience a dibujar.")
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setStyleSheet("color:#555; font-size:11px;")
        lv.addWidget(self.lbl_status)
        derecha.addWidget(lienzo)

        self.tabs_res = QTabWidget()
        self.tbl_nodos = QTableView()
        self.tbl_reac = QTableView()
        self.tbl_elem = QTableView()
        for t, nombre in ((self.tbl_nodos, "Desplazamientos"),
                          (self.tbl_reac, "Reacciones"),
                          (self.tbl_elem, "Fuerzas / esfuerzos")):
            t.horizontalHeader().setSectionResizeMode(
                QHeaderView.ResizeMode.Stretch)
            style_table(t)
            self.tabs_res.addTab(t, nombre)
        derecha.addWidget(self.tabs_res)
        derecha.setStretchFactor(0, 3)
        derecha.setStretchFactor(1, 2)
        root.addWidget(derecha, 1)

    def _grp_grid(self) -> QGroupBox:
        """Paso 2: sistema de coordenadas."""
        gb = QGroupBox("1. Sistema de coordenadas (grilla)")
        f = QFormLayout(gb)
        self.ed_gx = QLineEdit("0, 5, 10")
        self.ed_gy = QLineEdit("0, 5, 10")
        self.ed_gz = QLineEdit("0, 3")
        for ed, t in ((self.ed_gx, "X"), (self.ed_gy, "Y"), (self.ed_gz, "Z")):
            ed.setToolTip(f"Coordenadas de las líneas de grilla en {t}, "
                          "separadas por comas. Ejemplo: 0, 5, 10")
            f.addRow(f"Líneas {t} (m):", ed)
        b = QPushButton("Aplicar grilla")
        b.clicked.connect(self._apply_grid)
        f.addRow(b)
        return gb

    def _grp_draw(self) -> QGroupBox:
        """Pasos 1 y 3: qué elemento y con qué sección se dibuja."""
        gb = QGroupBox("2. Dibujar")
        v = QVBoxLayout(gb)

        f = QFormLayout()
        self.cmb_tipo = QComboBox()
        self.cmb_tipo.addItems(["frame", "plane", "plate", "shell"])
        self.cmb_tipo.setCurrentText("shell")
        self.cmb_tipo.currentTextChanged.connect(self._on_tipo)
        self.cmb_seccion = QComboBox()
        self.cmb_seccion.currentTextChanged.connect(self._on_seccion)
        f.addRow("Tipo de elemento:", self.cmb_tipo)
        f.addRow("Sección:", self.cmb_seccion)
        self.cmb_plano = QComboBox()
        self.cmb_plano.currentIndexChanged.connect(self._on_plano)
        self.cmb_plano.setToolTip(
            "Plano sobre el que caen los clics al dibujar.")
        f.addRow("Plano de trabajo:", self.cmb_plano)
        v.addLayout(f)

        fila = QHBoxLayout()
        self._grp_modo = QButtonGroup(self)
        for modo, texto, ic in (
            (Mode.SELECT, "Seleccionar", "fa5s.mouse-pointer"),
            (Mode.NODE, "Nodo", "fa5s.circle"),
            (Mode.FRAME, "Frame", "fa5s.grip-lines"),
            (Mode.AREA, "Área", "fa5s.vector-square"),
        ):
            b = QPushButton(f" {texto}")
            b.setIcon(icon(ic))
            b.setCheckable(True)
            b.setToolTip(f"Modo {texto}")
            b.clicked.connect(lambda _c=False, m=modo: self.canvas.set_mode(m))
            self._grp_modo.addButton(b)
            fila.addWidget(b)
            if modo is Mode.SELECT:
                b.setChecked(True)
        v.addLayout(fila)
        return gb

    def _grp_assign(self) -> QGroupBox:
        """Pasos 4 y 5: apoyos en lote, cargas y desplazamientos prescritos."""
        gb = QGroupBox("3. Asignar a la selección")
        v = QVBoxLayout(gb)
        self.lbl_sel = QLabel("Sin nodos seleccionados.")
        self.lbl_sel.setStyleSheet("color:#555; font-size:11px;")
        v.addWidget(self.lbl_sel)

        fila = QHBoxLayout()
        for texto, tipo in (("Empotrado", "empotrado"),
                            ("Simple", "simple"),
                            ("Libre", "libre")):
            b = QPushButton(texto)
            b.setToolTip(f"Asignar apoyo {texto.lower()} a los nodos seleccionados")
            b.clicked.connect(lambda _c=False, t=tipo: self._assign_support(t))
            fila.addWidget(b)
        v.addLayout(fila)

        f = QFormLayout()
        self.cmb_gdl = QComboBox()
        self.cmb_gdl.addItems([f"{i + 1}. {n}" for i, n in enumerate(DOF_NAMES)])
        self.sp_valor = QDoubleSpinBox()
        self.sp_valor.setRange(-1e12, 1e12)
        self.sp_valor.setDecimals(6)
        self.sp_valor.setValue(-1000.0)
        f.addRow("Grado de libertad:", self.cmb_gdl)
        f.addRow("Valor:", self.sp_valor)
        v.addLayout(f)

        fila2 = QHBoxLayout()
        b_carga = QPushButton("Aplicar carga")
        b_carga.setToolTip("Fuerza (GDL 1-3) o momento (GDL 4-6) en los "
                           "nodos seleccionados")
        b_carga.clicked.connect(self._assign_load)
        b_desp = QPushButton("Desplazamiento impuesto")
        b_desp.setToolTip("Restringe el GDL y le impone el valor indicado")
        b_desp.clicked.connect(self._assign_prescribed)
        fila2.addWidget(b_carga)
        fila2.addWidget(b_desp)
        v.addLayout(fila2)

        f2 = QFormLayout()
        self.sp_q = QDoubleSpinBox()
        self.sp_q.setRange(-1e12, 1e12)
        self.sp_q.setDecimals(4)
        self.sp_q.setValue(-1000.0)
        self.sp_q.setToolTip("Presión uniforme sobre TODOS los elementos "
                             "de área (plate y shell)")
        f2.addRow("Carga distribuida q (N/m²):", self.sp_q)
        b_q = QPushButton("Aplicar q a las áreas")
        b_q.clicked.connect(self._assign_pressure)
        f2.addRow(b_q)
        v.addLayout(f2)
        return gb

    def _grp_run(self) -> QGroupBox:
        """Paso 9: análisis y resumen."""
        gb = QGroupBox("4. Analizar")
        v = QVBoxLayout(gb)
        self.btn_run = QPushButton(" Resolver modelo")
        self.btn_run.setIcon(icon("fa5s.play"))
        self.btn_run.setProperty("primary", True)
        self.btn_run.clicked.connect(self.solve)
        v.addWidget(self.btn_run)

        self.btn_report = QPushButton(" Reporte de análisis (PDF)")
        self.btn_report.setIcon(icon("fa5s.file-pdf"))
        self.btn_report.setToolTip(
            "Genera el documento con el modelo, las asignaciones y los "
            "resultados obtenidos.")
        self.btn_report.clicked.connect(self.export_report)
        v.addWidget(self.btn_report)
        self.lbl_info = QLabel("")
        self.lbl_info.setWordWrap(True)
        self.lbl_info.setStyleSheet("color:#333; font-size:11px;")
        v.addWidget(self.lbl_info)
        return gb

    def _barra_vistas(self) -> QHBoxLayout:
        """Paso 2: vistas rápidas 3D / XY / XZ / YZ."""
        fila = QHBoxLayout()
        fila.addWidget(QLabel("Vista:"))
        for texto, preset, tip in (
            ("3D", "3d", "Vista isométrica"),
            ("XY", "xy", "Planta (mirando desde +Z)"),
            ("XZ", "xz", "Elevación frontal"),
            ("YZ", "yz", "Elevación lateral"),
        ):
            b = QPushButton(texto)
            b.setMaximumWidth(52)
            b.setToolTip(tip)
            b.clicked.connect(lambda _c=False, p=preset: self.canvas.set_view(p))
            fila.addWidget(b)
        b_fit = QPushButton("Encuadrar")
        b_fit.setToolTip("Ajustar la cámara al modelo completo")
        b_fit.clicked.connect(self.canvas.fit_view)
        fila.addWidget(b_fit)
        fila.addStretch(1)
        return fila

    # ------------------------------------------------------------- slots
    def _apply_grid(self) -> None:
        try:
            self.grid.set_axis("x", GridSystem.parse_axis(self.ed_gx.text()))
            self.grid.set_axis("y", GridSystem.parse_axis(self.ed_gy.text()))
            self.grid.set_axis("z", GridSystem.parse_axis(self.ed_gz.text()))
        except ValueError as e:
            QMessageBox.critical(self, "Grilla inválida", str(e))
            return
        self.canvas.set_grid(self.grid)
        self._refresh_planes()
        self._on_status(self.grid.describe())

    def _refresh_sections(self) -> None:
        """Puebla el combo de secciones con las del tipo activo."""
        tipo = self.cmb_tipo.currentText()
        self.cmb_seccion.blockSignals(True)
        self.cmb_seccion.clear()
        nombres = [s.name for s in self.model.sections.of_type(tipo)]
        self.cmb_seccion.addItems(nombres)
        self.cmb_seccion.blockSignals(False)
        if nombres:
            self._on_seccion(nombres[0])

    def _refresh_planes(self) -> None:
        """Puebla el combo de planos de trabajo con las líneas de la grilla."""
        actual = self.cmb_plano.currentText()
        self.cmb_plano.blockSignals(True)
        self.cmb_plano.clear()
        for c in self.grid.z:
            self.cmb_plano.addItem(f"XY  (Z = {c:g})", ("xy", c))
        for c in self.grid.y:
            self.cmb_plano.addItem(f"XZ  (Y = {c:g})", ("xz", c))
        for c in self.grid.x:
            self.cmb_plano.addItem(f"YZ  (X = {c:g})", ("yz", c))
        self.cmb_plano.blockSignals(False)
        i = self.cmb_plano.findText(actual)
        self.cmb_plano.setCurrentIndex(max(i, 0))
        self._on_plano()

    def _on_tipo(self, tipo: str) -> None:
        self._refresh_sections()
        modo = Mode.FRAME if tipo == "frame" else Mode.AREA
        for b in self._grp_modo.buttons():
            b.setChecked(b.text().strip() ==
                         ("Frame" if tipo == "frame" else "Área"))
        self.canvas.active_area_type = tipo if tipo != "frame" else "shell"
        self.canvas.set_mode(modo)

    def _on_seccion(self, nombre: str) -> None:
        if not nombre:
            return
        if self.cmb_tipo.currentText() == "frame":
            self.canvas.active_frame_section = nombre
        else:
            self.canvas.active_area_section = nombre
            self.canvas.active_area_type = self.cmb_tipo.currentText()

    def _on_plano(self) -> None:
        datos = self.cmb_plano.currentData()
        if datos:
            self.canvas.set_work_plane(*datos)

    def _on_status(self, texto: str) -> None:
        self.lbl_status.setText(texto)

    def _on_model_changed(self) -> None:
        self.result = None
        self._update_info()

    def _on_selection(self, ids: list[int]) -> None:
        if not ids:
            self.lbl_sel.setText("Sin nodos seleccionados.")
        else:
            muestra = ", ".join(f"N{i + 1}" for i in ids[:6])
            extra = f" y {len(ids) - 6} más" if len(ids) > 6 else ""
            self.lbl_sel.setText(f"{len(ids)} nodo(s): {muestra}{extra}")

    def _sel_or_warn(self) -> list[int]:
        if not self.canvas.selected:
            QMessageBox.information(
                self, "Sin selección",
                "Seleccione primero uno o más nodos en el modo Seleccionar "
                "(Ctrl+clic para varios).")
            return []
        return list(self.canvas.selected)

    def _assign_support(self, tipo: str) -> None:
        ids = self._sel_or_warn()
        if not ids:
            return
        self.model.assign_support(ids, tipo)
        self.result = None
        self.canvas.redraw()
        self._update_info()
        self._on_status(f"Apoyo {tipo} asignado a {len(ids)} nodo(s).")

    def _assign_load(self) -> None:
        ids = self._sel_or_warn()
        if not ids:
            return
        k = self.cmb_gdl.currentIndex()
        valor = float(self.sp_valor.value())
        for nid in ids:
            self.model.node(nid).loads[k] = valor
        self.result = None
        self.canvas.redraw()
        self._on_status(f"Carga {DOF_NAMES[k]} = {_fmt(valor)} en "
                        f"{len(ids)} nodo(s).")

    def _assign_prescribed(self) -> None:
        ids = self._sel_or_warn()
        if not ids:
            return
        k = self.cmb_gdl.currentIndex()
        valor = float(self.sp_valor.value())
        for nid in ids:
            n = self.model.node(nid)
            n.restraints[k] = True
            n.prescribed[k] = valor
        self.result = None
        self.canvas.redraw()
        self._update_info()
        self._on_status(f"Desplazamiento impuesto {DOF_NAMES[k]} = "
                        f"{_fmt(valor)} en {len(ids)} nodo(s).")

    def _assign_pressure(self) -> None:
        q = float(self.sp_q.value())
        n = 0
        for m in self.model.members:
            if m.tipo in ("plate", "shell"):
                m.q = q
                n += 1
        self.result = None
        if n == 0:
            self._on_status("No hay elementos plate o shell en el modelo.")
        else:
            self._on_status(f"Carga q = {_fmt(q)} N/m² aplicada a "
                            f"{n} elemento(s) de área.")

    def _update_info(self) -> None:
        self.lbl_info.setText(self.model.describe())

    # ------------------------------------------------------------ análisis
    def solve(self) -> None:
        """Resuelve el modelo unificado y llena las tablas de resultados."""
        if not self.model.members:
            QMessageBox.information(self, "Modelo vacío",
                                    "Dibuje al menos un elemento antes de "
                                    "resolver.")
            return
        self._on_status("Calculando...")
        QApplication.setOverrideCursor(QCursor(Qt.CursorShape.WaitCursor))
        QApplication.processEvents()
        t0 = time.perf_counter()
        try:
            self.result = solve_model(self.model)
        except Exception as e:
            self._on_status("Error en el cálculo.")
            QMessageBox.critical(self, "Error en el cálculo", str(e))
            return
        finally:
            QApplication.restoreOverrideCursor()
        dt = time.perf_counter() - t0
        self._fill_results()
        t_txt = f"{dt * 1000:.0f} ms" if dt < 1.0 else f"{dt:.2f} s"
        u_max = float(np.max(np.abs(self.result.displacements))) if \
            self.result.displacements.size else 0.0
        self._on_status(f"Cálculo completado en {t_txt} — "
                        f"desplazamiento máximo {_fmt(u_max)} m.")
        self._update_info()

    def export_report(self) -> None:
        """Exporta el reporte de análisis del modelo unificado (punto 9)."""
        if not self.model.members:
            QMessageBox.information(self, "Modelo vacío",
                                    "Dibuje al menos un elemento antes de "
                                    "generar el reporte.")
            return
        from pathlib import Path
        from PySide6.QtWidgets import QFileDialog
        from ..export.model_report import export_model_report

        ruta, _ = QFileDialog.getSaveFileName(
            self, "Guardar reporte de análisis",
            "reporte_modelo.pdf", "Documento PDF (*.pdf)")
        if not ruta:
            return
        QApplication.setOverrideCursor(QCursor(Qt.CursorShape.WaitCursor))
        try:
            export_model_report(self.model, self.result, Path(ruta))
        except Exception as e:
            QMessageBox.critical(self, "Error al generar el reporte", str(e))
            return
        finally:
            QApplication.restoreOverrideCursor()
        aviso = ("" if self.result is not None
                 else " (sin resultados: el modelo no está resuelto)")
        self._on_status(f"Reporte guardado en {ruta}{aviso}.")

    def _fill_results(self) -> None:
        """Paso 9: desplazamientos, reacciones y fuerzas por elemento."""
        r = self.result
        if r is None:
            return
        activo = r.active_dofs

        filas = []
        for n in self.model.nodes:
            d = [r.displacements[g] for g in n.dofs]
            filas.append([f"N{n.id + 1}"] + [_fmt(v) for v in d])
        self.tbl_nodos.setModel(
            _table_model(["Nodo"] + list(DOF_NAMES), filas))

        filas = []
        for n in self.model.nodes:
            if not any(n.restraints):
                continue
            vals = []
            for k, g in enumerate(n.dofs):
                vals.append(_fmt(r.reactions[g])
                            if n.restraints[k] and activo[g] else "—")
            filas.append([f"N{n.id + 1}"] + vals)
        self.tbl_reac.setModel(
            _table_model(["Nodo", "Fx", "Fy", "Fz", "Mx", "My", "Mz"], filas))

        filas = []
        for mr in r.members:
            if mr.tipo == "frame" and mr.end_forces is not None:
                f = mr.end_forces
                filas.append([f"E{mr.member_id + 1}", "frame",
                              f"N={_fmt(f[0])}", f"Vy={_fmt(f[1])}",
                              f"Vz={_fmt(f[2])}", f"M={_fmt(f[4])}"])
            elif mr.moments:
                m = mr.moments[0]
                w = _fmt(mr.w_center) if mr.w_center is not None else "—"
                filas.append([f"E{mr.member_id + 1}", mr.tipo,
                              f"Mx={_fmt(m[0])}", f"My={_fmt(m[1])}",
                              f"Mxy={_fmt(m[2])}", f"w={w}"])
            elif mr.stresses:
                s = mr.stresses[0]
                filas.append([f"E{mr.member_id + 1}", mr.tipo,
                              f"σx={_fmt(s[0])}", f"σy={_fmt(s[1])}",
                              f"τxy={_fmt(s[2])}", "—"])
        self.tbl_elem.setModel(
            _table_model(["Elemento", "Tipo", "(1)", "(2)", "(3)", "(4)"],
                         filas))
