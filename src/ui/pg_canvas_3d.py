"""Vista 3D del modelo Q4 plano con camara orbital.

El elemento Q4 vive en el plano XY (z=0). Esta vista usa GLViewWidget de
PyQtGraph (OpenGL) para presentar la malla rotable: nodos como esferas,
elementos como poligonos semitransparentes en el plano, apoyos como
cubos verdes y cargas como flechas naranja.

No es un editor: solo visualizacion. El editor 2D plano sigue siendo
canvas_editor.py / pg_canvas.py para crear modelos comodamente. Esta
vista 3D se conecta al MISMO modelo y refleja sus cambios.

Modos de visualizacion soportados:
  - Malla original (nodos + aristas + caras)
  - Apoyos (cubos en nodos restringidos)
  - Cargas (flechas en nodos cargados)
  - Etiquetas de nodos / elementos (opcional)
  - Deformada superpuesta (segun displacements y escala)
  - Mapa de esfuerzos (color por elemento segun sigma seleccionado)
"""
from __future__ import annotations
from typing import Optional
import numpy as np

from PySide6.QtCore import Qt, QObject, Signal, QPointF
from PySide6.QtGui import QColor, QVector3D, QVector4D, QFont
from PySide6.QtWidgets import QWidget, QFrame, QLabel, QVBoxLayout

import pyqtgraph as pg
import pyqtgraph.opengl as gl

from ..fem.structure import Structure
from ..fem.node import Node
from ..fem.q4_element import Q4Element


# Paleta coherente con el resto del aplicativo
_C_NODE     = (0.18, 0.20, 0.24, 1.0)
_C_EDGE     = (0.30, 0.32, 0.36, 1.0)
_C_FACE     = (0.55, 0.72, 0.92, 0.35)
_C_EDGE_DEF = (0.86, 0.20, 0.20, 1.0)
_C_FACE_DEF = (0.96, 0.55, 0.20, 0.30)
_C_BC       = (0.10, 0.55, 0.10, 1.0)
_C_LOAD     = (0.95, 0.40, 0.10, 1.0)
_C_BG       = "#FAFAFC"


def _q4_triangles(idx4: list[int]) -> np.ndarray:
    """Convierte un quad (4 indices) en 2 triangulos para GLMeshItem."""
    a, b, c, d = idx4
    return np.array([[a, b, c], [a, c, d]], dtype=np.int32)


class _InfoPopup(QFrame):
    """Panel flotante con la informacion del objeto clickeado en la vista 3D.

    Para que sirve: al hacer clic sobre un nodo o sobre el area de un
    elemento, este panel aparece junto al cursor con los datos del objeto
    (coordenadas, apoyos, cargas, material, esfuerzos...). Se cierra solo
    al hacer clic en otro lado, orbitar la camara o hacer zoom.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        # Los clics lo atraviesan: es el canvas quien decide cuando cerrarlo
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setStyleSheet(
            "QFrame { background: rgba(255, 255, 255, 243);"
            " border: 1px solid #8E99A5; border-radius: 6px; }"
            "QLabel { border: none; background: transparent;"
            " color: #22262B; font-size: 9pt; }"
        )
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 8, 10, 8)
        self._label = QLabel("")
        self._label.setTextFormat(Qt.TextFormat.RichText)
        lay.addWidget(self._label)
        self.hide()

    def show_at(self, pos: QPointF, html: str) -> None:
        """Muestra el panel junto al cursor, sin salirse del canvas."""
        self._label.setText(html)
        self.adjustSize()
        x, y = pos.x() + 14, pos.y() + 14
        parent = self.parentWidget()
        if parent is not None:
            x = min(x, max(0, parent.width() - self.width() - 4))
            y = min(y, max(0, parent.height() - self.height() - 4))
        self.move(int(x), int(y))
        self.show()
        self.raise_()


class Canvas3DQ4(gl.GLViewWidget):
    """Vista 3D del modelo Q4 (placa plana en perspectiva).

    API publica:
      - set_structure(structure)
      - set_displacements(u_array)
      - set_scale(factor)
      - set_show_undeformed(bool)
      - set_show_deformed(bool)
      - set_show_faces(bool)
      - set_show_labels(bool)
      - set_show_supports(bool)
      - set_show_loads(bool)
      - fit_view()
      - set_stress_field(name, per_element_values, vmin, vmax)
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setBackgroundColor(QColor(_C_BG))
        # Camara: vista isometrica suave
        self.setCameraPosition(distance=3.0, elevation=28, azimuth=35)

        # Items persistentes (ejes + grilla)
        self._static_items: list = []
        # Items dinamicos (se borran y recrean en cada refresh)
        self._dyn_items: list = []
        # Etiquetas de nodos (GLTextItem)
        self._label_items: list = []

        self._structure: Optional[Structure] = None
        self._displacements: Optional[np.ndarray] = None
        self._scale: float = 1.0

        self._show_undeformed = True
        self._show_deformed = True
        self._show_faces = True
        self._show_labels = False
        self._show_supports = True
        self._show_loads = True

        # Mapa de esfuerzos por elemento (opcional)
        self._stress_field_name: Optional[str] = None
        self._stress_per_element: Optional[np.ndarray] = None  # shape (n_elem,)
        self._stress_vmin: float = 0.0
        self._stress_vmax: float = 1.0

        # Picking: panel de informacion junto al cursor + posicion del
        # clic inicial (para distinguir CLIC de ARRASTRE de camara)
        self._popup = _InfoPopup(self)
        self._press_pos: Optional[QPointF] = None

        self._add_axes(length=1.0)
        self._add_grid()

    # ------------------------------------------------------------------ ejes + grilla
    def _add_axes(self, length: float = 1.0) -> None:
        """Dibuja los 3 ejes con su ETIQUETA (X roja, Y verde, Z azul)."""
        font = QFont("Arial", 11)
        font.setBold(True)
        for end, color, rgba_lbl, lbl in [
            ((length, 0, 0), (0.90, 0.18, 0.18, 1), (230, 46, 46, 255), "X"),
            ((0, length, 0), (0.18, 0.70, 0.18, 1), (40, 165, 40, 255), "Y"),
            ((0, 0, length * 0.5), (0.18, 0.32, 0.92, 1), (46, 82, 235, 255), "Z"),
        ]:
            ln = gl.GLLinePlotItem(
                pos=np.array([[0, 0, 0], list(end)], dtype=np.float64),
                color=color, width=2.5, antialias=True,
            )
            ln.setGLOptions("opaque")
            self.addItem(ln)
            self._static_items.append(ln)
            # Etiqueta del eje un poco mas alla de la punta, en su color
            try:
                tip = (end[0] * 1.10, end[1] * 1.10, end[2] * 1.16 + (0.04 if lbl == "Z" else 0))
                t = gl.GLTextItem(pos=tip, text=lbl, color=rgba_lbl, font=font)
                self.addItem(t)
                self._static_items.append(t)
            except Exception:
                pass   # GLTextItem no disponible en pyqtgraph muy antiguos

    def _add_grid(self) -> None:
        grid = gl.GLGridItem()
        grid.setSize(x=6, y=6)
        grid.setSpacing(x=0.5, y=0.5)
        grid.setColor((180, 180, 188, 130))
        self.addItem(grid)
        self._static_items.append(grid)

    # ------------------------------------------------------------------ API
    def set_structure(self, structure: Optional[Structure],
                      displacements: Optional[np.ndarray] = None) -> None:
        self._structure = structure
        self._displacements = displacements
        self._refresh()

    def set_displacements(self, displacements: Optional[np.ndarray]) -> None:
        self._displacements = displacements
        self._refresh()

    def set_scale(self, scale: float) -> None:
        self._scale = float(scale)
        self._refresh()

    def set_show_undeformed(self, on: bool) -> None:
        self._show_undeformed = bool(on); self._refresh()

    def set_show_deformed(self, on: bool) -> None:
        self._show_deformed = bool(on); self._refresh()

    def set_show_faces(self, on: bool) -> None:
        self._show_faces = bool(on); self._refresh()

    def set_show_labels(self, on: bool) -> None:
        self._show_labels = bool(on); self._refresh()

    def set_show_supports(self, on: bool) -> None:
        self._show_supports = bool(on); self._refresh()

    def set_show_loads(self, on: bool) -> None:
        self._show_loads = bool(on); self._refresh()

    def set_stress_field(self, name: Optional[str],
                         per_element_values: Optional[np.ndarray],
                         vmin: float = 0.0, vmax: float = 1.0) -> None:
        self._stress_field_name = name
        self._stress_per_element = per_element_values
        self._stress_vmin = float(vmin)
        self._stress_vmax = float(vmax)
        self._refresh()

    def fit_view(self) -> None:
        if self._structure is None or not self._structure.nodes:
            return
        pts = np.array([[n.x, n.y, 0.0] for n in self._structure.nodes])
        mn = pts.min(axis=0); mx = pts.max(axis=0)
        center = (mn + mx) / 2.0
        size = float(np.linalg.norm(mx - mn))
        self.opts["center"] = QVector3D(*center)
        self.setCameraPosition(distance=max(size * 2.0, 1.5), elevation=28, azimuth=35)
        self.update()

    # ------------------------------------------------------------------ picking
    # Al hacer CLIC sobre un nodo o sobre el area de un elemento se muestra
    # un panel flotante con su informacion junto al cursor. Un ARRASTRE
    # (orbitar la camara) no dispara el panel: se distingue midiendo cuanto
    # se movio el mouse entre presionar y soltar.
    def mousePressEvent(self, ev) -> None:
        self._popup.hide()
        self._press_pos = ev.position()
        super().mousePressEvent(ev)

    def mouseReleaseEvent(self, ev) -> None:
        super().mouseReleaseEvent(ev)
        if self._press_pos is None:
            return
        moved = (ev.position() - self._press_pos).manhattanLength()
        self._press_pos = None
        if ev.button() == Qt.MouseButton.LeftButton and moved < 6.0:
            self._pick(ev.position())

    def wheelEvent(self, ev) -> None:
        # El zoom mueve la escena: el panel quedaria "flotando" en un lugar
        # que ya no corresponde, mejor cerrarlo.
        self._popup.hide()
        super().wheelEvent(ev)

    def _project_point(self, x: float, y: float, z: float):
        """Proyecta un punto 3D del modelo a pixeles del widget.

        Usa las mismas matrices de camara del render (proyeccion y vista):
        punto 3D -> coordenadas normalizadas (NDC) -> pixeles. Devuelve
        (px, py) o None si el punto queda detras de la camara.
        """
        # La firma de projectionMatrix cambia entre versiones de pyqtgraph:
        # 0.14+ exige (region, viewport); versiones previas no llevan args.
        try:
            vp = self.getViewport()
            proj = self.projectionMatrix(vp, vp)
        except (TypeError, AttributeError):
            proj = self.projectionMatrix()
        m = proj * self.viewMatrix()
        v = m.map(QVector4D(x, y, z, 1.0))
        w = v.w()
        if abs(w) < 1e-12:
            return None
        ndc_x, ndc_y, ndc_z = v.x() / w, v.y() / w, v.z() / w
        if not (-1.0 <= ndc_z <= 1.0):
            return None
        px = (ndc_x + 1.0) * 0.5 * self.width()
        py = (1.0 - ndc_y) * 0.5 * self.height()
        return px, py

    @staticmethod
    def _point_in_polygon(px: float, py: float,
                          poly: list[tuple[float, float]]) -> bool:
        """Test par-impar: True si (px, py) cae dentro del poligono 2D."""
        inside = False
        n = len(poly)
        j = n - 1
        for i in range(n):
            xi, yi = poly[i]
            xj, yj = poly[j]
            if (yi > py) != (yj > py):
                x_cross = xi + (py - yi) * (xj - xi) / (yj - yi)
                if px < x_cross:
                    inside = not inside
            j = i
        return inside

    def _pick(self, pos: QPointF) -> None:
        """Determina que objeto esta bajo el cursor y muestra su informacion.

        Prioridad: primero nodos (radio de 14 px alrededor del clic),
        despues el area de los elementos (poligono proyectado en pantalla).
        """
        s = self._structure
        if s is None or not s.nodes:
            return
        px, py = pos.x(), pos.y()

        # 1) Nodos: el mas cercano al clic dentro del radio de tolerancia
        best_node, best_d = None, 14.0
        for n in s.nodes:
            pr = self._project_point(n.x, n.y, 0.0)
            if pr is None:
                continue
            d = ((pr[0] - px) ** 2 + (pr[1] - py) ** 2) ** 0.5
            if d < best_d:
                best_d, best_node = d, n
        if best_node is not None:
            self._popup.show_at(pos, self._node_html(best_node))
            return

        # 2) Elementos: el clic cae dentro del contorno proyectado
        for idx, el in enumerate(s.elements):
            poly = []
            visible = True
            for n in el.nodes:
                pr = self._project_point(n.x, n.y, 0.0)
                if pr is None:
                    visible = False
                    break
                poly.append(pr)
            if visible and self._point_in_polygon(px, py, poly):
                self._popup.show_at(pos, self._element_html(el, idx))
                return

    @staticmethod
    def _fmt(v: float) -> str:
        """Formato de numeros para el panel: 15 digitos, ceros residuales -> 0."""
        if abs(v) < 1e-13:
            return "0"
        return f"{v:.15g}"

    def _node_html(self, n) -> str:
        """Arma la ficha HTML de un nodo: posicion, apoyo, cargas y u."""
        filas = [
            f"<b>Nodo N{n.id + 1}</b>",
            f"x = {self._fmt(n.x)} m &nbsp;&nbsp; y = {self._fmt(n.y)} m",
        ]
        restr = []
        if n.restraint_x:
            restr.append("X")
        if n.restraint_y:
            restr.append("Y")
        filas.append("Apoyo: " + (", ".join(restr) if restr else "libre"))
        if n.load_x != 0.0 or n.load_y != 0.0:
            filas.append(
                f"Carga: Fx = {self._fmt(n.load_x)} N, "
                f"Fy = {self._fmt(n.load_y)} N"
            )
        if self._displacements is not None and 2 * n.id + 1 < len(self._displacements):
            ux = float(self._displacements[2 * n.id])
            uy = float(self._displacements[2 * n.id + 1])
            filas.append(f"ux = {self._fmt(ux)} m")
            filas.append(f"uy = {self._fmt(uy)} m")
        return "<br/>".join(filas)

    def _element_html(self, el, idx: int) -> str:
        """Arma la ficha HTML de un elemento: nodos, area, material y esfuerzo."""
        node_names = ", ".join(f"N{n.id + 1}" for n in el.nodes)
        # Area del cuadrilatero por la formula del lazo (shoelace)
        xs = [n.x for n in el.nodes]
        ys = [n.y for n in el.nodes]
        area = 0.0
        for i in range(4):
            j = (i + 1) % 4
            area += xs[i] * ys[j] - xs[j] * ys[i]
        area = abs(area) / 2.0
        hipotesis = "tensión plana" if el.plane_stress else "deformación plana"
        filas = [
            f"<b>Elemento E{el.id + 1}</b>",
            f"Nodos: {node_names}",
            f"Área = {self._fmt(area)} m²",
            f"E = {self._fmt(el.E)} Pa &nbsp; ν = {self._fmt(el.nu)}",
            f"t = {self._fmt(el.t)} m &nbsp; ({hipotesis})",
        ]
        if (self._stress_field_name is not None
                and self._stress_per_element is not None
                and idx < len(self._stress_per_element)):
            filas.append(
                f"{self._stress_field_name} (promedio) = "
                f"{self._fmt(float(self._stress_per_element[idx]))} Pa"
            )
        return "<br/>".join(filas)

    # ------------------------------------------------------------------ helpers internos
    def _node_array(self, with_def: bool = False) -> np.ndarray:
        """Arreglo (N,3) de nodos del modelo. with_def aplica deformacion."""
        s = self._structure
        if s is None or not s.nodes:
            return np.zeros((0, 3))
        pts = np.array([[n.x, n.y, 0.0] for n in s.nodes])
        if with_def and self._displacements is not None and self._scale != 0.0:
            # Q4 plano: solo ux, uy (no uz). El plano sigue plano.
            d = np.zeros_like(pts)
            d[:, 0] = self._displacements[0::2] * self._scale
            d[:, 1] = self._displacements[1::2] * self._scale
            pts = pts + d
        return pts

    def _element_quads(self) -> list[list[int]]:
        s = self._structure
        if s is None:
            return []
        return [[n.id for n in el.nodes] for el in s.elements]

    def _stress_to_color(self, value: float) -> tuple[float, float, float, float]:
        """Mapeo simple azul -> rojo entre vmin y vmax."""
        if self._stress_vmax <= self._stress_vmin:
            t = 0.5
        else:
            t = (value - self._stress_vmin) / (self._stress_vmax - self._stress_vmin)
            t = max(0.0, min(1.0, t))
        # azul (0,0,1) -> verde (0,1,0) -> rojo (1,0,0), suave
        if t < 0.5:
            s = t * 2.0
            return (0.0, s, 1.0 - s, 0.55)
        else:
            s = (t - 0.5) * 2.0
            return (s, 1.0 - s, 0.0, 0.55)

    # ------------------------------------------------------------------ refresh
    def _refresh(self) -> None:
        # Borrar items dinamicos previos
        for it in self._dyn_items + self._label_items:
            try:
                self.removeItem(it)
            except Exception:
                pass
        self._dyn_items.clear()
        self._label_items.clear()

        s = self._structure
        if s is None or not s.nodes:
            return

        pts_ref = self._node_array(with_def=False)
        quads = self._element_quads()

        # ---------- malla original ----------
        if self._show_undeformed:
            # Nodos como puntos
            sc = gl.GLScatterPlotItem(pos=pts_ref, color=_C_NODE, size=10, pxMode=True)
            sc.setGLOptions("translucent")
            self.addItem(sc); self._dyn_items.append(sc)

            # Aristas (4 por elemento)
            if quads:
                edges = []
                for q in quads:
                    for i in range(4):
                        edges.append([q[i], q[(i + 1) % 4]])
                edges = np.array(edges, dtype=np.int32)
                edge_pts = np.empty((edges.shape[0] * 2, 3))
                edge_pts[0::2] = pts_ref[edges[:, 0]]
                edge_pts[1::2] = pts_ref[edges[:, 1]]
                ln = gl.GLLinePlotItem(
                    pos=edge_pts, color=_C_EDGE, width=1.8,
                    antialias=True, mode="lines",
                )
                self.addItem(ln); self._dyn_items.append(ln)

            # Caras semitransparentes — coloreadas por esfuerzo si esta activo
            if self._show_faces and quads:
                if self._stress_per_element is not None and len(self._stress_per_element) == len(quads):
                    # Una GLMeshItem por elemento (para colorear cada uno)
                    for i, q in enumerate(quads):
                        tris = _q4_triangles(q)
                        col = self._stress_to_color(float(self._stress_per_element[i]))
                        m = gl.GLMeshItem(
                            vertexes=pts_ref, faces=tris,
                            color=col, drawEdges=False, smooth=False,
                            glOptions="translucent",
                        )
                        self.addItem(m); self._dyn_items.append(m)
                else:
                    tris_all = np.vstack([_q4_triangles(q) for q in quads])
                    mesh = gl.GLMeshItem(
                        vertexes=pts_ref, faces=tris_all,
                        color=_C_FACE, drawEdges=False, smooth=False,
                        glOptions="translucent",
                    )
                    self.addItem(mesh); self._dyn_items.append(mesh)

        # ---------- deformada ----------
        if self._show_deformed and self._displacements is not None and self._scale != 0.0:
            pts_def = self._node_array(with_def=True)
            if quads:
                edges = []
                for q in quads:
                    for i in range(4):
                        edges.append([q[i], q[(i + 1) % 4]])
                edges = np.array(edges, dtype=np.int32)
                edge_pts = np.empty((edges.shape[0] * 2, 3))
                edge_pts[0::2] = pts_def[edges[:, 0]]
                edge_pts[1::2] = pts_def[edges[:, 1]]
                ln = gl.GLLinePlotItem(
                    pos=edge_pts, color=_C_EDGE_DEF, width=2.2,
                    antialias=True, mode="lines",
                )
                self.addItem(ln); self._dyn_items.append(ln)
                if self._show_faces:
                    tris_all = np.vstack([_q4_triangles(q) for q in quads])
                    mesh = gl.GLMeshItem(
                        vertexes=pts_def, faces=tris_all,
                        color=_C_FACE_DEF, drawEdges=False, smooth=False,
                        glOptions="translucent",
                    )
                    self.addItem(mesh); self._dyn_items.append(mesh)

        # ---------- apoyos ----------
        if self._show_supports:
            size_support = self._geom_scale(s) * 0.04
            for n in s.nodes:
                if n.restraint_x or n.restraint_y:
                    cube = self._make_support_cube(n.x, n.y, 0.0, size=size_support)
                    self.addItem(cube); self._dyn_items.append(cube)

        # ---------- cargas ----------
        if self._show_loads:
            scale_load = self._geom_scale(s) * 0.35
            for n in s.nodes:
                if abs(n.load_x) > 0 or abs(n.load_y) > 0:
                    arr = self._make_load_arrow(n.x, n.y, 0.0,
                                                n.load_x, n.load_y, 0.0,
                                                length_max=scale_load)
                    if arr is not None:
                        self.addItem(arr); self._dyn_items.append(arr)

        # ---------- etiquetas ----------
        if self._show_labels and len(s.nodes) <= 60:
            font = QFont("Arial", 9)
            for n in s.nodes:
                try:
                    t = gl.GLTextItem(pos=(n.x, n.y, 0.02), text=f"N{n.id + 1}",
                                      color=(40, 40, 50, 255), font=font)
                    self.addItem(t); self._label_items.append(t)
                except Exception:
                    pass

    def _geom_scale(self, s: Structure) -> float:
        """Escala caracteristica del modelo para dimensionar adornos."""
        pts = np.array([[n.x, n.y] for n in s.nodes])
        mn = pts.min(axis=0); mx = pts.max(axis=0)
        d = np.linalg.norm(mx - mn)
        return float(d) if d > 1e-9 else 1.0

    def _make_support_cube(self, x, y, z, size=0.05):
        s = size / 2.0
        verts = np.array([
            [x - s, y - s, z - s], [x + s, y - s, z - s],
            [x + s, y + s, z - s], [x - s, y + s, z - s],
            [x - s, y - s, z + s], [x + s, y - s, z + s],
            [x + s, y + s, z + s], [x - s, y + s, z + s],
        ])
        faces = np.array([
            [0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6],
            [1, 2, 6], [1, 6, 5], [0, 3, 7], [0, 7, 4],
        ])
        return gl.GLMeshItem(
            vertexes=verts, faces=faces,
            color=_C_BC, drawEdges=False, smooth=False, glOptions="opaque",
        )

    def _make_load_arrow(self, x, y, z, fx, fy, fz, length_max=0.3):
        mag = float(np.sqrt(fx * fx + fy * fy + fz * fz))
        if mag < 1e-30:
            return None
        ux, uy, uz = fx / mag, fy / mag, fz / mag
        end = (x + ux * length_max, y + uy * length_max, z + uz * length_max)
        line = gl.GLLinePlotItem(
            pos=np.array([[x, y, z], end]),
            color=_C_LOAD, width=3.5, antialias=True,
        )
        line.setGLOptions("opaque")
        return line
