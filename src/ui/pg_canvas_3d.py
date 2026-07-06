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

from PySide6.QtCore import Qt, QObject, Signal
from PySide6.QtGui import QColor, QVector3D, QFont
from PySide6.QtWidgets import QWidget

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

        self._add_axes(length=1.0)
        self._add_grid()

    # ------------------------------------------------------------------ ejes + grilla
    def _add_axes(self, length: float = 1.0) -> None:
        for end, color, lbl in [
            ((length, 0, 0), (0.90, 0.18, 0.18, 1), "X"),
            ((0, length, 0), (0.18, 0.70, 0.18, 1), "Y"),
            ((0, 0, length * 0.5), (0.18, 0.32, 0.92, 1), "Z"),
        ]:
            ln = gl.GLLinePlotItem(
                pos=np.array([[0, 0, 0], list(end)], dtype=np.float64),
                color=color, width=2.5, antialias=True,
            )
            ln.setGLOptions("opaque")
            self.addItem(ln)
            self._static_items.append(ln)

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
