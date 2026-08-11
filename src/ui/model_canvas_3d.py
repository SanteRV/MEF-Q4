"""Lienzo 3D de modelado — Corrección 2, paso 2 y 3.

Para qué sirve: es la ÚNICA ventana de modelo del aplicativo. El usuario
ve su grilla de coordenadas, gira libremente en el espacio 3D, cambia a
las vistas de planos XY / XZ / YZ, y dibuja sobre la grilla los elementos
que necesite (frame, plane, plate o shell), seleccionando antes el tipo y
la sección.

Cómo se dibuja en 3D con un mouse 2D: el clic se convierte en un rayo que
sale de la cámara (des-proyección con las matrices reales del render) y se
intersecta con el PLANO DE TRABAJO activo (por ejemplo "XY en Z = 0"). El
punto obtenido se ajusta a la intersección de grilla más cercana.

Interacción:
    - Arrastrar con el botón izquierdo  -> orbitar la cámara
    - Clic (sin arrastrar)              -> actuar según el modo activo
    - Ctrl + clic en modo Seleccionar   -> añadir/quitar de la selección
    - Esc                               -> cancelar el elemento en curso
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

import numpy as np
from PySide6.QtCore import Qt, QPointF, Signal
from PySide6.QtGui import QColor, QVector3D, QVector4D, QFont
from PySide6.QtWidgets import QWidget
import pyqtgraph.opengl as gl

from ..fem.grid import GridSystem
from ..fem.model import Model


class Mode(Enum):
    """Modos de trabajo del lienzo."""
    SELECT = "Seleccionar"
    NODE = "Nodo"
    FRAME = "Frame (viga/columna)"
    AREA = "Área (plane/plate/shell)"


# Paleta coherente con el resto del aplicativo
_C_GRID = (0.62, 0.66, 0.72, 0.55)
_C_NODE = (0.16, 0.18, 0.22, 1.0)
_C_NODE_SEL = (0.95, 0.45, 0.10, 1.0)
_C_FRAME = (0.14, 0.31, 0.49, 1.0)
_C_AREA_EDGE = (0.20, 0.42, 0.62, 1.0)
_C_AREA_FACE = (0.55, 0.72, 0.92, 0.42)
_C_PENDING = (0.95, 0.45, 0.10, 1.0)
_C_SUPPORT = (0.10, 0.55, 0.10, 1.0)
_C_LOAD = (0.85, 0.20, 0.15, 1.0)


class ModelCanvas3D(gl.GLViewWidget):
    """Vista 3D editable del modelo unificado."""

    node_clicked = Signal(int)          # id de nodo pulsado
    selection_changed = Signal(list)    # ids de nodos seleccionados
    model_changed = Signal()            # se creó o borró algo
    status_message = Signal(str)        # texto para la barra de estado

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setBackgroundColor(QColor("#FAFAFC"))

        self.model: Model = Model()
        self.grid: GridSystem = GridSystem()
        self.mode: Mode = Mode.SELECT

        # Plano de trabajo: ('xy'|'xz'|'yz', coordenada del plano)
        self.work_plane: tuple[str, float] = ("xy", 0.0)
        # Tipo y sección con los que se dibuja
        self.active_area_type: str = "shell"
        self.active_area_section: str = ""
        self.active_frame_section: str = ""

        self.selected: list[int] = []
        self._pending: list[int] = []       # nodos del elemento en curso
        self._press_pos: Optional[QPointF] = None
        self._items: list = []              # items dinámicos (modelo)
        self._grid_items: list = []         # items de la grilla

        self.setCameraPosition(distance=18.0, elevation=24, azimuth=-55)
        self._add_axes()
        self.redraw()

    # ------------------------------------------------------------ ejes
    def _add_axes(self) -> None:
        """Ejes X (rojo), Y (verde), Z (azul) con su etiqueta."""
        font = QFont("Arial", 11)
        font.setBold(True)
        L = 1.6
        for end, color, rgba, lbl in [
            ((L, 0, 0), (0.90, 0.18, 0.18, 1), (230, 46, 46, 255), "X"),
            ((0, L, 0), (0.18, 0.70, 0.18, 1), (40, 165, 40, 255), "Y"),
            ((0, 0, L), (0.18, 0.32, 0.92, 1), (46, 82, 235, 255), "Z"),
        ]:
            ln = gl.GLLinePlotItem(
                pos=np.array([[0, 0, 0], list(end)], dtype=float),
                color=color, width=2.5, antialias=True)
            ln.setGLOptions("opaque")
            self.addItem(ln)
            try:
                t = gl.GLTextItem(
                    pos=(end[0] * 1.12, end[1] * 1.12, end[2] * 1.12),
                    text=lbl, color=rgba, font=font)
                self.addItem(t)
            except Exception:
                pass

    # ------------------------------------------------------------- API
    def set_model(self, model: Model) -> None:
        self.model = model
        self.selected.clear()
        self._pending.clear()
        self.redraw()

    def set_grid(self, grid: GridSystem) -> None:
        self.grid = grid
        # Si el plano de trabajo quedó fuera de la grilla, recolocarlo
        kind, _ = self.work_plane
        eje = {"xy": grid.z, "xz": grid.y, "yz": grid.x}[kind]
        if eje:
            self.work_plane = (kind, eje[0])
        self.redraw()
        self.fit_view()

    def set_mode(self, mode: Mode) -> None:
        self.mode = mode
        self._pending.clear()
        cursores = {
            Mode.SELECT: Qt.CursorShape.ArrowCursor,
            Mode.NODE: Qt.CursorShape.CrossCursor,
            Mode.FRAME: Qt.CursorShape.CrossCursor,
            Mode.AREA: Qt.CursorShape.CrossCursor,
        }
        self.setCursor(cursores[mode])
        self.status_message.emit(self._hint())
        self.redraw()

    def set_work_plane(self, kind: str, coord: float) -> None:
        """Fija el plano sobre el que se dibuja ('xy', 'xz' o 'yz')."""
        if kind not in ("xy", "xz", "yz"):
            raise ValueError("El plano debe ser 'xy', 'xz' o 'yz'.")
        self.work_plane = (kind, float(coord))
        self.status_message.emit(self._hint())
        self.redraw()

    def set_view(self, preset: str) -> None:
        """Vistas rápidas: '3d', 'xy' (planta), 'xz' o 'yz' (elevaciones)."""
        vistas = {
            "3d": (24, -55),
            "xy": (89.9, -90),     # planta: mirando desde +Z
            "xz": (0, -90),        # elevación frontal: desde -Y
            "yz": (0, 0),          # elevación lateral: desde +X
        }
        if preset not in vistas:
            raise ValueError("Vista debe ser '3d', 'xy', 'xz' o 'yz'.")
        elev, azim = vistas[preset]
        self.setCameraPosition(elevation=elev, azimuth=azim)
        self.update()

    def fit_view(self) -> None:
        """Encuadra la grilla y el modelo completos."""
        pts = [self.grid.intersections()]
        if self.model.nodes:
            pts.append(np.array([[n.x, n.y, n.z] for n in self.model.nodes]))
        todos = np.vstack([p for p in pts if p.size])
        if todos.size == 0:
            return
        mn, mx = todos.min(axis=0), todos.max(axis=0)
        centro = (mn + mx) / 2.0
        size = float(np.linalg.norm(mx - mn)) or 1.0
        self.opts["center"] = QVector3D(*centro)
        self.setCameraPosition(distance=max(size * 1.7, 3.0))
        self.update()

    def clear_selection(self) -> None:
        self.selected.clear()
        self.selection_changed.emit([])
        self.redraw()

    def select_all_nodes(self) -> None:
        self.selected = [n.id for n in self.model.nodes]
        self.selection_changed.emit(list(self.selected))
        self.redraw()

    def _hint(self) -> str:
        """Texto de ayuda del modo actual, para la barra de estado."""
        kind, coord = self.work_plane
        plano = f"{kind.upper()} en {'XYZ'[{'xy': 2, 'xz': 1, 'yz': 0}[kind]]} = {coord:g}"
        if self.mode is Mode.SELECT:
            return f"Seleccionar — clic en un nodo, Ctrl+clic para varios. Plano: {plano}"
        if self.mode is Mode.NODE:
            return f"Nodo — clic sobre la grilla para crearlo. Plano: {plano}"
        if self.mode is Mode.FRAME:
            return f"Frame — clic en 2 puntos (nodo inicial y final). Plano: {plano}"
        return f"Área — clic en 4 puntos en sentido antihorario. Plano: {plano}"

    # --------------------------------------------------- geometría de clic
    def _matrices(self):
        """Matriz proyección·vista, con shim para pyqtgraph 0.14."""
        try:
            vp = self.getViewport()
            proj = self.projectionMatrix(vp, vp)
        except (TypeError, AttributeError):
            proj = self.projectionMatrix()
        return proj * self.viewMatrix()

    def _project_point(self, x: float, y: float, z: float):
        """Punto 3D -> píxeles del widget; None si queda detrás de la cámara."""
        v = self._matrices().map(QVector4D(x, y, z, 1.0))
        w = v.w()
        if abs(w) < 1e-12:
            return None
        ndc = (v.x() / w, v.y() / w, v.z() / w)
        if not (-1.0 <= ndc[2] <= 1.0):
            return None
        return ((ndc[0] + 1.0) * 0.5 * self.width(),
                (1.0 - ndc[1]) * 0.5 * self.height())

    def _ray_from_screen(self, pos: QPointF):
        """Rayo (origen, dirección) en el mundo que pasa por el píxel dado."""
        m = self._matrices()
        inv, ok = m.inverted()
        if not ok:
            return None
        ndc_x = 2.0 * pos.x() / max(self.width(), 1) - 1.0
        ndc_y = 1.0 - 2.0 * pos.y() / max(self.height(), 1)
        puntos = []
        for ndc_z in (-1.0, 1.0):
            v = inv.map(QVector4D(ndc_x, ndc_y, ndc_z, 1.0))
            if abs(v.w()) < 1e-12:
                return None
            puntos.append(np.array([v.x() / v.w(), v.y() / v.w(), v.z() / v.w()]))
        origen, lejos = puntos
        d = lejos - origen
        n = np.linalg.norm(d)
        if n < 1e-12:
            return None
        return origen, d / n

    def _point_on_work_plane(self, pos: QPointF, snap_px: float = 16.0):
        """Intersección del rayo del clic con el plano de trabajo, con snap.

        El ajuste se mide en PÍXELES, no en metros: el usuario percibe la
        cercanía en pantalla, así que a mucho zoom el snap debe ser más
        fino en el mundo y a poco zoom más grueso. Solo compiten las
        intersecciones que están sobre el plano de trabajo activo.
        """
        rayo = self._ray_from_screen(pos)
        if rayo is None:
            return None
        origen, direccion = rayo
        kind, coord = self.work_plane
        eje = {"yz": 0, "xz": 1, "xy": 2}[kind]
        normal = np.zeros(3)
        normal[eje] = 1.0
        denom = float(np.dot(normal, direccion))
        if abs(denom) < 1e-9:      # el rayo es paralelo al plano
            return None
        t = (coord - float(np.dot(normal, origen))) / denom
        if t <= 0:
            return None
        p = origen + t * direccion
        libre = (float(p[0]), float(p[1]), float(p[2]))
        if not self.grid.snap:
            return libre

        candidatos = self.grid.points_on_plane(kind, coord)
        mejor, mejor_d = None, snap_px
        for c in candidatos:
            pr = self._project_point(float(c[0]), float(c[1]), float(c[2]))
            if pr is None:
                continue
            d = ((pr[0] - pos.x()) ** 2 + (pr[1] - pos.y()) ** 2) ** 0.5
            if d < mejor_d:
                mejor, mejor_d = c, d
        if mejor is not None:
            return (float(mejor[0]), float(mejor[1]), float(mejor[2]))
        return libre

    def _node_at(self, pos: QPointF, tol: float = 14.0) -> Optional[int]:
        """Id del nodo más cercano al clic dentro de la tolerancia en píxeles."""
        mejor, mejor_d = None, tol
        for n in self.model.nodes:
            pr = self._project_point(n.x, n.y, n.z)
            if pr is None:
                continue
            d = ((pr[0] - pos.x()) ** 2 + (pr[1] - pos.y()) ** 2) ** 0.5
            if d < mejor_d:
                mejor, mejor_d = n.id, d
        return mejor

    # ------------------------------------------------------------ eventos
    def mousePressEvent(self, ev) -> None:
        self._press_pos = ev.position()
        super().mousePressEvent(ev)

    def mouseReleaseEvent(self, ev) -> None:
        super().mouseReleaseEvent(ev)
        if self._press_pos is None:
            return
        movido = (ev.position() - self._press_pos).manhattanLength()
        self._press_pos = None
        if ev.button() != Qt.MouseButton.LeftButton or movido >= 6.0:
            return          # fue un arrastre: la cámara ya orbitó
        self._click(ev.position(),
                    bool(ev.modifiers() & Qt.KeyboardModifier.ControlModifier))

    def keyPressEvent(self, ev) -> None:
        if ev.key() == Qt.Key.Key_Escape:
            if self._pending:
                self._pending.clear()
                self.status_message.emit("Elemento cancelado.")
            else:
                self.clear_selection()
            self.redraw()
            return
        super().keyPressEvent(ev)

    def _click(self, pos: QPointF, ctrl: bool) -> None:
        """Resuelve un clic según el modo activo."""
        if self.mode is Mode.SELECT:
            nid = self._node_at(pos)
            if nid is None:
                if not ctrl:
                    self.clear_selection()
                return
            if ctrl:
                if nid in self.selected:
                    self.selected.remove(nid)
                else:
                    self.selected.append(nid)
            else:
                self.selected = [nid]
            self.node_clicked.emit(nid)
            self.selection_changed.emit(list(self.selected))
            self.redraw()
            return

        # Modos de dibujo: el clic define un punto sobre el plano de trabajo
        nid = self._node_at(pos)
        if nid is None:
            p = self._point_on_work_plane(pos)
            if p is None:
                self.status_message.emit(
                    "El clic no corta el plano de trabajo. Gire la vista o "
                    "cambie de plano.")
                return
            nid = self.model.add_node(*p).id

        if self.mode is Mode.NODE:
            self.model_changed.emit()
            self.status_message.emit(f"Nodo N{nid + 1} creado.")
            self.redraw()
            return

        self._pending.append(nid)
        requeridos = 2 if self.mode is Mode.FRAME else 4
        if len(self._pending) < requeridos:
            self.status_message.emit(
                f"{len(self._pending)} de {requeridos} nodos — Esc cancela.")
            self.redraw()
            return

        nodos = list(self._pending)
        self._pending.clear()
        try:
            if self.mode is Mode.FRAME:
                if nodos[0] == nodos[1]:
                    raise ValueError("Un frame necesita dos nodos distintos.")
                m = self.model.add_member("frame", nodos,
                                          self.active_frame_section)
            else:
                if len(set(nodos)) != 4:
                    raise ValueError("Un área necesita cuatro nodos distintos.")
                m = self.model.add_member(self.active_area_type, nodos,
                                          self.active_area_section)
        except Exception as e:
            self.status_message.emit(f"No se pudo crear el elemento: {e}")
            self.redraw()
            return
        self.model_changed.emit()
        self.status_message.emit(
            f"Elemento {m.tipo} E{m.id + 1} creado con sección "
            f"{m.section!r}.")
        self.redraw()

    # ------------------------------------------------------------ dibujo
    def redraw(self) -> None:
        """Redibuja grilla y modelo completos."""
        for it in self._items + self._grid_items:
            try:
                self.removeItem(it)
            except Exception:
                pass
        self._items.clear()
        self._grid_items.clear()
        self._draw_grid()
        self._draw_model()
        self.update()

    def _add(self, item, grid: bool = False) -> None:
        self.addItem(item)
        (self._grid_items if grid else self._items).append(item)

    def _draw_grid(self) -> None:
        if not self.grid.visible:
            return
        segs = self.grid.lines()
        if not segs:
            return
        pos = np.array([c for s in segs for c in s], dtype=float)
        ln = gl.GLLinePlotItem(pos=pos, color=_C_GRID, width=1.0,
                               antialias=True, mode="lines")
        self._add(ln, grid=True)

        # Resaltar el plano de trabajo activo con su contorno
        kind, coord = self.work_plane
        x0, x1, y0, y1, z0, z1 = self.grid.bounds
        if kind == "xy":
            esquinas = [(x0, y0, coord), (x1, y0, coord),
                        (x1, y1, coord), (x0, y1, coord), (x0, y0, coord)]
        elif kind == "xz":
            esquinas = [(x0, coord, z0), (x1, coord, z0),
                        (x1, coord, z1), (x0, coord, z1), (x0, coord, z0)]
        else:
            esquinas = [(coord, y0, z0), (coord, y1, z0),
                        (coord, y1, z1), (coord, y0, z1), (coord, y0, z0)]
        marco = gl.GLLinePlotItem(pos=np.array(esquinas, dtype=float),
                                  color=(0.25, 0.55, 0.85, 0.9), width=2.0,
                                  antialias=True)
        self._add(marco, grid=True)

    def _draw_model(self) -> None:
        m = self.model
        if not m.nodes:
            return
        pts = np.array([[n.x, n.y, n.z] for n in m.nodes], dtype=float)

        # Elementos de área: caras + aristas
        caras, aristas = [], []
        for mem in m.members:
            if mem.tipo == "frame":
                a, b = mem.node_ids
                aristas.extend([pts[a], pts[b]])
                continue
            q = mem.node_ids
            caras.append([q[0], q[1], q[2]])
            caras.append([q[0], q[2], q[3]])
            for k in range(4):
                aristas.extend([pts[q[k]], pts[q[(k + 1) % 4]]])
        if caras:
            malla = gl.GLMeshItem(vertexes=pts, faces=np.array(caras, dtype=np.int32),
                                  color=_C_AREA_FACE, drawEdges=False,
                                  smooth=False, glOptions="translucent")
            self._add(malla)
        if aristas:
            ln = gl.GLLinePlotItem(pos=np.array(aristas, dtype=float),
                                   color=_C_AREA_EDGE, width=2.0,
                                   antialias=True, mode="lines")
            self._add(ln)

        # Frames aparte, con línea más gruesa
        seg_frame = []
        for mem in m.members:
            if mem.tipo == "frame":
                a, b = mem.node_ids
                seg_frame.extend([pts[a], pts[b]])
        if seg_frame:
            lf = gl.GLLinePlotItem(pos=np.array(seg_frame, dtype=float),
                                   color=_C_FRAME, width=4.0,
                                   antialias=True, mode="lines")
            self._add(lf)

        # Nodos: normales, seleccionados y los del elemento en curso
        sel = set(self.selected)
        pend = set(self._pending)
        normales = [p for n, p in zip(m.nodes, pts)
                    if n.id not in sel and n.id not in pend]
        if normales:
            self._add(gl.GLScatterPlotItem(
                pos=np.array(normales), color=_C_NODE, size=9, pxMode=True))
        marcados = [p for n, p in zip(m.nodes, pts) if n.id in sel or n.id in pend]
        if marcados:
            self._add(gl.GLScatterPlotItem(
                pos=np.array(marcados), color=_C_NODE_SEL, size=15, pxMode=True))

        # Apoyos: cubo verde bajo el nodo restringido
        escala = self.grid.size * 0.02
        for n, p in zip(m.nodes, pts):
            if any(n.restraints):
                self._add(self._cubo(p, escala, _C_SUPPORT))

        # Cargas nodales: línea roja en la dirección de la fuerza
        largo = self.grid.size * 0.12
        for n, p in zip(m.nodes, pts):
            f = np.array(n.loads[:3], dtype=float)
            mag = float(np.linalg.norm(f))
            if mag < 1e-30:
                continue
            fin = p + (f / mag) * largo
            self._add(gl.GLLinePlotItem(
                pos=np.array([p, fin]), color=_C_LOAD, width=3.0,
                antialias=True))

    @staticmethod
    def _cubo(centro: np.ndarray, s: float, color) -> gl.GLMeshItem:
        """Cubo centrado para representar un apoyo."""
        h = s / 2.0
        cx, cy, cz = centro
        v = np.array([
            [cx - h, cy - h, cz - h], [cx + h, cy - h, cz - h],
            [cx + h, cy + h, cz - h], [cx - h, cy + h, cz - h],
            [cx - h, cy - h, cz + h], [cx + h, cy - h, cz + h],
            [cx + h, cy + h, cz + h], [cx - h, cy + h, cz + h],
        ])
        f = np.array([
            [0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6],
            [1, 2, 6], [1, 6, 5], [0, 3, 7], [0, 7, 4],
        ], dtype=np.int32)
        return gl.GLMeshItem(vertexes=v, faces=f, color=color,
                             drawEdges=False, smooth=False, glOptions="opaque")
