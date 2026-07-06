"""Canvas de edición FEM basado en PyQtGraph.

Reemplaza al `FigureCanvasQTAgg` de matplotlib en el editor gráfico
porque matplotlib no rinde para edición en tiempo real (5-10 fps con
muchos nodos). PyQtGraph usa el rasterizador de Qt y alcanza 60+ fps
fácilmente.

Encapsula:
- Pan/zoom built-in suaves (botón izquierdo arrastra, rueda zoom)
- Grid configurable
- Render incremental por nodo/elemento (artistas se guardan y removible)
- Eventos del mouse en coordenadas de datos
- Snap a grid configurable
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QPointF, Signal, QObject
from PySide6.QtGui import QPen, QBrush, QColor, QFont, QPolygonF
from PySide6.QtWidgets import QGraphicsPolygonItem, QGraphicsTextItem

from ..fem.node import Node
from ..fem.q4_element import Q4Element
from ..fem.structure import Structure
from .theme import Colors


# Configuración global de PyQtGraph: fondo blanco, foreground negro
pg.setConfigOption("background", "w")
pg.setConfigOption("foreground", "k")
pg.setConfigOptions(antialias=True)


# ============= Colores como QColor (más rápido que parsear hex en cada draw) =============
_C_PRIMARY = QColor(Colors.PRIMARY)
_C_PRIMARY_LIGHT = QColor(Colors.PRIMARY_LIGHT)
_C_BORDER = QColor(Colors.BORDER)
_C_HOVER = QColor(Colors.PRIMARY)
_C_PENDING = QColor("#ff9800")
_C_SELECTED = QColor("#ff0066")
_C_ELEMENT_FILL = QColor("#cfe2f3")
_C_ELEMENT_FILL.setAlpha(180)
_C_ELEMENT_EDGE = QColor("#1f77b4")
_C_NODE = QColor("black")
_C_SUPPORT = QColor("green")
_C_LOAD = QColor("red")


class CanvasSignals(QObject):
    """Señales emitidas por el canvas hacia el editor."""
    left_click = Signal(float, float)
    right_click = Signal(float, float)
    left_double_click = Signal(float, float)
    mouse_move = Signal(float, float)
    mouse_leave = Signal()
    key_delete = Signal()
    key_home = Signal()
    key_escape = Signal()
    # Rubber band para multi-selección rectangular.
    # x1,y1,x2,y2 ya normalizados (xmin,ymin,xmax,ymax).
    rubber_band_finished = Signal(float, float, float, float)


class PgCanvas(pg.PlotWidget):
    """Canvas pyqtgraph con API similar al canvas matplotlib anterior.

    Se encarga de:
    - Render rápido de nodos, elementos, apoyos, cargas, grid
    - Tracking de artistas por id de nodo/elemento (render incremental)
    - Pan/zoom built-in suaves
    - Reenviar eventos del mouse al editor vía señales
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setBackground("w")
        self.setAspectLocked(True)
        self.showGrid(x=False, y=False)   # usamos nuestro grid custom

        # IMPORTANTE: desactivar el auto-SI-prefix de PyQtGraph en los ejes,
        # que convierte valores pequeños a mm con notación "(x0.001)".
        # Queremos las coordenadas literales en metros, como las introdujo el usuario.
        ax_left = self.getPlotItem().getAxis("left")
        ax_bot = self.getPlotItem().getAxis("bottom")
        ax_left.enableAutoSIPrefix(False)
        ax_bot.enableAutoSIPrefix(False)
        ax_left.setLabel("Y", units="m")
        ax_bot.setLabel("X", units="m")
        # Más decimales en los ticks para que se vea (0.5) y no (500m)
        ax_left.setStyle(autoExpandTextSpace=False)
        ax_bot.setStyle(autoExpandTextSpace=False)
        self.signals = CanvasSignals()

        # Estado
        self.show_labels = True
        self.show_grid = True
        self.grid_size = 0.25

        # Tracking de artistas por id
        self._artists_per_node: dict[int, list] = {}
        self._artists_per_element: dict[int, list] = {}
        self._overlay_artists: list = []
        self._grid_items: list = []

        # Eventos
        self.scene().sigMouseMoved.connect(self._on_mouse_moved)
        self.scene().sigMouseClicked.connect(self._on_mouse_clicked)

        # Cursor estilo crosshair por defecto
        self.setCursor(Qt.ArrowCursor)

        # Configurar viewbox: deshabilitar menú contextual default de pyqtgraph
        vb = self.getPlotItem().getViewBox()
        vb.setMenuEnabled(False)
        vb.setMouseMode(pg.ViewBox.PanMode)

        # Estado del rubber-band (multi-selección con Ctrl+arrastrar)
        self._rubber_active = False
        self._rubber_start_view = None
        self._rubber_rect_item = None

    # ============= API de configuración =============
    def set_grid_size(self, size: float) -> None:
        self.grid_size = size
        self._refresh_grid()

    def set_grid_visible(self, visible: bool) -> None:
        self.show_grid = visible
        self._refresh_grid()

    def set_labels_visible(self, visible: bool) -> None:
        self.show_labels = visible
        # Re-render: las etiquetas son hijas de los artists de nodos/elementos
        # Más fácil: redibuja todo
        # (el llamador debería hacer self.render_all(structure) tras este toggle)

    def set_view_limits(self, xmin: float, xmax: float,
                        ymin: float, ymax: float) -> None:
        self.setRange(xRange=(xmin, xmax), yRange=(ymin, ymax), padding=0)

    def fit_to_structure(self, structure: Structure,
                         margin_factor: float = 0.15) -> None:
        if not structure.nodes:
            self.setRange(xRange=(-1, 1), yRange=(-1, 1), padding=0)
            return
        xs = [n.x for n in structure.nodes]
        ys = [n.y for n in structure.nodes]
        size = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
        m = size * margin_factor
        self.setRange(
            xRange=(min(xs) - m, max(xs) + m),
            yRange=(min(ys) - m, max(ys) + m),
            padding=0,
        )

    # ============= Render de un nodo =============
    def render_node(self, n: Node) -> None:
        """Dibuja UN nodo: marker + label + apoyo + carga."""
        # Si ya existía, quitarlo primero
        if n.id in self._artists_per_node:
            self.unrender_node(n.id)

        artists = []
        # Marker base
        scatter = pg.ScatterPlotItem(
            pos=[(n.x, n.y)], size=10,
            brush=_C_NODE, pen=pg.mkPen(_C_NODE),
        )
        scatter.setZValue(10)
        self.addItem(scatter)
        artists.append(scatter)

        # Label
        if self.show_labels:
            txt = pg.TextItem(text=f"N{n.id + 1}", color=Colors.TEXT,
                              anchor=(-0.2, 1.2))
            txt.setPos(n.x, n.y)
            txt.setFont(QFont("Segoe UI", 8, QFont.Bold))
            txt.setZValue(11)
            self.addItem(txt)
            artists.append(txt)

        # Apoyo
        if n.restraint_x and n.restraint_y:
            sup = pg.ScatterPlotItem(
                pos=[(n.x, n.y)], size=22, symbol="s",
                brush=None, pen=pg.mkPen(_C_SUPPORT, width=2),
            )
            sup.setZValue(8)
            self.addItem(sup)
            artists.append(sup)
        elif n.restraint_x or n.restraint_y:
            sup = pg.ScatterPlotItem(
                pos=[(n.x, n.y)], size=18, symbol="t",
                brush=None, pen=pg.mkPen(_C_SUPPORT, width=2),
            )
            sup.setZValue(8)
            self.addItem(sup)
            artists.append(sup)

        # Carga
        if n.load_x or n.load_y:
            norm = float(np.hypot(n.load_x, n.load_y))
            # Tamaño visual: 8% del rango actual de X
            vb = self.getViewBox()
            xr = vb.viewRange()[0]
            size = max((xr[1] - xr[0]) * 0.08, 0.1)
            dx = (n.load_x / norm) * size if norm else 0.0
            dy = (n.load_y / norm) * size if norm else 0.0
            # Línea (flecha) — la dibujo como un Line + marker en la punta
            arrow_line = pg.PlotDataItem(
                x=[n.x, n.x + dx], y=[n.y, n.y + dy],
                pen=pg.mkPen(_C_LOAD, width=2),
            )
            arrow_line.setZValue(12)
            self.addItem(arrow_line)
            artists.append(arrow_line)
            # Punta de la flecha
            arrow = pg.ArrowItem(
                pos=(n.x + dx, n.y + dy),
                angle=np.degrees(np.arctan2(dy, dx)) - 180,
                tipAngle=30, baseAngle=0, headLen=12, tailLen=0,
                brush=_C_LOAD, pen=pg.mkPen(_C_LOAD),
            )
            arrow.setZValue(13)
            self.addItem(arrow)
            artists.append(arrow)
            if self.show_labels:
                txt = pg.TextItem(text=f"({n.load_x:g}, {n.load_y:g}) N",
                                  color=_C_LOAD,
                                  anchor=(0, 1))
                txt.setPos(n.x + dx, n.y + dy)
                txt.setFont(QFont("Segoe UI", 8))
                txt.setZValue(14)
                self.addItem(txt)
                artists.append(txt)

        self._artists_per_node[n.id] = artists

    def unrender_node(self, node_id: int) -> None:
        if node_id in self._artists_per_node:
            for a in self._artists_per_node[node_id]:
                try:
                    self.removeItem(a)
                except Exception:
                    pass
            del self._artists_per_node[node_id]

    # ============= Render de un elemento =============
    def render_element(self, el: Q4Element) -> None:
        if el.id in self._artists_per_element:
            self.unrender_element(el.id)

        artists = []
        # Polígono relleno: usamos QGraphicsPolygonItem nativo de Qt
        polygon = QPolygonF([QPointF(n.x, n.y) for n in el.nodes])
        poly_item = QGraphicsPolygonItem(polygon)
        poly_item.setBrush(QBrush(_C_ELEMENT_FILL))
        poly_item.setPen(QPen(_C_ELEMENT_EDGE, 0))  # 0 = pixel width
        poly_item.setZValue(2)
        self.getPlotItem().scene().addItem(poly_item)
        # Importante: QGraphicsPolygonItem necesita ser hijo del ViewBox
        # para escalar con la vista. La forma correcta es usar pg.QtWidgets:
        # Mejor reemplazo: usar pg.PlotCurveItem cerrado con fill.
        try:
            self.getPlotItem().scene().removeItem(poly_item)
        except Exception:
            pass
        # Usamos PlotCurveItem cerrado con fillLevel=None pero fillColor + fillBrush
        # PyQtGraph: PlotCurveItem soporta fillBrush si se cierra el camino
        xs = [n.x for n in el.nodes] + [el.nodes[0].x]
        ys = [n.y for n in el.nodes] + [el.nodes[0].y]
        curve = pg.PlotCurveItem(
            x=xs, y=ys,
            pen=pg.mkPen(_C_ELEMENT_EDGE, width=1.5),
            fillLevel=None,
            brush=pg.mkBrush(_C_ELEMENT_FILL),
        )
        # Truco: fillLevel solo funciona con eje horizontal; mejor usamos un
        # FillBetweenItem o un polígono manual con setData.
        # Para Q4 dibujamos el borde y rellenamos manualmente con QPolygonF
        # dentro del scene del ViewBox (sí escala correctamente):
        # Acercamiento más robusto: usar pg.PolyLineROI? No, son interactivas.
        # Solución definitiva: crear un QGraphicsPolygonItem y añadirlo al ViewBox.
        del curve  # no usaremos esto
        polygon = QPolygonF([QPointF(n.x, n.y) for n in el.nodes])
        poly_item = QGraphicsPolygonItem(polygon)
        poly_item.setBrush(QBrush(_C_ELEMENT_FILL))
        pen = QPen(_C_ELEMENT_EDGE, 0)
        pen.setCosmetic(True)
        pen.setWidthF(1.5)
        poly_item.setPen(pen)
        poly_item.setZValue(2)
        # Añadirlo al ViewBox (no al scene) para que escale con los ejes
        self.getViewBox().addItem(poly_item)
        artists.append(poly_item)

        # Label del elemento (en el centroide)
        if self.show_labels:
            cx = float(np.mean([n.x for n in el.nodes]))
            cy = float(np.mean([n.y for n in el.nodes]))
            txt = pg.TextItem(text=f"E{el.id + 1}", color=_C_ELEMENT_EDGE,
                              anchor=(0.5, 0.5))
            txt.setPos(cx, cy)
            txt.setFont(QFont("Segoe UI", 9, QFont.Bold))
            txt.setZValue(3)
            self.addItem(txt)
            artists.append(txt)

        self._artists_per_element[el.id] = artists

    def unrender_element(self, el_id: int) -> None:
        if el_id in self._artists_per_element:
            for a in self._artists_per_element[el_id]:
                try:
                    if a.parentItem() is not None or a.scene() is not None:
                        if hasattr(a, "scene") and a.scene() is not None:
                            a.scene().removeItem(a)
                        else:
                            self.removeItem(a)
                    else:
                        self.removeItem(a)
                except Exception:
                    try:
                        self.removeItem(a)
                    except Exception:
                        pass
            del self._artists_per_element[el_id]

    # ============= Render del overlay (hover, pending Q4, selected) =============
    def clear_overlay(self) -> None:
        for a in self._overlay_artists:
            try:
                self.removeItem(a)
            except Exception:
                pass
        self._overlay_artists.clear()

    def render_overlay(self, *, hovered: Node | None = None,
                       selected: Node | None = None,
                       pending_q4: list[Node] | None = None) -> None:
        self.clear_overlay()

        # Preview del Q4 parcial
        if pending_q4 and len(pending_q4) >= 2:
            xy = [(n.x, n.y) for n in pending_q4]
            if len(pending_q4) >= 3:
                xy += [xy[0]]
            xs = [p[0] for p in xy]
            ys = [p[1] for p in xy]
            preview = pg.PlotDataItem(
                x=xs, y=ys,
                pen=pg.mkPen(_C_PENDING, width=2, style=Qt.DashLine),
            )
            preview.setZValue(5)
            self.addItem(preview)
            self._overlay_artists.append(preview)
        # Marcadores naranjas con número de orden
        if pending_q4:
            for idx, n in enumerate(pending_q4, start=1):
                mk = pg.ScatterPlotItem(
                    pos=[(n.x, n.y)], size=14,
                    brush=_C_PENDING, pen=pg.mkPen(_C_PENDING),
                )
                mk.setZValue(15)
                self.addItem(mk)
                self._overlay_artists.append(mk)
                txt = pg.TextItem(text=str(idx), color="white",
                                  anchor=(0.5, 0.5))
                txt.setPos(n.x, n.y)
                txt.setFont(QFont("Segoe UI", 8, QFont.Bold))
                txt.setZValue(16)
                self.addItem(txt)
                self._overlay_artists.append(txt)

        # Nodo seleccionado: marcador rosa
        if selected is not None and selected not in (pending_q4 or []):
            mk = pg.ScatterPlotItem(
                pos=[(selected.x, selected.y)], size=12,
                brush=_C_SELECTED, pen=pg.mkPen(_C_SELECTED),
            )
            mk.setZValue(15)
            self.addItem(mk)
            self._overlay_artists.append(mk)

        # Hover: anillo navy alrededor del nodo
        if hovered is not None and hovered not in (pending_q4 or []):
            ring = pg.ScatterPlotItem(
                pos=[(hovered.x, hovered.y)], size=22, symbol="o",
                brush=None, pen=pg.mkPen(_C_HOVER, width=2),
            )
            ring.setZValue(16)
            self.addItem(ring)
            self._overlay_artists.append(ring)

    # ============= Grid =============
    def _refresh_grid(self) -> None:
        # Quitar grid actual
        for ln in self._grid_items:
            try:
                self.removeItem(ln)
            except Exception:
                pass
        self._grid_items.clear()
        if not self.show_grid or self.grid_size <= 0:
            return
        # Calcular líneas dentro del rango visible
        vb = self.getViewBox()
        xr, yr = vb.viewRange()
        g = self.grid_size
        # Limit lines to avoid drawing thousands at extreme zoom out
        n_max = 200
        n_x = int((xr[1] - xr[0]) / g) + 1
        n_y = int((yr[1] - yr[0]) / g) + 1
        if n_x > n_max or n_y > n_max:
            return
        xs = np.arange(np.floor(xr[0] / g) * g, xr[1] + g, g)
        ys = np.arange(np.floor(yr[0] / g) * g, yr[1] + g, g)
        pen_grid = pg.mkPen("#dddddd", width=0.5)
        pen_axis = pg.mkPen("#888888", width=0.8)
        for x in xs:
            ln = pg.InfiniteLine(
                pos=x, angle=90, movable=False,
                pen=pen_axis if abs(x) < 1e-12 else pen_grid,
            )
            ln.setZValue(0)
            self.addItem(ln, ignoreBounds=True)
            self._grid_items.append(ln)
        for y in ys:
            ln = pg.InfiniteLine(
                pos=y, angle=0, movable=False,
                pen=pen_axis if abs(y) < 1e-12 else pen_grid,
            )
            ln.setZValue(0)
            self.addItem(ln, ignoreBounds=True)
            self._grid_items.append(ln)

    # ============= Limpieza total =============
    def clear_all(self) -> None:
        # Quitar todos los nodos
        for node_id in list(self._artists_per_node.keys()):
            self.unrender_node(node_id)
        # Quitar todos los elementos
        for el_id in list(self._artists_per_element.keys()):
            self.unrender_element(el_id)
        self.clear_overlay()
        for ln in self._grid_items:
            try:
                self.removeItem(ln)
            except Exception:
                pass
        self._grid_items.clear()

    def render_all(self, structure: Structure) -> None:
        """Full re-render. Usar tras undo o cambio masivo."""
        self.clear_all()
        self._refresh_grid()
        for el in structure.elements:
            self.render_element(el)
        for n in structure.nodes:
            self.render_node(n)

    # ============= Eventos del mouse =============
    def _on_mouse_moved(self, scene_pos) -> None:
        vb = self.getViewBox()
        if not vb.sceneBoundingRect().contains(scene_pos):
            self.signals.mouse_leave.emit()
            return
        mouse_point = vb.mapSceneToView(scene_pos)
        self.signals.mouse_move.emit(mouse_point.x(), mouse_point.y())

    def _on_mouse_clicked(self, mouse_event) -> None:
        scene_pos = mouse_event.scenePos()
        vb = self.getViewBox()
        if not vb.sceneBoundingRect().contains(scene_pos):
            return
        view_pt = vb.mapSceneToView(scene_pos)
        if mouse_event.button() == Qt.LeftButton:
            mouse_event.accept()
            # PyQtGraph marca double() en el segundo click del doble-click
            if mouse_event.double():
                self.signals.left_double_click.emit(view_pt.x(), view_pt.y())
            else:
                self.signals.left_click.emit(view_pt.x(), view_pt.y())
        elif mouse_event.button() == Qt.RightButton:
            mouse_event.accept()
            self.signals.right_click.emit(view_pt.x(), view_pt.y())

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_Delete or event.key() == Qt.Key_Backspace:
            self.signals.key_delete.emit()
        elif event.key() == Qt.Key_Home:
            self.signals.key_home.emit()
        elif event.key() == Qt.Key_Escape:
            self.signals.key_escape.emit()
        else:
            super().keyPressEvent(event)

    # ---------- Rubber-band (Ctrl + drag con botón izquierdo) ----------
    def mousePressEvent(self, event) -> None:
        # Si Ctrl está presionado + botón izquierdo: iniciar rubber-band
        if (event.button() == Qt.LeftButton and
                event.modifiers() & Qt.ControlModifier):
            scene_pos = self.mapToScene(event.position().toPoint())
            vb = self.getViewBox()
            if vb.sceneBoundingRect().contains(scene_pos):
                view_pt = vb.mapSceneToView(scene_pos)
                self._rubber_active = True
                self._rubber_start_view = (view_pt.x(), view_pt.y())
                self._draw_rubber_rect(view_pt.x(), view_pt.y(),
                                        view_pt.x(), view_pt.y())
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._rubber_active and self._rubber_start_view is not None:
            scene_pos = self.mapToScene(event.position().toPoint())
            vb = self.getViewBox()
            if vb.sceneBoundingRect().contains(scene_pos):
                view_pt = vb.mapSceneToView(scene_pos)
                self._draw_rubber_rect(
                    self._rubber_start_view[0], self._rubber_start_view[1],
                    view_pt.x(), view_pt.y(),
                )
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if self._rubber_active and event.button() == Qt.LeftButton:
            scene_pos = self.mapToScene(event.position().toPoint())
            vb = self.getViewBox()
            if vb.sceneBoundingRect().contains(scene_pos):
                view_pt = vb.mapSceneToView(scene_pos)
                x0, y0 = self._rubber_start_view
                x1, y1 = view_pt.x(), view_pt.y()
                # Normalizar
                xmin, xmax = min(x0, x1), max(x0, x1)
                ymin, ymax = min(y0, y1), max(y0, y1)
                self.signals.rubber_band_finished.emit(xmin, ymin, xmax, ymax)
            self._rubber_active = False
            self._rubber_start_view = None
            self._clear_rubber_rect()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def _draw_rubber_rect(self, x0: float, y0: float,
                           x1: float, y1: float) -> None:
        self._clear_rubber_rect()
        # Rectángulo punteado navy
        xs = [x0, x1, x1, x0, x0]
        ys = [y0, y0, y1, y1, y0]
        rect = pg.PlotDataItem(
            x=xs, y=ys,
            pen=pg.mkPen(_C_PRIMARY, width=1.5, style=Qt.DashLine),
        )
        rect.setZValue(30)
        self.addItem(rect)
        self._rubber_rect_item = rect

    def _clear_rubber_rect(self) -> None:
        if self._rubber_rect_item is not None:
            try:
                self.removeItem(self._rubber_rect_item)
            except Exception:
                pass
            self._rubber_rect_item = None
