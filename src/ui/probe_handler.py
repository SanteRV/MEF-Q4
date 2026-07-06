"""Probe interactivo sobre el canvas de Matplotlib.

Permite al usuario hacer click en cualquier punto del mapa de contornos
(paso 14) y obtener σx, σy, τxy, εx, εy, γxy interpolados en esa ubicación
física a partir de las funciones de forma del elemento Q4 que contiene
al punto.

Flujo:
    probe = ProbeHandler(plot_widget, structure, fem_result)
    probe.enable()    # activa cursor en cruz + escucha de clicks
    probe.disable()   # restablece cursor y desconecta
"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QCursor

from ..fem.q4_element import shape_functions, shape_function_derivatives
from .theme import Colors


# Tolerancia para considerar un punto dentro del elemento en coordenadas naturales
_NATURAL_TOL = 1e-6
# Máximo de iteraciones de Newton-Raphson para invertir x(ξ,η)
_NEWTON_MAX_ITER = 20
# Convergencia en coordenadas físicas
_NEWTON_TOL = 1e-10


def find_natural_coords(element, x: float, y: float) -> tuple[float, float] | None:
    """Encuentra (ξ, η) ∈ [-1, 1]^2 tal que el mapeo isoparamétrico del elemento
    devuelve el punto físico (x, y).

    Usa Newton-Raphson partiendo de (0, 0):
        f(ξ, η) = N(ξ, η) · X_nodos  -  (x, y)
        J = ∂f/∂(ξ, η) = (dN/d(ξ,η))ᵀ · X_nodos   (2x2)
        Δ = -J⁻¹ · f

    Devuelve (ξ, η) si converge dentro del elemento (|ξ|,|η| ≤ 1+tol),
    o None en caso contrario.
    """
    coords = element.coords  # (4, 2)
    target = np.array([x, y], dtype=float)
    xi, eta = 0.0, 0.0

    for _ in range(_NEWTON_MAX_ITER):
        N = shape_functions(xi, eta)            # (4,)
        current = N @ coords                     # (2,) = (x(ξ,η), y(ξ,η))
        residual = current - target              # f
        if np.linalg.norm(residual) < _NEWTON_TOL:
            break
        dN_nat = shape_function_derivatives(xi, eta)   # (2, 4): filas ξ, η
        # J de la transformación (ξ,η) -> (x,y): dx/dξ, dx/dη; dy/dξ, dy/dη
        # current_x = Σ Ni xi  =>  ∂current/∂ξ = Σ dNi/dξ xi  = (dN_nat[0] · X_col, ...)
        # En forma matricial: J_map = (dN_nat @ coords).T  con shape (2,2):
        #   filas = (∂x/∂ξ ∂x/∂η ; ∂y/∂ξ ∂y/∂η)
        J_map = (dN_nat @ coords).T              # (2, 2)
        try:
            delta = np.linalg.solve(J_map, -residual)
        except np.linalg.LinAlgError:
            return None
        xi += float(delta[0])
        eta += float(delta[1])

    # Validar convergencia y pertenencia al elemento
    N = shape_functions(xi, eta)
    current = N @ coords
    if np.linalg.norm(current - target) > 1e-6:
        return None
    if abs(xi) > 1.0 + _NATURAL_TOL or abs(eta) > 1.0 + _NATURAL_TOL:
        return None
    return xi, eta


class ProbeHandler:
    """Maneja clicks en el canvas para hacer probe (σ, ε en cualquier punto)."""

    def __init__(self, plot_widget, structure, fem_result):
        # plot_widget: PlotWidget de la app
        # structure: Structure con elementos
        # fem_result: FEMResult ya resuelto
        self.plot_widget = plot_widget
        self.structure = structure
        self.fem_result = fem_result

        self._cid_press: int | None = None
        self._enabled: bool = False

        # Artistas activos del último probe (se borran en el siguiente click).
        self._marker = None       # Line2D del círculo
        self._annotation = None   # ax.annotate

    # ------------------------------------------------------------------ API
    def enable(self) -> None:
        """Activa el modo probe: cambia el cursor a cross y conecta eventos."""
        if self._enabled:
            return
        canvas = self.plot_widget.canvas
        self._cid_press = canvas.mpl_connect("button_press_event", self._on_click)
        canvas.setCursor(QCursor(Qt.CrossCursor))
        self._enabled = True

    def disable(self) -> None:
        """Desactiva el modo probe: vuelve a cursor normal, desconecta."""
        if not self._enabled:
            return
        canvas = self.plot_widget.canvas
        if self._cid_press is not None:
            canvas.mpl_disconnect(self._cid_press)
            self._cid_press = None
        canvas.unsetCursor()
        self._clear_probe()
        self._enabled = False

    def set_structure_and_result(self, structure, fem_result) -> None:
        """Actualiza estructura y resultado (cuando se recalcula)."""
        self.structure = structure
        self.fem_result = fem_result
        self._clear_probe()

    # -------------------------------------------------------------- internos
    def _on_click(self, event) -> None:
        # Solo click izquierdo dentro de unos ejes con coordenadas válidas.
        if event.button != 1:
            return
        if event.inaxes is None:
            return
        if event.xdata is None or event.ydata is None:
            return

        x, y = float(event.xdata), float(event.ydata)

        hit = self._locate_point(x, y)
        if hit is None:
            # Click fuera de cualquier elemento: limpiar último probe y salir.
            self._clear_probe()
            self.plot_widget.canvas.draw_idle()
            return

        element, xi, eta = hit
        u_e = self._element_displacements(element)
        strain, stress = element.strains_stresses_at(xi, eta, u_e)

        self._draw_probe(event.inaxes, x, y, strain, stress)
        self.plot_widget.canvas.draw_idle()

    def _locate_point(self, x: float, y: float):
        """Devuelve (element, ξ, η) del primer elemento que contiene (x, y), o None."""
        for el in self.structure.elements:
            try:
                result = find_natural_coords(el, x, y)
            except Exception:
                result = None
            if result is not None:
                xi, eta = result
                return el, xi, eta
        return None

    def _element_displacements(self, element) -> np.ndarray:
        """Extrae el vector de desplazamientos (8,) del elemento desde el resultado global."""
        dofs = element.global_dofs()
        return np.asarray(self.fem_result.displacements)[dofs]

    def _clear_probe(self) -> None:
        """Borra el marcador y la anotación del probe anterior, si existen."""
        if self._marker is not None:
            try:
                self._marker.remove()
            except (ValueError, AttributeError):
                pass
            self._marker = None
        if self._annotation is not None:
            try:
                self._annotation.remove()
            except (ValueError, AttributeError):
                pass
            self._annotation = None

    def _draw_probe(self, ax, x: float, y: float,
                    strain: np.ndarray, stress: np.ndarray) -> None:
        """Dibuja el círculo en (x, y) y la anotación con σ y ε."""
        # Limpia probe previo (mantenemos uno a la vez).
        self._clear_probe()

        sx, sy, txy = float(stress[0]), float(stress[1]), float(stress[2])
        ex, ey, gxy = float(strain[0]), float(strain[1]), float(strain[2])

        # Marcador rojo en el punto clickeado.
        (self._marker,) = ax.plot(
            x, y, "o",
            markerfacecolor=Colors.DANGER,
            markeredgecolor=Colors.DANGER,
            markersize=7,
            zorder=10,
        )

        text = (
            f"Punto ({x:.4g}, {y:.4g})\n"
            f"σx  = {sx:.4g}\n"
            f"σy  = {sy:.4g}\n"
            f"τxy = {txy:.4g}\n"
            f"εx  = {ex:.4g}\n"
            f"εy  = {ey:.4g}\n"
            f"γxy = {gxy:.4g}"
        )

        self._annotation = ax.annotate(
            text,
            xy=(x, y),
            xytext=(14, 14),
            textcoords="offset points",
            fontsize=8.5,
            family="Consolas",
            color=Colors.TEXT if hasattr(Colors, "TEXT") else "#212529",
            ha="left",
            va="bottom",
            bbox=dict(
                boxstyle="round,pad=0.4",
                facecolor="white",
                edgecolor=Colors.PRIMARY,
                linewidth=1.2,
                alpha=0.95,
            ),
            zorder=11,
        )
