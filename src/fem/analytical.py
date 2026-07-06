"""Soluciones analíticas para validar el MEF.

Cada caso define:
- Una función `setup(E, nu, t, ...)` que devuelve la Structure configurada.
- Una función `analytical(...)` que devuelve los valores esperados (u_max, σ).
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np
from .node import Node
from .q4_element import Q4Element
from .structure import Structure


@dataclass
class AnalyticalCase:
    name: str
    description: str
    # σx, σy, τxy esperados en el centro del elemento (estado uniforme idealmente)
    sigma_x_expected: float
    sigma_y_expected: float
    tau_xy_expected: float
    u_max_expected: float


def case_uniform_traction(E: float, nu: float, t: float,
                          L: float = 1.0, H: float = 1.0,
                          q: float = 1000.0,
                          nx: int = 1, ny: int = 1) -> tuple[Structure, AnalyticalCase]:
    """Placa rectangular L×H bajo tracción uniforme q (N/m²) en el borde derecho.

    Estado de esfuerzos teórico (uniforme): σx = q, σy = 0, τxy = 0
    Desplazamiento del borde derecho:     u_x = (σx · L) / E = (q · L) / E
    """
    from ..ui.convergence_dialog import _generate_rect_mesh, _apply_boundary_left_right
    # Total de fuerza horizontal aplicada en el borde derecho (de altura H, espesor t):
    total_fx = q * H * t
    s = _generate_rect_mesh(0, L, 0, H, nx, ny, E, nu, t, plane_stress=True)
    _apply_boundary_left_right(s, 0, L, total_fx=total_fx, total_fy=0.0)
    # Adicional: fijar también uy=0 en un solo nodo del borde izquierdo
    # para evitar movimiento rígido si ya no está fijado
    # (left_right ya pone restraint_x y restraint_y True en todos los del borde izq, OK)
    case = AnalyticalCase(
        name="Tracción uniforme en placa rectangular",
        description=(
            f"Placa rectangular {L}×{H} m, espesor t={t} m, "
            f"bajo carga distribuida q = {q} Pa en el borde derecho.\n"
            f"Borde izquierdo restringido.\n\n"
            "Solución analítica (estado uniforme):\n"
            "    σx = q = {q} Pa\n"
            "    σy = 0\n"
            "    τxy = 0\n"
            "    u_max (extremo derecho) = q·L/E = "
            f"{q * L / E:.6e} m"
        ),
        sigma_x_expected=q,
        sigma_y_expected=0.0,
        tau_xy_expected=0.0,
        u_max_expected=q * L / E,
    )
    return s, case


def case_pure_shear(E: float, nu: float, t: float,
                    L: float = 1.0, H: float = 1.0,
                    tau: float = 500.0,
                    nx: int = 1, ny: int = 1) -> tuple[Structure, AnalyticalCase]:
    """Placa bajo corte puro (tracciones de borde tangenciales). Estado uniforme."""
    from ..ui.convergence_dialog import _generate_rect_mesh
    s = _generate_rect_mesh(0, L, 0, H, nx, ny, E, nu, t, plane_stress=True)
    # En el borde derecho aplicamos fuerza vertical, en el superior fuerza horizontal
    right = [n for n in s.nodes if abs(n.x - L) < 1e-9]
    top = [n for n in s.nodes if abs(n.y - H) < 1e-9]
    # Fijar esquina inferior izquierda totalmente
    for n in s.nodes:
        if abs(n.x) < 1e-9 and abs(n.y) < 1e-9:
            n.restraint_x = True
            n.restraint_y = True
        # Fijar borde inferior en y (roller)
        if abs(n.y) < 1e-9:
            n.restraint_y = True
        # Fijar borde izquierdo en x (roller)
        if abs(n.x) < 1e-9:
            n.restraint_x = True
    # Cargas tangenciales: τ * H * t en cada borde
    Fy_right = tau * H * t
    Fx_top = tau * L * t
    for n in right:
        n.load_y += Fy_right / len(right)
    for n in top:
        n.load_x += Fx_top / len(top)

    G = E / (2 * (1 + nu))
    case = AnalyticalCase(
        name="Corte puro en placa rectangular",
        description=(
            f"Placa {L}×{H} m con tracciones tangenciales τ = {tau} Pa "
            f"en bordes superior y derecho.\n\n"
            "Solución analítica:\n"
            f"    σx = σy = 0,  τxy = {tau} Pa\n"
            f"    γxy = τ/G  con G = E / [2(1+ν)] = {G:.3e} Pa\n"
            f"    γxy = {tau / G:.6e}"
        ),
        sigma_x_expected=0.0,
        sigma_y_expected=0.0,
        tau_xy_expected=tau,
        u_max_expected=float("nan"),
    )
    return s, case


# Catálogo de casos
ANALYTICAL_CASES = {
    "Tracción uniforme":   case_uniform_traction,
    "Corte puro":           case_pure_shear,
}
