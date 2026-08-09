"""Nodos para el modelo SHELL (flat shell delgado).

Para qué sirve: en el elemento flat shell cada nodo tiene 5 grados de
libertad (cap. 01.01.04 y 01.01.07.01 del documento teórico):

    u   desplazamiento en X   (membrana, viene del elemento plane)
    v   desplazamiento en Y   (membrana, viene del elemento plane)
    w   desplazamiento en Z   (flexión, viene del elemento plate)
    θx  giro alrededor de X   (θx =  ∂w/∂y)
    θy  giro alrededor de Y   (θy = -∂w/∂x)

El orden (u, v, w, θx, θy) es el que fija el documento en 01.01.07.01 para
el vector de desplazamientos del flat shell:

    q = (u1, v1, w1, θx1, θy1, u2, ... , u4, v4, w4, θx4, θy4)

Es una familia PARALELA a node.py (2 GDL) y node_plate.py (3 GDL): cada
modelo conserva su propia clase de nodo, sin tocar los otros dos.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class NodeShell:
    """Un nodo del modelo flat shell: posición, apoyos y cargas.

    Los índices globales de los GDL se derivan del id: el nodo i ocupa las
    posiciones 5i (u), 5i+1 (v), 5i+2 (w), 5i+3 (θx) y 5i+4 (θy).
    """
    id: int
    x: float                       # coordenada fisica X (m)
    y: float                       # coordenada fisica Y (m)

    # Restricciones por GDL: True = el desplazamiento es CONOCIDO
    restraint_u: bool = False
    restraint_v: bool = False
    restraint_w: bool = False
    restraint_rx: bool = False
    restraint_ry: bool = False

    # Valores impuestos en los GDL restringidos (vector U_c, ec. 1.6.1).
    # Con 0 se recupera el apoyo rígido habitual.
    prescribed_u: float = 0.0
    prescribed_v: float = 0.0
    prescribed_w: float = 0.0
    prescribed_rx: float = 0.0
    prescribed_ry: float = 0.0

    # Cargas nodales
    load_fx: float = 0.0           # fuerza en X (N), en el plano
    load_fy: float = 0.0           # fuerza en Y (N), en el plano
    load_fz: float = 0.0           # fuerza transversal en Z (N)
    load_mx: float = 0.0           # momento alrededor de X (N.m)
    load_my: float = 0.0           # momento alrededor de Y (N.m)

    @property
    def dofs(self) -> tuple[int, int, int, int, int]:
        """Índices globales de los 5 GDL (u, v, w, θx, θy)."""
        base = 5 * self.id
        return (base, base + 1, base + 2, base + 3, base + 4)

    @property
    def restraints(self) -> tuple[bool, bool, bool, bool, bool]:
        """Quíntupla (u, v, w, θx, θy) restringidos."""
        return (self.restraint_u, self.restraint_v, self.restraint_w,
                self.restraint_rx, self.restraint_ry)

    @property
    def loads(self) -> tuple[float, float, float, float, float]:
        """Quíntupla (Fx, Fy, Fz, Mx, My) para armar el vector F."""
        return (self.load_fx, self.load_fy, self.load_fz,
                self.load_mx, self.load_my)

    @property
    def prescribed(self) -> tuple[float, float, float, float, float]:
        """Quíntupla de valores impuestos — solo se leen en GDL restringidos."""
        return (self.prescribed_u, self.prescribed_v, self.prescribed_w,
                self.prescribed_rx, self.prescribed_ry)
