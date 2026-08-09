"""Nodos para el modelo PLATE (flexión de placas, teoría de Kirchhoff).

Para qué sirve: en el elemento plate cada nodo tiene 3 grados de libertad
(cap. 01.01.03 del documento teórico):
    w   desplazamiento transversal (eje Z, positivo hacia +Z)
    θx  giro alrededor del eje X  (θx = ∂w/∂y, ec. 1.3.3)
    θy  giro alrededor del eje Y  (θy = -∂w/∂x, ec. 1.3.4)

Es una familia PARALELA a node.py (plane): el modelo plane conserva sus
nodos de 2 GDL intactos y el plate usa estos de 3 GDL. El futuro elemento
shell (20 GDL) combinará ambos con nodos de 5 GDL.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class NodePlate:
    """Un nodo del modelo plate: posición, apoyos y cargas de flexión.

    Los índices globales de los GDL se derivan del id: el nodo i ocupa
    las posiciones 3i (w), 3i+1 (θx) y 3i+2 (θy) del sistema global.
    """
    id: int
    x: float                      # coordenada fisica X (m)
    y: float                      # coordenada fisica Y (m)
    # Restricciones por GDL: True = restringido (apoyo)
    restraint_w: bool = False     # desplazamiento transversal impedido
    restraint_rx: bool = False    # giro alrededor de X impedido
    restraint_ry: bool = False    # giro alrededor de Y impedido
    # Valores IMPUESTOS en los GDL restringidos (vector U_c de la ec. 1.6.1).
    # Con 0 se recupera el apoyo rigido; distintos de 0 permiten modelar
    # asentamientos o giros conocidos (cap. 01.01.06.03).
    prescribed_w: float = 0.0     # asentamiento impuesto (m)
    prescribed_rx: float = 0.0    # giro impuesto alrededor de X (rad)
    prescribed_ry: float = 0.0    # giro impuesto alrededor de Y (rad)
    # Cargas aplicadas en el nodo
    load_fz: float = 0.0          # fuerza transversal (N, +Z)
    load_mx: float = 0.0          # momento alrededor de X (N.m)
    load_my: float = 0.0          # momento alrededor de Y (N.m)

    @property
    def dofs(self) -> tuple[int, int, int]:
        """Índices globales de los 3 GDL (w, θx, θy) — mapa del ensamblaje."""
        base = 3 * self.id
        return (base, base + 1, base + 2)

    @property
    def restraints(self) -> tuple[bool, bool, bool]:
        """Terna (w, θx, θy) restringidos, para consultas rápidas."""
        return (self.restraint_w, self.restraint_rx, self.restraint_ry)

    @property
    def loads(self) -> tuple[float, float, float]:
        """Terna (Fz, Mx, My) de cargas nodales para armar el vector F."""
        return (self.load_fz, self.load_mx, self.load_my)

    @property
    def prescribed(self) -> tuple[float, float, float]:
        """Terna (w, θx, θy) impuestos — solo se leen en los GDL restringidos."""
        return (self.prescribed_w, self.prescribed_rx, self.prescribed_ry)
