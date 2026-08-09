"""Nodos de la malla de elementos finitos.

Para qué sirve: el nodo es la unidad básica del modelo. Cada nodo aporta
2 grados de libertad (GDL) al sistema global — su desplazamiento horizontal
ux y vertical uy — y sobre él se definen los apoyos (restricciones) y las
cargas puntuales. Los elementos Q4 se construyen conectando 4 nodos.
"""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class Node:
    """Un nodo del modelo: posición, apoyos y cargas.

    La numeración es continua (id = 0, 1, 2, ...) y de ella se derivan los
    índices globales de los GDL: el nodo i ocupa las posiciones 2i (ux) y
    2i+1 (uy) en el vector de desplazamientos y en la matriz K global.
    """
    id: int
    x: float                     # coordenada fisica X (m)
    y: float                     # coordenada fisica Y (m)
    # Restricciones por grado de libertad: True = restringido (apoyo).
    # Un apoyo fijo restringe ambos; un patin restringe solo uno.
    restraint_x: bool = False
    restraint_y: bool = False
    # Desplazamiento IMPUESTO en el GDL restringido (m). El documento
    # (cap. 01.01.06.03) admite dos tipos de condicion de borde: cargas
    # aplicadas y desplazamientos conocidos. Estos valores forman el vector
    # U_c de la ec. 1.6.1; con 0 se recupera el apoyo rigido habitual.
    prescribed_x: float = 0.0
    prescribed_y: float = 0.0
    # Cargas puntuales aplicadas en el nodo (N). Entran directo al vector F.
    load_x: float = 0.0
    load_y: float = 0.0

    @property
    def dofs(self) -> tuple[int, int]:
        """Índices globales de los grados de libertad (ux, uy).

        Para qué sirve: es el "mapa" del ensamblaje — conecta los GDL
        locales del elemento con su posición en el sistema global.
        """
        return (2 * self.id, 2 * self.id + 1)

    @property
    def restraints(self) -> tuple[bool, bool]:
        """Par (restringido_x, restringido_y) para consultas rápidas."""
        return (self.restraint_x, self.restraint_y)

    @property
    def loads(self) -> tuple[float, float]:
        """Par (Fx, Fy) de cargas nodales para armar el vector F."""
        return (self.load_x, self.load_y)

    @property
    def prescribed(self) -> tuple[float, float]:
        """Par (ux, uy) impuestos — solo se leen en los GDL restringidos."""
        return (self.prescribed_x, self.prescribed_y)
