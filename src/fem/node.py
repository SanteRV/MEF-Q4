"""Nodos de la malla de elementos finitos."""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class Node:
    id: int
    x: float
    y: float
    # Restricciones por grado de libertad: True = restringido (apoyo)
    restraint_x: bool = False
    restraint_y: bool = False
    # Cargas puntuales aplicadas en el nodo
    load_x: float = 0.0
    load_y: float = 0.0

    @property
    def dofs(self) -> tuple[int, int]:
        """Índices globales de los grados de libertad (ux, uy)."""
        return (2 * self.id, 2 * self.id + 1)

    @property
    def restraints(self) -> tuple[bool, bool]:
        return (self.restraint_x, self.restraint_y)

    @property
    def loads(self) -> tuple[float, float]:
        return (self.load_x, self.load_y)
