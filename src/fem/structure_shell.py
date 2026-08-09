"""Estructura del modelo SHELL: contenedor de nodos (5 GDL) + elementos.

Para qué sirve: igual que structure.py (plane) y structure_plate.py (plate),
pero con 5 GDL por nodo (u, v, w, θx, θy). Responde lo que el solver
necesita: número de GDL, cuáles están libres, cuáles restringidos y qué
valores se les imponen (vector U_c de la ec. 1.6.1).
"""
from __future__ import annotations
from dataclasses import dataclass, field

import numpy as np

from .node_shell import NodeShell
from .shell_element import ShellElement


@dataclass
class StructureShell:
    """El modelo flat shell completo: lista de nodos + lista de elementos."""
    nodes: list[NodeShell] = field(default_factory=list)
    elements: list[ShellElement] = field(default_factory=list)

    @property
    def n_dofs(self) -> int:
        """Número total de GDL del sistema: 5 por nodo."""
        return 5 * len(self.nodes)

    def add_node(self, node: NodeShell) -> None:
        self.nodes.append(node)

    def add_element(self, element: ShellElement) -> None:
        self.elements.append(element)

    def free_dofs(self) -> list[int]:
        """GDL sin apoyo — las incógnitas del sistema reducido."""
        free: list[int] = []
        for n in self.nodes:
            for dof, restringido in zip(n.dofs, n.restraints):
                if not restringido:
                    free.append(dof)
        return free

    def restrained_dofs(self) -> list[int]:
        """GDL con apoyo — desplazamiento conocido y reacción incógnita."""
        libres = set(self.free_dofs())
        return [d for d in range(self.n_dofs) if d not in libres]

    def prescribed_displacements(self) -> np.ndarray:
        """Vector U_c (ec. 1.6.1) colocado en posiciones globales."""
        u_c = np.zeros(self.n_dofs)
        for n in self.nodes:
            for dof, restringido, valor in zip(n.dofs, n.restraints,
                                               n.prescribed):
                if restringido:
                    u_c[dof] = valor
        return u_c
