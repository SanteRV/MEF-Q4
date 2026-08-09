"""Estructura del modelo FRAME 3D: nodos de 6 GDL + elementos de barra.

Para qué sirve: igual que las otras structure_*.py, pero con 6 GDL por
nodo (ux, uy, uz, rx, ry, rz), como exige 01.02.01.01 y .08.
"""
from __future__ import annotations
from dataclasses import dataclass, field

import numpy as np

from .frame_element import FrameElement
from .node_frame import NodeFrame


@dataclass
class StructureFrame:
    """El modelo frame completo: lista de nodos + lista de elementos."""
    nodes: list[NodeFrame] = field(default_factory=list)
    elements: list[FrameElement] = field(default_factory=list)

    @property
    def n_dofs(self) -> int:
        """Número total de GDL del sistema: 6 por nodo."""
        return 6 * len(self.nodes)

    def add_node(self, node: NodeFrame) -> None:
        self.nodes.append(node)

    def add_element(self, element: FrameElement) -> None:
        self.elements.append(element)

    def free_dofs(self) -> list[int]:
        """GDL sin apoyo — las incógnitas del sistema (01.02.01.08)."""
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
        """Vector v_c (ec. 2.9.1) colocado en posiciones globales."""
        v_c = np.zeros(self.n_dofs)
        for n in self.nodes:
            for dof, restringido, valor in zip(n.dofs, n.restraints,
                                               n.prescribed):
                if restringido:
                    v_c[dof] = valor
        return v_c
