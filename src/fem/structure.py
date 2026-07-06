"""Estructura completa: contenedor de nodos + elementos Q4."""
from __future__ import annotations
from dataclasses import dataclass, field
from .node import Node
from .q4_element import Q4Element


@dataclass
class Structure:
    nodes: list[Node] = field(default_factory=list)
    elements: list[Q4Element] = field(default_factory=list)

    @property
    def n_dofs(self) -> int:
        return 2 * len(self.nodes)

    def add_node(self, node: Node) -> None:
        self.nodes.append(node)

    def add_element(self, element: Q4Element) -> None:
        self.elements.append(element)

    def free_dofs(self) -> list[int]:
        free = []
        for n in self.nodes:
            if not n.restraint_x:
                free.append(n.dofs[0])
            if not n.restraint_y:
                free.append(n.dofs[1])
        return free

    def restrained_dofs(self) -> list[int]:
        return [d for d in range(self.n_dofs) if d not in self.free_dofs()]
