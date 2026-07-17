"""Estructura del modelo PLATE: contenedor de nodos (3 GDL) + elementos.

Para qué sirve: igual que structure.py para el plane, pero con 3 GDL por
nodo (w, θx, θy). Responde lo que el solver necesita: número de GDL,
cuáles están libres y cuáles restringidos por apoyos.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from .node_plate import NodePlate
from .plate_element import PlateElement


@dataclass
class StructurePlate:
    """El modelo de placa completo: lista de nodos + lista de elementos."""
    nodes: list[NodePlate] = field(default_factory=list)
    elements: list[PlateElement] = field(default_factory=list)

    @property
    def n_dofs(self) -> int:
        """Número total de GDL del sistema: 3 por nodo (w, θx, θy)."""
        return 3 * len(self.nodes)

    def add_node(self, node: NodePlate) -> None:
        """Agrega un nodo al modelo (id único y consecutivo)."""
        self.nodes.append(node)

    def add_element(self, element: PlateElement) -> None:
        """Agrega un elemento plate (sus 4 nodos deben existir en el modelo)."""
        self.elements.append(element)

    def free_dofs(self) -> list[int]:
        """GDL sin apoyo — las incógnitas del sistema reducido."""
        free: list[int] = []
        for n in self.nodes:
            d = n.dofs
            if not n.restraint_w:
                free.append(d[0])
            if not n.restraint_rx:
                free.append(d[1])
            if not n.restraint_ry:
                free.append(d[2])
        return free

    def restrained_dofs(self) -> list[int]:
        """GDL con apoyo — desplazamiento conocido (0) y reacción incógnita."""
        libres = set(self.free_dofs())
        return [d for d in range(self.n_dofs) if d not in libres]
