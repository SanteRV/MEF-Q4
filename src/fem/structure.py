"""Estructura completa: contenedor de nodos + elementos Q4.

Para qué sirve: agrupa todo el modelo (malla, apoyos y cargas viven en los
nodos; material y espesor viven en los elementos) y responde las preguntas
que el solver necesita para armar y reducir el sistema K·u = F:
    - cuántos GDL tiene el sistema (n_dofs),
    - cuáles GDL están libres (free_dofs) y
    - cuáles están restringidos por apoyos (restrained_dofs).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from .node import Node
from .q4_element import Q4Element


@dataclass
class Structure:
    """El modelo completo que se analiza: lista de nodos + lista de elementos."""
    nodes: list[Node] = field(default_factory=list)
    elements: list[Q4Element] = field(default_factory=list)

    @property
    def n_dofs(self) -> int:
        """Número total de GDL del sistema: 2 por nodo (ux, uy)."""
        return 2 * len(self.nodes)

    def add_node(self, node: Node) -> None:
        """Agrega un nodo al modelo (el id debe ser único y consecutivo)."""
        self.nodes.append(node)

    def add_element(self, element: Q4Element) -> None:
        """Agrega un elemento Q4 (sus 4 nodos deben existir en el modelo)."""
        self.elements.append(element)

    def free_dofs(self) -> list[int]:
        """GDL SIN apoyo — las incógnitas del sistema reducido K_ff·u_f = F_f.

        Para qué sirve: aplicar las condiciones de borde. Al quedarse solo
        con filas/columnas de GDL libres, los desplazamientos conocidos
        (= 0 en los apoyos) salen del sistema y este se vuelve resoluble.
        """
        free = []
        for n in self.nodes:
            if not n.restraint_x:
                free.append(n.dofs[0])
            if not n.restraint_y:
                free.append(n.dofs[1])
        return free

    def restrained_dofs(self) -> list[int]:
        """GDL CON apoyo — donde el desplazamiento es 0 y aparecen reacciones."""
        return [d for d in range(self.n_dofs) if d not in self.free_dofs()]
