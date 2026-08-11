"""Modelo unificado: un solo modelo con frame + plane + plate + shell.

Corrección 2 (puntos 1, 3 y 6): el aplicativo debe tener UNA ventana de
modelo donde conviven todos los tipos de elemento, no un modo separado
por formulación. Este módulo define ese modelo de datos:

    NodeModel   nodo de 6 GDL (u, v, w, θx, θy, θz) — el superconjunto de
                los GDL que usa cualquiera de los cuatro elementos.
    Member      un elemento dibujado: tipo + nodos + nombre de sección.
    Model       nodos + miembros + biblioteca de secciones.

Los GDL de cada elemento se mapean sobre los 6 del nodo:

    frame  6/nodo   u, v, w, θx, θy, θz      (todos)
    plane  2/nodo   u, v
    plate  3/nodo   w, θx, θy
    shell  5/nodo   u, v, w, θx, θy          (plane + plate)

Un GDL que ningún elemento conectado activa queda SIN RIGIDEZ; el solver
lo detecta y lo restringe automáticamente para que el sistema no sea
singular (por ejemplo θz en un modelo hecho solo de losas).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .sections import Section, SectionLibrary, TipoElemento, default_library


# Índice de cada GDL dentro del nodo (orden fijo del modelo unificado)
DOF_U, DOF_V, DOF_W, DOF_RX, DOF_RY, DOF_RZ = range(6)
DOF_NAMES = ("u", "v", "w", "θx", "θy", "θz")

# GDL que ACTIVA cada tipo de elemento, en el orden en que su núcleo los
# numera. Esta tabla es el corazón del ensamblaje unificado: traduce el
# GDL local de cada formulación al GDL global del nodo de 6 componentes.
DOF_MAP: dict[TipoElemento, tuple[int, ...]] = {
    "frame": (DOF_U, DOF_V, DOF_W, DOF_RX, DOF_RY, DOF_RZ),
    "plane": (DOF_U, DOF_V),
    "plate": (DOF_W, DOF_RX, DOF_RY),
    "shell": (DOF_U, DOF_V, DOF_W, DOF_RX, DOF_RY),
}

# Cuántos nodos tiene cada tipo de elemento
N_NODES: dict[TipoElemento, int] = {
    "frame": 2, "plane": 4, "plate": 4, "shell": 4,
}


@dataclass
class NodeModel:
    """Nodo del modelo unificado: 6 GDL, apoyos, desplazamientos y cargas.

    Los índices globales son 6·id + k, con k el orden de DOF_NAMES.
    """
    id: int
    x: float
    y: float
    z: float = 0.0

    # Apoyos: True = restringido. Se asignan en lote desde la UI
    # (Corrección 2, paso 4).
    restraints: list[bool] = field(
        default_factory=lambda: [False] * 6)
    # Desplazamientos prescritos U_c (Corrección 2, paso 5)
    prescribed: list[float] = field(
        default_factory=lambda: [0.0] * 6)
    # Cargas nodales [Fx, Fy, Fz, Mx, My, Mz]
    loads: list[float] = field(
        default_factory=lambda: [0.0] * 6)

    @property
    def coords(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)

    @property
    def dofs(self) -> tuple[int, ...]:
        """Los 6 índices globales del nodo."""
        base = 6 * self.id
        return tuple(base + k for k in range(6))

    # ---- helpers de apoyo, pensados para la asignación en lote ----
    def fix_all(self) -> None:
        """Empotramiento: restringe los 6 GDL."""
        self.restraints = [True] * 6

    def pin(self) -> None:
        """Apoyo simple (rótula): restringe las 3 traslaciones."""
        self.restraints = [True, True, True, False, False, False]

    def free(self) -> None:
        """Libera todos los GDL del nodo."""
        self.restraints = [False] * 6


@dataclass
class Member:
    """Un elemento dibujado por el usuario.

    Guarda el TIPO, los nodos que conecta y el NOMBRE de la sección
    asignada (la biblioteca resuelve el nombre a propiedades). Así el
    usuario cambia una sección y todos sus elementos se actualizan.
    """
    id: int
    tipo: TipoElemento
    node_ids: list[int]
    section: str
    # Solo frame: rotación de ejes locales (ec. 2.2.1) y cargas de tramo
    psi: float = 0.0
    loads: list = field(default_factory=list)
    # Solo áreas: presión transversal uniforme q (N/m²)
    q: float = 0.0

    def __post_init__(self) -> None:
        n_req = N_NODES.get(self.tipo)
        if n_req is None:
            raise ValueError(f"Tipo de elemento desconocido: {self.tipo!r}")
        if len(self.node_ids) != n_req:
            raise ValueError(
                f"Un elemento {self.tipo} requiere {n_req} nodos, "
                f"se recibieron {len(self.node_ids)}."
            )

    @property
    def dof_map(self) -> tuple[int, ...]:
        """GDL del nodo que este elemento activa."""
        return DOF_MAP[self.tipo]

    def global_dofs(self) -> list[int]:
        """Índices globales de los GDL del elemento, en el orden de su núcleo.

        Para qué sirve: es el mapa del ensamblaje. Recorre los nodos del
        elemento y, por cada uno, toma solo los GDL que su formulación usa.
        """
        dofs: list[int] = []
        for nid in self.node_ids:
            base = 6 * nid
            dofs.extend(base + k for k in self.dof_map)
        return dofs


@dataclass
class Model:
    """El modelo completo: nodos, miembros y biblioteca de secciones."""
    nodes: list[NodeModel] = field(default_factory=list)
    members: list[Member] = field(default_factory=list)
    sections: SectionLibrary = field(default_factory=default_library)

    # ------------------------------------------------------------ nodos
    @property
    def n_dofs(self) -> int:
        """GDL totales: 6 por nodo."""
        return 6 * len(self.nodes)

    def node(self, nid: int) -> NodeModel:
        """Nodo por id (los ids son consecutivos y coinciden con el índice)."""
        return self.nodes[nid]

    def add_node(self, x: float, y: float, z: float = 0.0,
                 tol: float = 1e-9) -> NodeModel:
        """Añade un nodo, REUSANDO el existente si ya hay uno en ese punto.

        Para qué sirve: al dibujar, dos elementos que comparten una esquina
        deben compartir el nodo — si no, la estructura quedaría desconectada.
        """
        for n in self.nodes:
            if (abs(n.x - x) < tol and abs(n.y - y) < tol
                    and abs(n.z - z) < tol):
                return n
        n = NodeModel(id=len(self.nodes), x=float(x), y=float(y), z=float(z))
        self.nodes.append(n)
        return n

    # --------------------------------------------------------- miembros
    def add_member(self, tipo: TipoElemento, node_ids: list[int],
                   section: str, **kwargs) -> Member:
        """Añade un elemento validando que la sección exista y sea del tipo."""
        sec = self.sections.get(section)
        if getattr(sec, "tipo", None) != tipo:
            raise ValueError(
                f"La sección {section!r} es de tipo "
                f"{getattr(sec, 'tipo', '?')!r}, no {tipo!r}."
            )
        m = Member(id=len(self.members), tipo=tipo,
                   node_ids=list(node_ids), section=section, **kwargs)
        self.members.append(m)
        return m

    def section_of(self, m: Member) -> Section:
        """Sección asignada a un miembro, ya resuelta desde la biblioteca."""
        return self.sections.get(m.section)

    def members_of_type(self, tipo: TipoElemento) -> list[Member]:
        """Miembros de un tipo — para tablas de resultados por familia."""
        return [m for m in self.members if m.tipo == tipo]

    # ------------------------------------------------- grados de libertad
    def active_dofs(self) -> np.ndarray:
        """Máscara (n_dofs,) de GDL que reciben rigidez de algún elemento.

        Para qué sirve: en un modelo mixto no todos los nodos usan sus 6
        GDL. Un nodo de losa (shell) nunca activa θz; si ese GDL quedara
        libre y sin rigidez, K sería singular. El solver restringe los
        GDL inactivos automáticamente.
        """
        activo = np.zeros(self.n_dofs, dtype=bool)
        for m in self.members:
            activo[m.global_dofs()] = True
        return activo

    def free_dofs(self) -> list[int]:
        """GDL libres: activos y sin apoyo."""
        activo = self.active_dofs()
        libres: list[int] = []
        for n in self.nodes:
            for k, gdl in enumerate(n.dofs):
                if activo[gdl] and not n.restraints[k]:
                    libres.append(gdl)
        return libres

    def restrained_dofs(self) -> list[int]:
        """GDL restringidos: con apoyo, o inactivos (sin rigidez)."""
        libres = set(self.free_dofs())
        return [d for d in range(self.n_dofs) if d not in libres]

    def prescribed_vector(self) -> np.ndarray:
        """Vector U_c con los desplazamientos impuestos en sus posiciones."""
        u_c = np.zeros(self.n_dofs)
        for n in self.nodes:
            for k, gdl in enumerate(n.dofs):
                if n.restraints[k]:
                    u_c[gdl] = n.prescribed[k]
        return u_c

    # ------------------------------------------------------------ apoyos
    def assign_support(self, node_ids: list[int], tipo: str) -> None:
        """Asigna un apoyo a VARIOS nodos de una vez (Corrección 2, paso 4).

        tipo: "empotrado" (6 GDL), "simple" (3 traslaciones) o "libre".
        """
        for nid in node_ids:
            n = self.node(nid)
            if tipo == "empotrado":
                n.fix_all()
            elif tipo == "simple":
                n.pin()
            elif tipo == "libre":
                n.free()
            else:
                raise ValueError(
                    "tipo de apoyo debe ser 'empotrado', 'simple' o 'libre'."
                )

    def describe(self) -> str:
        """Resumen legible del modelo, para la barra de estado y el reporte."""
        por_tipo = {t: len(self.members_of_type(t))
                    for t in ("frame", "plane", "plate", "shell")}
        partes = [f"{v} {k}" for k, v in por_tipo.items() if v]
        detalle = ", ".join(partes) if partes else "sin elementos"
        n_libres = len(self.free_dofs())
        return (f"{len(self.nodes)} nodos, {len(self.members)} elementos "
                f"({detalle}) — {n_libres} GDL libres de {self.n_dofs}")
