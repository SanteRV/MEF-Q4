"""Solver del modelo unificado (frame + plane + plate + shell mezclados).

Para qué sirve: ensambla y resuelve UN solo sistema K·U = F donde conviven
los cuatro tipos de elemento. No reimplementa ninguna formulación: para
cada miembro construye el objeto del núcleo correspondiente (Q4Element,
PlateElement, ShellElement, FrameElement), le pide su K^e y lo coloca en
los GDL globales que ese elemento activa (tabla DOF_MAP de model.py).

Secuencia (cap. 01.01.06 del documento teórico):
    1. K global = suma de los K^e mapeados a los GDL del modelo.
    2. F global = cargas nodales + cargas de tramo (frame) + presión (áreas).
    3. Partición por condiciones de borde (ec. 1.6.1), con U_c prescrito.
    4. K_ff·U_f = F_f − K_fc·U_c   (ec. 1.6.2)
    5. Reacciones R = K_cf·U_f + K_cc·U_c − F_c   (ec. 1.6.3)
    6. Post-proceso por miembro, delegado a cada núcleo.

Los GDL sin rigidez (por ejemplo θz en un modelo de puras losas) se tratan
como restringidos, de modo que el sistema reducido nunca queda singular.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .model import Model, Member
from .node import Node
from .node_plate import NodePlate
from .node_shell import NodeShell
from .node_frame import NodeFrame
from .q4_element import Q4Element
from .plate_element import PlateElement
from .shell_element import ShellElement
from .frame_element import FrameElement
from .sections import AreaSection, FrameSection


@dataclass
class MemberResult:
    """Resultados de post-proceso de UN miembro del modelo unificado."""
    member_id: int
    tipo: str
    displacements: np.ndarray            # GDL del miembro, en orden del núcleo
    # Áreas (plane / plate / shell)
    stresses: list[np.ndarray] = field(default_factory=list)
    moments: list[np.ndarray] = field(default_factory=list)
    w_center: float | None = None
    # Frame
    end_forces: np.ndarray | None = None


@dataclass
class ModelResult:
    """Resultado completo del análisis del modelo unificado."""
    K_global: np.ndarray
    F_global: np.ndarray
    displacements: np.ndarray
    reactions: np.ndarray
    members: list[MemberResult] = field(default_factory=list)
    active_dofs: np.ndarray | None = None


# ---------------------------------------------------------------- núcleos
def _build_core_element(model: Model, m: Member):
    """Crea el objeto del núcleo que corresponde al miembro.

    Para qué sirve: aísla el "traductor" entre el modelo unificado y las
    cuatro formulaciones ya validadas. Los nodos que se pasan a cada núcleo
    son copias locales numeradas 0..n-1: solo importan sus coordenadas,
    porque el mapeo real de GDL lo hace Member.global_dofs().
    """
    sec = model.section_of(m)
    nodos = [model.node(nid) for nid in m.node_ids]

    if m.tipo == "plane":
        assert isinstance(sec, AreaSection)
        locales = [Node(id=i, x=n.x, y=n.y) for i, n in enumerate(nodos)]
        return Q4Element(id=m.id, nodes=locales, E=sec.E, nu=sec.nu,
                         t=sec.t, plane_stress=sec.plane_stress)

    if m.tipo == "plate":
        assert isinstance(sec, AreaSection)
        locales = [NodePlate(id=i, x=n.x, y=n.y) for i, n in enumerate(nodos)]
        return PlateElement(id=m.id, nodes=locales, E=sec.E, nu=sec.nu,
                            t=sec.t)

    if m.tipo == "shell":
        assert isinstance(sec, AreaSection)
        locales = [NodeShell(id=i, x=n.x, y=n.y) for i, n in enumerate(nodos)]
        return ShellElement(id=m.id, nodes=locales, E=sec.E, nu=sec.nu,
                            t=sec.t)

    if m.tipo == "frame":
        assert isinstance(sec, FrameSection)
        locales = [NodeFrame(id=i, x=n.x, y=n.y, z=n.z)
                   for i, n in enumerate(nodos)]
        return FrameElement(id=m.id, nodes=locales, E=sec.E, G=sec.G,
                            A=sec.A, Iy=sec.Iy, Iz=sec.Iz, J=sec.J,
                            psi=m.psi, loads=list(m.loads))

    raise ValueError(f"Tipo de elemento no soportado: {m.tipo!r}")


def _element_load_vector(core, m: Member) -> np.ndarray | None:
    """Vector de cargas del elemento (presión en áreas, tramo en frames)."""
    if m.tipo in ("plate", "shell") and m.q != 0.0:
        return core.load_vector_uniform(m.q)
    if m.tipo == "frame" and m.loads:
        return core.equivalent_nodal_loads()
    return None


# ------------------------------------------------------------- ensamblaje
def assemble_model(model: Model) -> tuple[np.ndarray, np.ndarray]:
    """Ensambla K y F globales del modelo unificado."""
    n = model.n_dofs
    K = np.zeros((n, n))
    F = np.zeros(n)

    # Cargas nodales directas
    for nd in model.nodes:
        for k, gdl in enumerate(nd.dofs):
            F[gdl] += nd.loads[k]

    # Aporte de cada miembro
    for m in model.members:
        core = _build_core_element(model, m)
        ke = core.stiffness_matrix()
        if isinstance(ke, tuple):          # Q4Element devuelve (K, gauss_data)
            ke = ke[0]
        dofs = m.global_dofs()
        K[np.ix_(dofs, dofs)] += ke
        fe = _element_load_vector(core, m)
        if fe is not None:
            F[dofs] += fe
    return K, F


def solve_model(model: Model) -> ModelResult:
    """Resuelve el modelo unificado completo."""
    K, F = assemble_model(model)
    n = model.n_dofs

    activo = model.active_dofs()
    free = model.free_dofs()
    restrained = model.restrained_dofs()
    u_c = model.prescribed_vector()

    u = np.zeros(n)
    u[restrained] = u_c[restrained]
    u_f = np.zeros(0)
    if free:
        K_ff = K[np.ix_(free, free)]
        K_fc = K[np.ix_(free, restrained)]
        F_f = F[free] - K_fc @ u_c[restrained]
        try:
            u_f = np.linalg.solve(K_ff, F_f)
        except np.linalg.LinAlgError as e:
            raise ValueError(
                "El sistema no tiene solución única: la estructura está "
                "insuficientemente restringida o hay elementos desconectados. "
                f"(detalle: {e})"
            ) from e
        u[free] = u_f

    # Reacciones solo en GDL con apoyo real (los inactivos no son apoyos)
    reactions = np.zeros(n)
    if restrained:
        K_cf = K[np.ix_(restrained, free)] if free else np.zeros((len(restrained), 0))
        K_cc = K[np.ix_(restrained, restrained)]
        R = (K_cf @ u_f if free else 0.0) + K_cc @ u_c[restrained] - F[restrained]
        reactions[restrained] = R
        reactions[~activo] = 0.0

    # Post-proceso delegado a cada núcleo
    resultados: list[MemberResult] = []
    for m in model.members:
        core = _build_core_element(model, m)
        u_e = u[m.global_dofs()]
        r = MemberResult(member_id=m.id, tipo=m.tipo, displacements=u_e)
        xs = [model.node(i).x for i in m.node_ids]
        ys = [model.node(i).y for i in m.node_ids]
        if m.tipo == "plane":
            r.stresses = [sig for _, sig in
                          core.strains_stresses_at_corners(u_e)]
        elif m.tipo in ("plate", "shell"):
            xc = (min(xs) + max(xs)) / 2.0
            yc = (min(ys) + max(ys)) / 2.0
            r.moments = [core.moments_at(xc, yc, u_e)]
            r.w_center = core.w_at(xc, yc, u_e)
            if m.tipo == "shell":
                r.stresses = [core.total_stress_at(xc, yc, core.t / 2.0, u_e)]
        elif m.tipo == "frame":
            r.end_forces = core.end_forces(u_e)
        resultados.append(r)

    return ModelResult(K_global=K, F_global=F, displacements=u,
                       reactions=reactions, members=resultados,
                       active_dofs=activo)
