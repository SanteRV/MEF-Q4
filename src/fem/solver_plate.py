"""Ensamblaje y solución del sistema MEF para el modelo PLATE.

Para qué sirve: la parte GLOBAL del método para placas a flexión, con la
misma secuencia que el documento teórico (cap. 01.01.06):

    1. Ensamblar K global (suma de los K^e de 12×12 según sus GDL).
    2. Ensamblar F global (cargas nodales Fz, Mx, My + cargas repartidas).
    3. Condiciones de borde por partición (ec. 5.3): reducir a GDL libres.
    4. Resolver K_ff · U_f = F_f (ec. 5.4).
    5. Reacciones en los apoyos (ec. 5.5): R = K_cf · U_f - F_c.
    6. Post-proceso por elemento: w, curvaturas, momentos y esfuerzos.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

from .structure_plate import StructurePlate


@dataclass
class ElementResultPlate:
    """Resultados de post-proceso de UN elemento plate."""
    element_id: int
    # Momentos [Mx, My, Mxy] evaluados en el centro y en las 4 esquinas
    moments_center: np.ndarray            # (3,)
    moments_at_corners: list[np.ndarray]  # 4 x (3,)
    # Esfuerzos [σx, σy, τxy] en la cara superior (z = +t/2), centro y esquinas
    stresses_center_top: np.ndarray            # (3,)
    stresses_at_corners_top: list[np.ndarray]  # 4 x (3,)
    # Desplazamiento w en el centro del elemento
    w_center: float = 0.0


@dataclass
class FEMResultPlate:
    """Resultado completo del análisis de placa."""
    K_global: np.ndarray         # matriz de rigidez global (n×n)
    F_global: np.ndarray         # vector de cargas global (n,)
    displacements: np.ndarray    # desplazamientos de TODOS los GDL (n,)
    reactions: np.ndarray        # reacciones (solo GDL restringidos)
    elements: list[ElementResultPlate] = field(default_factory=list)


def assemble_global_stiffness_plate(structure: StructurePlate) -> np.ndarray:
    """Ensambla K global sumando los K^e (12×12) en los GDL de cada elemento."""
    n = structure.n_dofs
    K = np.zeros((n, n))
    for el in structure.elements:
        ke = el.stiffness_matrix()
        dofs = el.global_dofs()
        for i_local, i_global in enumerate(dofs):
            for j_local, j_global in enumerate(dofs):
                K[i_global, j_global] += ke[i_local, j_local]
    return K


def assemble_load_vector_plate(structure: StructurePlate,
                               q_uniform: float = 0.0) -> np.ndarray:
    """Ensambla F global: cargas nodales + carga repartida q (ec. 3.19).

    q_uniform es una presión transversal (N/m², positiva hacia +Z) aplicada
    a TODOS los elementos; cada elemento la reparte q·A/4 a sus nodos.
    """
    F = np.zeros(structure.n_dofs)
    for node in structure.nodes:
        d = node.dofs
        F[d[0]] += node.load_fz
        F[d[1]] += node.load_mx
        F[d[2]] += node.load_my
    if q_uniform != 0.0:
        for el in structure.elements:
            fe = el.load_vector_uniform(q_uniform)
            for i_local, i_global in enumerate(el.global_dofs()):
                F[i_global] += fe[i_local]
    return F


def solve_plate(structure: StructurePlate,
                q_uniform: float = 0.0) -> FEMResultPlate:
    """Resuelve el modelo de placa completo (desplazamientos, reacciones,
    momentos y esfuerzos)."""
    K = assemble_global_stiffness_plate(structure)
    F = assemble_load_vector_plate(structure, q_uniform)

    free = structure.free_dofs()
    restrained = structure.restrained_dofs()

    # Partición del sistema (ec. 5.3) y solución del bloque libre (ec. 5.4)
    K_ff = K[np.ix_(free, free)]
    F_f = F[free]

    u = np.zeros(structure.n_dofs)
    if len(free) > 0:
        u_f = np.linalg.solve(K_ff, F_f)
        u[free] = u_f
    else:
        u_f = np.zeros(0)

    # Reacciones (ec. 5.5): fuerzas en los GDL restringidos
    reactions = np.zeros(structure.n_dofs)
    if len(restrained) > 0 and len(free) > 0:
        reactions[restrained] = K[np.ix_(restrained, free)] @ u_f - F[restrained]

    # Post-proceso por elemento: momentos y esfuerzos en centro y esquinas
    element_results: list[ElementResultPlate] = []
    for el in structure.elements:
        dofs = el.global_dofs()
        u_e = u[dofs]
        xs = [n.x for n in el.nodes]
        ys = [n.y for n in el.nodes]
        xc, yc = (min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0
        z_top = el.t / 2.0

        m_center = el.moments_at(xc, yc, u_e)
        m_corners = [el.moments_at(n.x, n.y, u_e) for n in el.nodes]
        _, s_center = el.strains_stresses_at(xc, yc, z_top, u_e)
        s_corners = [el.strains_stresses_at(n.x, n.y, z_top, u_e)[1]
                     for n in el.nodes]

        element_results.append(ElementResultPlate(
            element_id=el.id,
            moments_center=m_center,
            moments_at_corners=m_corners,
            stresses_center_top=s_center,
            stresses_at_corners_top=s_corners,
            w_center=el.w_at(xc, yc, u_e),
        ))

    return FEMResultPlate(
        K_global=K,
        F_global=F,
        displacements=u,
        reactions=reactions,
        elements=element_results,
    )
