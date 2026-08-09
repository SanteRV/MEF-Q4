"""Ensamblaje y solución del sistema MEF para el modelo FLAT SHELL.

Misma secuencia que el documento (cap. 01.01.06 y 01.01.07), ahora con
elementos de 20 GDL:

    1. Ensamblar K global sumando los K^e (20×20) según sus GDL.
       El ejemplo de la figura 15 y la tabla 2 del documento es exactamente
       este caso: 9 nodos x 5 GDL = matriz global de 45x45.
    2. Ensamblar F global (cargas nodales + carga transversal repartida).
    3. Partición por condiciones de borde (ec. 1.6.1).
    4. Resolver K_ff·U_f = F_f − K_fc·U_c (ec. 1.6.2).
    5. Reacciones: K_cf·U_f + K_cc·U_c = F_c (ec. 1.6.3).
    6. Recuperación de resultados: ε = B·q y σ = D·B·q (cap. 01.01.07).
"""
from __future__ import annotations
from dataclasses import dataclass, field

import numpy as np

from .structure_shell import StructureShell


@dataclass
class ElementResultShell:
    """Resultados de post-proceso de UN elemento shell."""
    element_id: int
    # Deformaciones y esfuerzos (6 componentes: 3 de membrana + 3 de flexión)
    # evaluados en el centro del elemento, en la cara superior z = +t/2
    strains_center: np.ndarray          # (6,)
    stresses_center: np.ndarray         # (6,)
    # Esfuerzo total de la fibra superior [σx, σy, τxy] = membrana + flexión
    stress_total_top: np.ndarray        # (3,)
    stress_total_bottom: np.ndarray     # (3,)
    # Momentos [Mx, My, Mxy] en el centro y desplazamiento transversal
    moments_center: np.ndarray          # (3,)
    w_center: float = 0.0


@dataclass
class FEMResultShell:
    """Resultado completo del análisis flat shell."""
    K_global: np.ndarray
    F_global: np.ndarray
    displacements: np.ndarray
    reactions: np.ndarray
    elements: list[ElementResultShell] = field(default_factory=list)


def assemble_global_stiffness_shell(structure: StructureShell) -> np.ndarray:
    """Ensambla K global sumando los K^e (20×20) en los GDL de cada elemento."""
    n = structure.n_dofs
    K = np.zeros((n, n))
    for el in structure.elements:
        ke = el.stiffness_matrix()
        dofs = el.global_dofs()
        for i_local, i_global in enumerate(dofs):
            for j_local, j_global in enumerate(dofs):
                K[i_global, j_global] += ke[i_local, j_local]
    return K


def assemble_load_vector_shell(structure: StructureShell,
                               q_uniform: float = 0.0,
                               load_case: str = "nodos") -> np.ndarray:
    """Ensambla F global: cargas nodales + carga transversal repartida.

    load_case usa los mismos dos casos del plate: "nodos" (ec. 1.3.19),
    "vigas_y" y "vigas_x" (ec. 1.3.20). La carga en el plano se aplica
    directamente como cargas nodales Fx, Fy.
    """
    F = np.zeros(structure.n_dofs)
    for node in structure.nodes:
        for dof, carga in zip(node.dofs, node.loads):
            F[dof] += carga
    if q_uniform != 0.0:
        for el in structure.elements:
            if load_case == "nodos":
                fe = el.load_vector_uniform(q_uniform)
            elif load_case in ("vigas_x", "vigas_y"):
                fe = el.load_vector_one_way(q_uniform, span=load_case[-1])
            else:
                raise ValueError(
                    "load_case debe ser 'nodos', 'vigas_x' o 'vigas_y'.")
            for i_local, i_global in enumerate(el.global_dofs()):
                F[i_global] += fe[i_local]
    return F


def solve_shell(structure: StructureShell,
                q_uniform: float = 0.0,
                load_case: str = "nodos") -> FEMResultShell:
    """Resuelve el modelo flat shell completo."""
    K = assemble_global_stiffness_shell(structure)
    F = assemble_load_vector_shell(structure, q_uniform, load_case)

    free = structure.free_dofs()
    restrained = structure.restrained_dofs()

    u_c = structure.prescribed_displacements()[restrained]
    K_ff = K[np.ix_(free, free)]
    K_fc = K[np.ix_(free, restrained)]
    F_f = F[free]

    # Ec. 1.6.2
    u = np.zeros(structure.n_dofs)
    u[restrained] = u_c
    if len(free) > 0:
        u_f = np.linalg.solve(K_ff, F_f - K_fc @ u_c)
        u[free] = u_f
    else:
        u_f = np.zeros(0)

    # Ec. 1.6.3
    reactions = np.zeros(structure.n_dofs)
    if len(restrained) > 0:
        K_cf = K[np.ix_(restrained, free)]
        K_cc = K[np.ix_(restrained, restrained)]
        reactions[restrained] = (
            (K_cf @ u_f if len(free) > 0 else 0.0)
            + K_cc @ u_c
            - F[restrained]
        )

    element_results: list[ElementResultShell] = []
    for el in structure.elements:
        d = u[el.global_dofs()]
        xs = [n.x for n in el.nodes]
        ys = [n.y for n in el.nodes]
        xc, yc = (min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0
        z_top = el.t / 2.0
        eps, sig = el.strains_stresses_at(xc, yc, z_top, d)
        element_results.append(ElementResultShell(
            element_id=el.id,
            strains_center=eps,
            stresses_center=sig,
            stress_total_top=el.total_stress_at(xc, yc, z_top, d),
            stress_total_bottom=el.total_stress_at(xc, yc, -z_top, d),
            moments_center=el.moments_at(xc, yc, d),
            w_center=el.w_at(xc, yc, d),
        ))

    return FEMResultShell(
        K_global=K,
        F_global=F,
        displacements=u,
        reactions=reactions,
        elements=element_results,
    )
