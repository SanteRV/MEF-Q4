"""Ensamblaje y solución del sistema MEF con elementos Q4."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .structure import Structure


@dataclass
class ElementResult:
    element_id: int
    # Esfuerzos y deformaciones en los 4 puntos de Gauss (cada uno (3,) σx, σy, τxy)
    strains_at_gauss: list[np.ndarray]
    stresses_at_gauss: list[np.ndarray]
    # Esfuerzos y deformaciones evaluados directamente en los 4 nodos (esquinas)
    strains_at_corners: list[np.ndarray]
    stresses_at_corners: list[np.ndarray]


@dataclass
class FEMResult:
    K_global: np.ndarray
    F_global: np.ndarray
    displacements: np.ndarray
    reactions: np.ndarray
    elements: list[ElementResult]


def assemble_global_stiffness(structure: Structure) -> np.ndarray:
    n = structure.n_dofs
    K = np.zeros((n, n))
    for el in structure.elements:
        ke, _ = el.stiffness_matrix()
        dofs = el.global_dofs()
        for i_local, i_global in enumerate(dofs):
            for j_local, j_global in enumerate(dofs):
                K[i_global, j_global] += ke[i_local, j_local]
    return K


def assemble_load_vector(structure: Structure) -> np.ndarray:
    F = np.zeros(structure.n_dofs)
    for node in structure.nodes:
        F[node.dofs[0]] += node.load_x
        F[node.dofs[1]] += node.load_y
    return F


def solve(structure: Structure) -> FEMResult:
    K = assemble_global_stiffness(structure)
    F = assemble_load_vector(structure)

    free = structure.free_dofs()
    restrained = structure.restrained_dofs()

    K_ff = K[np.ix_(free, free)]
    F_f = F[free]

    u = np.zeros(structure.n_dofs)
    if len(free) > 0:
        u_f = np.linalg.solve(K_ff, F_f)
        u[free] = u_f
    else:
        u_f = np.zeros(0)

    reactions = np.zeros(structure.n_dofs)
    if len(restrained) > 0 and len(free) > 0:
        reactions[restrained] = K[np.ix_(restrained, free)] @ u_f - F[restrained]

    # Esfuerzos por elemento (en puntos de Gauss y en esquinas)
    element_results: list[ElementResult] = []
    from .q4_element import GAUSS_2X2
    for el in structure.elements:
        dofs = el.global_dofs()
        u_e = u[dofs]
        gp_strains, gp_stresses = [], []
        for xi, eta, _ in GAUSS_2X2:
            eps, sig = el.strains_stresses_at(xi, eta, u_e)
            gp_strains.append(eps)
            gp_stresses.append(sig)
        corner_eps_sig = el.strains_stresses_at_corners(u_e)
        corner_strains = [pair[0] for pair in corner_eps_sig]
        corner_stresses = [pair[1] for pair in corner_eps_sig]
        element_results.append(ElementResult(
            element_id=el.id,
            strains_at_gauss=gp_strains,
            stresses_at_gauss=gp_stresses,
            strains_at_corners=corner_strains,
            stresses_at_corners=corner_stresses,
        ))

    return FEMResult(
        K_global=K,
        F_global=F,
        displacements=u,
        reactions=reactions,
        elements=element_results,
    )
