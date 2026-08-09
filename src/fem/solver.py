"""Ensamblaje y solución del sistema MEF con elementos Q4.

Para qué sirve: este módulo ejecuta la parte GLOBAL del método —
lo que viene después de que cada elemento calculó su K^e:

    1. Ensamblar la matriz de rigidez global K (cap. 01.01.06.01).
    2. Ensamblar el vector de cargas F (cap. 01.01.06.02).
    3. Aplicar condiciones de borde por partición f/c (ec. 1.6.1).
    4. Resolver K_ff·U_f = F_f − K_fc·U_c (ec. 1.6.2).
    5. Reacciones en los apoyos: K_cf·U_f + K_cc·U_c = F_c (ec. 1.6.3).
    6. Recuperación de resultados (cap. 01.01.07): deformaciones y
       esfuerzos por elemento (ε = B·q, σ = D·ε) en los puntos de Gauss
       y en las esquinas.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .structure import Structure


@dataclass
class ElementResult:
    """Resultados de post-proceso de UN elemento (para tablas y gráficos)."""
    element_id: int
    # Esfuerzos y deformaciones en los 4 puntos de Gauss (cada uno (3,): σx, σy, τxy)
    strains_at_gauss: list[np.ndarray]
    stresses_at_gauss: list[np.ndarray]
    # Esfuerzos y deformaciones evaluados directamente en los 4 nodos (esquinas)
    strains_at_corners: list[np.ndarray]
    stresses_at_corners: list[np.ndarray]


@dataclass
class FEMResult:
    """Resultado completo del análisis — todo lo que la UI y los exports muestran."""
    K_global: np.ndarray        # matriz de rigidez global (n×n)
    F_global: np.ndarray        # vector de cargas global (n,)
    displacements: np.ndarray   # desplazamientos de TODOS los GDL (n,)
    reactions: np.ndarray       # reacciones (≠ 0 solo en GDL restringidos)
    elements: list[ElementResult]


def assemble_global_stiffness(structure: Structure) -> np.ndarray:
    """Ensambla la matriz de rigidez global K sumando los K^e de cada elemento.

    Para qué sirve: expresa el equilibrio de TODA la estructura. Cada
    entrada K^e[a, b] del elemento se suma en K[dofs[a], dofs[b]], donde
    dofs es el mapa local → global del elemento (método de rigidez directa).
    Los elementos que comparten nodos suman rigidez en las mismas posiciones.
    """
    n = structure.n_dofs
    K = np.zeros((n, n))
    for el in structure.elements:
        ke, _ = el.stiffness_matrix()      # K^e (8×8) del elemento
        dofs = el.global_dofs()            # mapa local → global (8 índices)
        for i_local, i_global in enumerate(dofs):
            for j_local, j_global in enumerate(dofs):
                K[i_global, j_global] += ke[i_local, j_local]
    return K


def assemble_load_vector(structure: Structure) -> np.ndarray:
    """Ensambla el vector global de cargas F con las cargas nodales.

    Para qué sirve: es el lado derecho del sistema K·u = F. Cada carga
    puntual (Fx, Fy) del nodo entra en las posiciones de sus GDL.
    """
    F = np.zeros(structure.n_dofs)
    for node in structure.nodes:
        F[node.dofs[0]] += node.load_x
        F[node.dofs[1]] += node.load_y
    return F


def solve(structure: Structure) -> FEMResult:
    """Resuelve el modelo completo y devuelve desplazamientos, reacciones y esfuerzos.

    Secuencia (pasos 9 a 15 del aplicativo):
        ensamblar K y F → reducir por apoyos → resolver → reacciones → σ, ε.
    """
    K = assemble_global_stiffness(structure)
    F = assemble_load_vector(structure)

    # Condiciones de borde por partición (ec. 1.6.1): f = libres, c = restringidos
    #     | K_ff  K_fc | | U_f |   | F_f |
    #     | K_cf  K_cc | | U_c | = | F_c |
    free = structure.free_dofs()
    restrained = structure.restrained_dofs()

    # U_c: desplazamientos conocidos en los apoyos (0 en un apoyo rígido)
    u_c = structure.prescribed_displacements()[restrained]

    K_ff = K[np.ix_(free, free)]
    K_fc = K[np.ix_(free, restrained)]
    F_f = F[free]

    # Ec. 1.6.2:  K_ff·U_f + K_fc·U_c = F_f  →  K_ff·U_f = F_f − K_fc·U_c
    # El término K_fc·U_c es la carga equivalente de los asentamientos
    # impuestos; con U_c = 0 se anula y queda el caso clásico K_ff·U_f = F_f.
    u = np.zeros(structure.n_dofs)
    u[restrained] = u_c
    if len(free) > 0:
        u_f = np.linalg.solve(K_ff, F_f - K_fc @ u_c)
        u[free] = u_f
    else:
        u_f = np.zeros(0)

    # Ec. 1.6.3:  K_cf·U_f + K_cc·U_c = F_c. F_c es la fuerza TOTAL en el
    # apoyo; la reacción es esa fuerza menos la carga externa ya aplicada
    # sobre ese mismo GDL.
    reactions = np.zeros(structure.n_dofs)
    if len(restrained) > 0:
        K_cf = K[np.ix_(restrained, free)]
        K_cc = K[np.ix_(restrained, restrained)]
        reactions[restrained] = (
            (K_cf @ u_f if len(free) > 0 else 0.0)
            + K_cc @ u_c
            - F[restrained]
        )

    # Post-proceso por elemento: con los desplazamientos u_e del elemento
    # se recuperan ε = B·q y σ = D·ε en los puntos de Gauss (donde la
    # integración es más precisa) y en las esquinas (donde el usuario
    # suele querer leerlos).
    element_results: list[ElementResult] = []
    from .q4_element import GAUSS_2X2
    for el in structure.elements:
        dofs = el.global_dofs()
        u_e = u[dofs]                      # desplazamientos del elemento (8,)
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
