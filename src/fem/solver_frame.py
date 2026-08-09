"""Ensamblaje y solución del modelo FRAME 3D (cap. 01.02.01.08 a .010).

Secuencia, en el orden del documento:

    1. Ensamblar K global sumando los K = Tᵀ·k·T de cada elemento
       (ec. 2.4.2) según sus 12 GDL.
    2. Ensamblar F global: cargas nodales + cargas nodales equivalentes de
       las cargas aplicadas sobre los elementos (-Tᵀ·Q_f, 01.02.01.07).
    3. Partición por condiciones de borde (ec. 2.9.1).
    4. Resolver K_ff·v_f = F_f − K_fc·v_c (ec. 2.9.2).
    5. Reacciones: K_cf·v_f + K_cc·v_c = F_c (ec. 2.9.3).
    6. Recuperación de resultados (01.02.01.010): u = T·v y Q = k·u + Q_f.
"""
from __future__ import annotations
from dataclasses import dataclass, field

import numpy as np

from .structure_frame import StructureFrame


@dataclass
class ElementResultFrame:
    """Resultados de post-proceso de UN elemento frame."""
    element_id: int
    local_displacements: np.ndarray   # (12,) u = T·v
    end_forces: np.ndarray            # (12,) Q = k·u + Q_f, en el SCL
    # Valores extremos de los diagramas, para el resumen de la UI
    axial_max: float = 0.0
    shear_y_max: float = 0.0
    shear_z_max: float = 0.0
    torsion_max: float = 0.0
    moment_y_max: float = 0.0
    moment_z_max: float = 0.0


@dataclass
class FEMResultFrame:
    """Resultado completo del análisis frame."""
    K_global: np.ndarray
    F_global: np.ndarray
    displacements: np.ndarray
    reactions: np.ndarray
    elements: list[ElementResultFrame] = field(default_factory=list)


def assemble_global_stiffness_frame(structure: StructureFrame) -> np.ndarray:
    """Ensambla K global sumando los K (12×12) de cada elemento."""
    n = structure.n_dofs
    K = np.zeros((n, n))
    for el in structure.elements:
        ke = el.stiffness_matrix()
        dofs = el.global_dofs()
        for i_local, i_global in enumerate(dofs):
            for j_local, j_global in enumerate(dofs):
                K[i_global, j_global] += ke[i_local, j_local]
    return K


def assemble_load_vector_frame(structure: StructureFrame) -> np.ndarray:
    """Ensambla F global: cargas nodales + equivalentes de las de elemento."""
    F = np.zeros(structure.n_dofs)
    for node in structure.nodes:
        for dof, carga in zip(node.dofs, node.loads):
            F[dof] += carga
    for el in structure.elements:
        if not el.loads:
            continue
        fe = el.equivalent_nodal_loads()
        for i_local, i_global in enumerate(el.global_dofs()):
            F[i_global] += fe[i_local]
    return F


def solve_frame(structure: StructureFrame) -> FEMResultFrame:
    """Resuelve el modelo frame completo."""
    K = assemble_global_stiffness_frame(structure)
    F = assemble_load_vector_frame(structure)

    free = structure.free_dofs()
    restrained = structure.restrained_dofs()

    v_c = structure.prescribed_displacements()[restrained]
    K_ff = K[np.ix_(free, free)]
    K_fc = K[np.ix_(free, restrained)]
    F_f = F[free]

    # Ec. 2.9.2
    v = np.zeros(structure.n_dofs)
    v[restrained] = v_c
    if len(free) > 0:
        v_f = np.linalg.solve(K_ff, F_f - K_fc @ v_c)
        v[free] = v_f
    else:
        v_f = np.zeros(0)

    # Ec. 2.9.3
    reactions = np.zeros(structure.n_dofs)
    if len(restrained) > 0:
        K_cf = K[np.ix_(restrained, free)]
        K_cc = K[np.ix_(restrained, restrained)]
        reactions[restrained] = (
            (K_cf @ v_f if len(free) > 0 else 0.0)
            + K_cc @ v_c
            - F[restrained]
        )

    element_results: list[ElementResultFrame] = []
    for el in structure.elements:
        v_e = v[el.global_dofs()]
        Q = el.end_forces(v_e)
        picos = {}
        for comp in ("P", "Vy", "Vz", "T", "My", "Mz"):
            _, vals = el.diagram(comp, v_e)
            picos[comp] = float(np.max(np.abs(vals))) if vals.size else 0.0
        element_results.append(ElementResultFrame(
            element_id=el.id,
            local_displacements=el.local_displacements(v_e),
            end_forces=Q,
            axial_max=picos["P"],
            shear_y_max=picos["Vy"],
            shear_z_max=picos["Vz"],
            torsion_max=picos["T"],
            moment_y_max=picos["My"],
            moment_z_max=picos["Mz"],
        ))

    return FEMResultFrame(
        K_global=K,
        F_global=F,
        displacements=v,
        reactions=reactions,
        elements=element_results,
    )
