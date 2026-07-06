"""Elemento cuadrilátero bilineal Q4 (tensión plana / deformación plana).

Convención de nodos (CCW desde la esquina superior derecha):
    2 --- 1
    |     |
    3 --- 4

Coordenadas naturales (ξ, η) ∈ [-1, 1]:
    N1 = (+,+),  N2 = (-,+),  N3 = (-,-),  N4 = (+,-)

Funciones de forma: Ni = 1/4 (1 + ξ·ξi) (1 + η·ηi)
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .node import Node


# Coordenadas naturales de los nodos del Q4 (orden 1,2,3,4)
NATURAL_COORDS = np.array([
    [+1.0, +1.0],   # nodo 1  (++)
    [-1.0, +1.0],   # nodo 2  (-+)
    [-1.0, -1.0],   # nodo 3  (--)
    [+1.0, -1.0],   # nodo 4  (+-)
])

# Puntos de Gauss 2x2 y pesos (mismo orden que los nodos: ++,-+,--,+-)
_G = 1.0 / np.sqrt(3.0)
GAUSS_2X2 = [
    (+_G, +_G, 1.0),   # GP1  (++)
    (-_G, +_G, 1.0),   # GP2  (-+)
    (-_G, -_G, 1.0),   # GP3  (--)
    (+_G, -_G, 1.0),   # GP4  (+-)
]


def shape_functions(xi: float, eta: float) -> np.ndarray:
    """Vector [N1, N2, N3, N4] en el punto (ξ, η)."""
    N = np.zeros(4)
    for i in range(4):
        xi_i, eta_i = NATURAL_COORDS[i]
        N[i] = 0.25 * (1.0 + xi * xi_i) * (1.0 + eta * eta_i)
    return N


def shape_function_derivatives(xi: float, eta: float) -> np.ndarray:
    """Devuelve dN/dξ y dN/dη en una matriz 2x4 (filas: ξ, η)."""
    dN = np.zeros((2, 4))
    for i in range(4):
        xi_i, eta_i = NATURAL_COORDS[i]
        dN[0, i] = 0.25 * xi_i * (1.0 + eta * eta_i)   # dNi/dξ
        dN[1, i] = 0.25 * eta_i * (1.0 + xi * xi_i)    # dNi/dη
    return dN


def constitutive_matrix(E: float, nu: float, plane_stress: bool = True) -> np.ndarray:
    """Matriz constitutiva D (3x3) para problema plano.

    Tensión plana:    D = E/(1-ν²) · [[1, ν, 0], [ν, 1, 0], [0, 0, (1-ν)/2]]
    Deformación pl.:  D = E/((1+ν)(1-2ν)) · [[1-ν, ν, 0], [ν, 1-ν, 0], [0, 0, (1-2ν)/2]]
    """
    if plane_stress:
        c = E / (1.0 - nu * nu)
        return c * np.array([
            [1.0, nu,  0.0],
            [nu,  1.0, 0.0],
            [0.0, 0.0, (1.0 - nu) / 2.0],
        ])
    else:
        c = E / ((1.0 + nu) * (1.0 - 2.0 * nu))
        return c * np.array([
            [1.0 - nu, nu,       0.0],
            [nu,       1.0 - nu, 0.0],
            [0.0,      0.0,      (1.0 - 2.0 * nu) / 2.0],
        ])


@dataclass
class Q4GaussData:
    """Información calculada en un punto de Gauss (para mostrar paso a paso)."""
    xi: float
    eta: float
    weight: float
    N: np.ndarray            # (4,)   funciones de forma
    dN_natural: np.ndarray   # (2,4)  dN/dξ, dN/dη
    J: np.ndarray            # (2,2)  Jacobiano
    detJ: float
    dN_xy: np.ndarray        # (2,4)  dN/dx, dN/dy
    B: np.ndarray            # (3,8)  strain-displacement


@dataclass
class Q4Element:
    """Elemento Q4: 4 nodos, 2 GDL por nodo (8 GDL total)."""

    id: int
    nodes: list[Node]        # 4 nodos en orden antihorario
    E: float                 # módulo de elasticidad (Pa)
    nu: float                # coeficiente de Poisson
    t: float                 # espesor (m)
    plane_stress: bool = True

    def __post_init__(self) -> None:
        if len(self.nodes) != 4:
            raise ValueError("Q4Element requiere exactamente 4 nodos.")

    # ---------- propiedades geométricas ----------
    @property
    def coords(self) -> np.ndarray:
        """Matriz 4x2 con las coordenadas (x,y) de los nodos."""
        return np.array([[n.x, n.y] for n in self.nodes])

    def global_dofs(self) -> list[int]:
        """8 GDL globales: [u1x, u1y, u2x, u2y, u3x, u3y, u4x, u4y]."""
        dofs = []
        for n in self.nodes:
            dofs.extend(n.dofs)
        return dofs

    # ---------- cálculos en un punto ----------
    def jacobian(self, xi: float, eta: float) -> tuple[np.ndarray, float]:
        dN = shape_function_derivatives(xi, eta)   # 2x4
        J = dN @ self.coords                        # (2x4) · (4x2) = 2x2
        detJ = float(np.linalg.det(J))
        return J, detJ

    def B_matrix(self, xi: float, eta: float) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        """Devuelve (B, J, detJ, dN_xy)."""
        dN_nat = shape_function_derivatives(xi, eta)        # 2x4
        J = dN_nat @ self.coords                            # 2x2
        detJ = float(np.linalg.det(J))
        if detJ <= 0:
            raise ValueError(
                f"Jacobiano no positivo (det={detJ}) en (ξ={xi}, η={eta}). "
                "Revisa el orden de los nodos (debe ser antihorario)."
            )
        invJ = np.linalg.inv(J)
        dN_xy = invJ @ dN_nat                               # 2x4: filas x,y
        B = np.zeros((3, 8))
        for i in range(4):
            dNi_dx = dN_xy[0, i]
            dNi_dy = dN_xy[1, i]
            B[0, 2 * i]     = dNi_dx
            B[1, 2 * i + 1] = dNi_dy
            B[2, 2 * i]     = dNi_dy
            B[2, 2 * i + 1] = dNi_dx
        return B, J, detJ, dN_xy

    def D_matrix(self) -> np.ndarray:
        return constitutive_matrix(self.E, self.nu, self.plane_stress)

    # ---------- integración numérica ----------
    def gauss_data(self) -> list[Q4GaussData]:
        """Calcula toda la info en los 4 puntos de Gauss (didáctico)."""
        data = []
        for xi, eta, w in GAUSS_2X2:
            B, J, detJ, dN_xy = self.B_matrix(xi, eta)
            data.append(Q4GaussData(
                xi=xi, eta=eta, weight=w,
                N=shape_functions(xi, eta),
                dN_natural=shape_function_derivatives(xi, eta),
                J=J, detJ=detJ, dN_xy=dN_xy, B=B,
            ))
        return data

    def stiffness_matrix(self) -> tuple[np.ndarray, list[Q4GaussData]]:
        """K = ∫∫ Bᵀ D B t dx dy = Σ Bᵀ D B t |J| w (2x2 Gauss).

        Devuelve la matriz K (8x8) y los datos de cada punto de Gauss.
        """
        D = self.D_matrix()
        K = np.zeros((8, 8))
        gp_data = self.gauss_data()
        for gp in gp_data:
            K += gp.B.T @ D @ gp.B * self.t * gp.detJ * gp.weight
        return K, gp_data

    # ---------- post-proceso ----------
    def strains_stresses_at(self, xi: float, eta: float,
                            displacements_8: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Devuelve (ε, σ) en (ξ, η) dado el vector de desplazamientos del elemento."""
        B, _, _, _ = self.B_matrix(xi, eta)
        strain = B @ displacements_8
        stress = self.D_matrix() @ strain
        return strain, stress

    def strains_stresses_at_corners(
        self, displacements_8: np.ndarray
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Devuelve [(ε, σ)] en los 4 nodos (esquinas), en el orden ++, -+, --, +-."""
        return [
            self.strains_stresses_at(xi_i, eta_i, displacements_8)
            for xi_i, eta_i in NATURAL_COORDS
        ]
