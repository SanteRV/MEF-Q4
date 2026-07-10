"""Elemento cuadrilátero bilineal Q4 (tensión plana / deformación plana).

Implementación fiel al desarrollo del documento guía de la tesis
("PARA LA MATRIZ B Y DE RIGIDEZ"), en su mismo orden:

    K^e = t · ∫∫ Bᵀ·D·B · det J · dξ · dη          (integración 2×2 de Gauss)

    1. Jacobiano J con componentes explícitos J11, J12, J21, J22   (ec. 31)
    2. Inversa del Jacobiano en forma explícita                    (ec. 32)
       [∂F/∂x; ∂F/∂y] = (1/det J)·[[J22, -J12], [-J21, J11]]·[∂F/∂ξ; ∂F/∂η]
    3. Cambio de variable  dx·dy = det J · dξ · dη                 (ec. 33)
    4. Matriz A (3×4), reordena las derivadas naturales            (ec. 37)
    5. Matriz G (4×8), derivadas de N respecto a ξ y η             (ec. 38)
    6. Matriz strain-displacement  B = A·G  (3×8)                  (ec. 39)

Convención de nodos (CCW desde la esquina superior derecha):
    2 --- 1
    |     |
    3 --- 4

Coordenadas naturales (ξ, η) ∈ [-1, 1]:
    N1 = (+,+),  N2 = (-,+),  N3 = (-,-),  N4 = (+,-)

Funciones de forma: Ni = 1/4 (1 + ξ·ξi) (1 + η·ηi)

NOTA sobre la numeración: la guía escribe sus fórmulas numerando los nodos
desde la esquina (-1,-1). Este programa (y su Excel de validación PLANE.xlsx)
numera desde (+1,+1). Ambas numeraciones son CCW y físicamente equivalentes;
las expresiones de J y G conservan aquí la ESTRUCTURA y el ORDEN de la guía,
con los signos que corresponden a la numeración del programa.
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
    """Vector [N1, N2, N3, N4] en el punto (ξ, η).

    Interpolación isoparamétrica (ec. 18 de la guía):
        u = N1·q1 + N2·q3 + N3·q5 + N4·q7
        v = N1·q2 + N2·q4 + N3·q6 + N4·q8
        x = N1·x1 + N2·x2 + N3·x3 + N4·x4
        y = N1·y1 + N2·y2 + N3·y3 + N4·y4
    """
    N = np.zeros(4)
    for i in range(4):
        xi_i, eta_i = NATURAL_COORDS[i]
        N[i] = 0.25 * (1.0 + xi * xi_i) * (1.0 + eta * eta_i)
    return N


def shape_function_derivatives(xi: float, eta: float) -> np.ndarray:
    """Devuelve dN/dξ y dN/dη en una matriz 2x4 (filas: ξ, η).

    Son las mismas derivadas que la guía agrupa en la matriz G (ec. 38).
    """
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
    J: np.ndarray            # (2,2)  Jacobiano (ec. 31)
    detJ: float
    A: np.ndarray            # (3,4)  matriz A (ec. 37)
    G: np.ndarray            # (4,8)  matriz G (ec. 38)
    dN_xy: np.ndarray        # (2,4)  dN/dx, dN/dy (via inversa explícita, ec. 32)
    B: np.ndarray            # (3,8)  strain-displacement B = A·G (ec. 39)


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

    # ---------- 1. Jacobiano (ec. 30-31 de la guía) ----------
    def jacobian(self, xi: float, eta: float) -> tuple[np.ndarray, float]:
        """Jacobiano J (2×2) con sus componentes explícitos.

            J = | J11  J12 |  =  | ∂x/∂ξ  ∂y/∂ξ |              (ec. 31)
                | J21  J22 |     | ∂x/∂η  ∂y/∂η |

        Componentes según la ec. 31 de la guía, con la numeración de nodos
        del programa (N1=++, N2=-+, N3=--, N4=+-):
        """
        (x1, y1), (x2, y2), (x3, y3), (x4, y4) = self.coords

        J11 = 0.25 * (+(1 + eta) * x1 - (1 + eta) * x2 - (1 - eta) * x3 + (1 - eta) * x4)
        J12 = 0.25 * (+(1 + eta) * y1 - (1 + eta) * y2 - (1 - eta) * y3 + (1 - eta) * y4)
        J21 = 0.25 * (+(1 + xi) * x1 + (1 - xi) * x2 - (1 - xi) * x3 - (1 + xi) * x4)
        J22 = 0.25 * (+(1 + xi) * y1 + (1 - xi) * y2 - (1 - xi) * y3 - (1 + xi) * y4)

        J = np.array([
            [J11, J12],
            [J21, J22],
        ])
        detJ = J11 * J22 - J12 * J21
        return J, detJ

    # ---------- 4. Matriz A (ec. 37 de la guía) ----------
    @staticmethod
    def A_matrix(J: np.ndarray, detJ: float) -> np.ndarray:
        """Matriz A (3×4): lleva las derivadas naturales de u y v a ε.

            ε = [∂u/∂x; ∂v/∂y; ∂u/∂y + ∂v/∂x] = A·[∂u/∂ξ; ∂u/∂η; ∂v/∂ξ; ∂v/∂η]

            A = (1/det J) · | J22  -J12   0     0   |             (ec. 37)
                            |  0     0  -J21   J11  |
                            | -J21  J11  J22  -J12  |
        """
        J11, J12 = J[0, 0], J[0, 1]
        J21, J22 = J[1, 0], J[1, 1]
        return (1.0 / detJ) * np.array([
            [J22, -J12,  0.0,  0.0],
            [0.0,  0.0, -J21,  J11],
            [-J21, J11,  J22, -J12],
        ])

    # ---------- 5. Matriz G (ec. 38 de la guía) ----------
    @staticmethod
    def G_matrix(xi: float, eta: float) -> np.ndarray:
        """Matriz G (4×8): derivadas naturales de u y v en términos de q.

            [∂u/∂ξ; ∂u/∂η; ∂v/∂ξ; ∂v/∂η] = G·q                   (ec. 38)

        con q = [q1..q8] = [u1, v1, u2, v2, u3, v3, u4, v4].
        Signos según la numeración de nodos del programa
        (N1=++, N2=-+, N3=--, N4=+-):
        """
        return 0.25 * np.array([
            [+(1 + eta), 0.0, -(1 + eta), 0.0, -(1 - eta), 0.0, +(1 - eta), 0.0],
            [+(1 + xi),  0.0, +(1 - xi),  0.0, -(1 - xi),  0.0, -(1 + xi),  0.0],
            [0.0, +(1 + eta), 0.0, -(1 + eta), 0.0, -(1 - eta), 0.0, +(1 - eta)],
            [0.0, +(1 + xi),  0.0, +(1 - xi),  0.0, -(1 - xi),  0.0, -(1 + xi)],
        ])

    # ---------- 6. Matriz B = A·G (ec. 39 de la guía) ----------
    def B_matrix(self, xi: float, eta: float) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        """Matriz strain-displacement B (3×8) siguiendo la guía: B = A·G.

        Devuelve (B, J, detJ, dN_xy). Las derivadas cartesianas dN_xy se
        obtienen con la inversa explícita del Jacobiano (ec. 32):

            [∂Ni/∂x; ∂Ni/∂y] = (1/det J)·[[J22, -J12], [-J21, J11]]·[∂Ni/∂ξ; ∂Ni/∂η]
        """
        J, detJ = self.jacobian(xi, eta)
        if detJ <= 0:
            raise ValueError(
                f"Jacobiano no positivo (det={detJ}) en (ξ={xi}, η={eta}). "
                "Revisa el orden de los nodos (debe ser antihorario)."
            )

        A = self.A_matrix(J, detJ)          # (3,4)  ec. 37
        G = self.G_matrix(xi, eta)          # (4,8)  ec. 38
        B = A @ G                           # (3,8)  ec. 39: ε = A·G·q = B·q

        # Derivadas cartesianas con la inversa explícita (ec. 32) — se usan
        # en la presentación didáctica paso a paso.
        J11, J12 = J[0, 0], J[0, 1]
        J21, J22 = J[1, 0], J[1, 1]
        invJ = (1.0 / detJ) * np.array([
            [J22, -J12],
            [-J21, J11],
        ])
        dN_xy = invJ @ shape_function_derivatives(xi, eta)   # (2,4)

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
                J=J, detJ=detJ,
                A=self.A_matrix(J, detJ),
                G=self.G_matrix(xi, eta),
                dN_xy=dN_xy, B=B,
            ))
        return data

    def stiffness_matrix(self) -> tuple[np.ndarray, list[Q4GaussData]]:
        """Matriz de rigidez del elemento según la guía:

            K^e = t · ∫∫ Bᵀ·D·B · det J · dξ · dη  ≈  Σᵢ Bᵢᵀ·D·Bᵢ·t·det Jᵢ·wᵢ

        (cuadratura de Gauss 2×2, wᵢ = 1; el cambio de variable
        dx·dy = det J·dξ·dη es la ec. 33 de la guía).

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
        """Devuelve (ε, σ) en (ξ, η): ε = B·q (guía) y σ = D·ε."""
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
