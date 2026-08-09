"""Elemento FLAT SHELL rectangular de 20 GDL (cap. 01.01.04).

Implementa la estrategia que describe el documento siguiendo a Cook et al.
(1989) y Bathe (1996): el shell se forma COMBINANDO un elemento de membrana
con uno de flexión, y las dos contribuciones quedan desacopladas en el
sistema local. No hay formulación nueva — se reutilizan íntegras las dos
que ya están validadas:

    membrana  ->  Q4 isoparamétrico de 8 GDL   (cap. 01.01.02)
    flexión   ->  plate rectangular de 12 GDL  (cap. 01.01.03)

    K[20x20] = | K_plane[8x8]      0        |                  (ec. 1.4.10)
               |      0       K_plate[12x12]|

    B[6x20]  = | B_plane[3x8]      0        |                  (ec. 1.4.11)
               |      0       B_plate[3x12] |

    D[6x6]   = | D_plane[3x3]      0        |         (cap. 01.01.07.03)
               |      0       D_plate[3x3]  |

ORDEN DE LOS GDL. El documento presenta las matrices por bloques (primero
todo lo de membrana, después todo lo de flexión) pero define el vector de
desplazamientos del elemento intercalado por nodo (01.01.07.01):

    q = (u1, v1, w1, θx1, θy1, ... , u4, v4, w4, θx4, θy4)

Las dos cosas describen la MISMA matriz en bases distintas. Aquí se ensambla
en el orden nodal del documento, que es el que necesita el ensamblaje
global, y se conserva la forma por bloques en `stiffness_blocks()` para
poder mostrarla tal como aparece en la ec. 1.4.10.

Restricción geométrica: el elemento debe ser un RECTÁNGULO alineado con los
ejes, porque la parte de flexión (plate de 12 GDL) solo está formulada para
rectángulos y solo supera la prueba del parche en mallas rectangulares
(01.01.08.04 y 01.01.08.05).

Alcance: facetas contenidas en el plano XY. El documento no desarrolla la
matriz de transformación del shell entre el sistema local de la faceta y el
global (sí lo hace para el elemento frame, cap. 01.02.01.03), de modo que
el modelo se mantiene, como en las figuras 12 y 15, en un solo plano.
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np

from .node import Node
from .node_plate import NodePlate
from .node_shell import NodeShell
from .plate_element import PlateElement
from .q4_element import Q4Element, constitutive_matrix


# Posición de cada GDL de membrana y de flexión dentro del vector nodal de
# 20 componentes (u, v, w, θx, θy por nodo).
#   membrana: GDL local j del Q4  -> nodo j//2, componente j%2   -> 5*nodo + comp
#   flexión:  GDL local j del plate -> nodo j//3, componente j%3 -> 5*nodo + 2 + comp
MAP_MEMBRANA = [5 * (j // 2) + (j % 2) for j in range(8)]
MAP_FLEXION = [5 * (j // 3) + 2 + (j % 3) for j in range(12)]


@dataclass
class ShellElement:
    """Elemento flat shell: 4 nodos, 5 GDL por nodo (20 GDL en total)."""

    id: int
    nodes: list[NodeShell]     # 4 nodos CCW desde inf-izq, como plane y plate
    E: float                   # módulo de elasticidad (Pa)
    nu: float                  # coeficiente de Poisson
    t: float                   # espesor (m)

    def __post_init__(self) -> None:
        if len(self.nodes) != 4:
            raise ValueError("ShellElement requiere exactamente 4 nodos.")
        xs = sorted({round(n.x, 12) for n in self.nodes})
        ys = sorted({round(n.y, 12) for n in self.nodes})
        if len(xs) != 2 or len(ys) != 2:
            raise ValueError(
                "ShellElement requiere un rectángulo alineado con los ejes "
                "(lo impone la parte de flexión, plate de 12 GDL)."
            )

    # ---------- sub-elementos que aportan cada comportamiento ----------
    @property
    def membrane(self) -> Q4Element:
        """Elemento plane equivalente — aporta la rigidez de membrana."""
        nodos = [Node(i, n.x, n.y) for i, n in enumerate(self.nodes)]
        return Q4Element(self.id, nodos, E=self.E, nu=self.nu, t=self.t,
                         plane_stress=True)

    @property
    def bending(self) -> PlateElement:
        """Elemento plate equivalente — aporta la rigidez de flexión."""
        nodos = [NodePlate(i, n.x, n.y) for i, n in enumerate(self.nodes)]
        return PlateElement(self.id, nodos, E=self.E, nu=self.nu, t=self.t)

    # ---------- propiedades geométricas ----------
    @property
    def coords(self) -> np.ndarray:
        """Matriz 4×2 con las coordenadas físicas (x, y) de los 4 nodos."""
        return np.array([[n.x, n.y] for n in self.nodes])

    @property
    def sides(self) -> tuple[float, float]:
        """Lados (a, b) del rectángulo: a en X, b en Y."""
        xs = [n.x for n in self.nodes]
        ys = [n.y for n in self.nodes]
        return (max(xs) - min(xs), max(ys) - min(ys))

    @property
    def area(self) -> float:
        a, b = self.sides
        return a * b

    def global_dofs(self) -> list[int]:
        """Los 20 GDL globales en el orden nodal del documento."""
        dofs: list[int] = []
        for n in self.nodes:
            dofs.extend(n.dofs)
        return dofs

    # ---------- matriz constitutiva por bloques ----------
    def D_matrix(self) -> np.ndarray:
        """D (6×6) por bloques: membrana arriba, flexión abajo.

        Ambos bloques son la misma matriz de estado plano de esfuerzos
        (ec. 1.2.6 y 1.3.9); se repiten porque actúan sobre deformaciones
        distintas: las de membrana y las de flexión.
        """
        D3 = constitutive_matrix(self.E, self.nu, plane_stress=True)
        D = np.zeros((6, 6))
        D[:3, :3] = D3
        D[3:, 3:] = D3
        return D

    # ---------- matriz de rigidez (ec. 1.4.10) ----------
    def stiffness_blocks(self) -> tuple[np.ndarray, np.ndarray]:
        """Devuelve (K_plane 8×8, K_plate 12×12) tal como los usa la ec. 1.4.10."""
        K_m, _ = self.membrane.stiffness_matrix()
        K_b = self.bending.stiffness_matrix()
        return K_m, K_b

    def stiffness_matrix(self) -> np.ndarray:
        """K^e (20×20) en el orden nodal (u, v, w, θx, θy) por nodo.

        Los dos bloques de la ec. 1.4.10 se colocan en las posiciones que
        les corresponden dentro del vector nodal; las casillas que cruzan
        membrana con flexión quedan en cero, que es exactamente lo que
        significa que ambos comportamientos estén desacoplados.
        """
        K_m, K_b = self.stiffness_blocks()
        K = np.zeros((20, 20))
        for i, gi in enumerate(MAP_MEMBRANA):
            for j, gj in enumerate(MAP_MEMBRANA):
                K[gi, gj] += K_m[i, j]
        for i, gi in enumerate(MAP_FLEXION):
            for j, gj in enumerate(MAP_FLEXION):
                K[gi, gj] += K_b[i, j]
        return K

    # ---------- matriz B (ec. 1.4.11) ----------
    def natural_coords_of(self, x: float, y: float) -> tuple[float, float]:
        """Convierte un punto físico (x, y) del rectángulo a (ξ, η)."""
        xs = [n.x for n in self.nodes]
        ys = [n.y for n in self.nodes]
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        xi = 2.0 * (x - (x0 + x1) / 2.0) / (x1 - x0)
        eta = 2.0 * (y - (y0 + y1) / 2.0) / (y1 - y0)
        return xi, eta

    def B_matrix(self, x: float, y: float, z: float) -> np.ndarray:
        """B (6×20) en el punto (x, y, z), en el orden nodal del documento.

        Filas 0-2: deformaciones de membrana (ec. 1.4.6).
        Filas 3-5: deformaciones de flexión a la cota z (ec. 1.4.7).
        """
        xi, eta = self.natural_coords_of(x, y)
        B_m, _, _, _ = self.membrane.B_matrix(xi, eta)     # (3, 8)
        B_b = self.bending.B_matrix(x, y, z)               # (3, 12)
        B = np.zeros((6, 20))
        for j, gj in enumerate(MAP_MEMBRANA):
            B[0:3, gj] = B_m[:, j]
        for j, gj in enumerate(MAP_FLEXION):
            B[3:6, gj] = B_b[:, j]
        return B

    def B_blocks(self, x: float, y: float, z: float) -> tuple[np.ndarray, np.ndarray]:
        """(B_plane 3×8, B_plate 3×12) por separado, como en la ec. 1.4.11."""
        xi, eta = self.natural_coords_of(x, y)
        B_m, _, _, _ = self.membrane.B_matrix(xi, eta)
        return B_m, self.bending.B_matrix(x, y, z)

    # ---------- vectores de fuerza ----------
    def load_vector_uniform(self, q: float) -> np.ndarray:
        """Carga transversal uniforme repartida a los nodos (ec. 1.3.19)."""
        F = np.zeros(20)
        fe = self.bending.load_vector_uniform(q)
        for j, gj in enumerate(MAP_FLEXION):
            F[gj] += fe[j]
        return F

    def load_vector_one_way(self, q: float, span: str = "y") -> np.ndarray:
        """Carga transversal repartida a vigas en una dirección (ec. 1.3.20)."""
        F = np.zeros(20)
        fe = self.bending.load_vector_one_way(q, span=span)
        for j, gj in enumerate(MAP_FLEXION):
            F[gj] += fe[j]
        return F

    # ---------- post-proceso (cap. 01.01.07) ----------
    def split_displacements(self, d20: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Separa el vector nodal de 20 GDL en (membrana 8, flexión 12)."""
        d20 = np.asarray(d20, dtype=float)
        return d20[MAP_MEMBRANA], d20[MAP_FLEXION]

    def strains_stresses_at(self, x: float, y: float, z: float,
                            displacements_20: np.ndarray
                            ) -> tuple[np.ndarray, np.ndarray]:
        """ε (6,) y σ (6,) en el punto (x, y, z): ε = B·q y σ = D·B·q.

        Las 3 primeras componentes son de membrana y las 3 siguientes de
        flexión a la cota z. La tensión total de una fibra es la suma de
        ambas contribuciones (ver `total_stress_at`).
        """
        B = self.B_matrix(x, y, z)
        strain = B @ np.asarray(displacements_20, dtype=float)
        stress = self.D_matrix() @ strain
        return strain, stress

    def total_stress_at(self, x: float, y: float, z: float,
                        displacements_20: np.ndarray) -> np.ndarray:
        """σ total [σx, σy, τxy] de la fibra a la cota z: membrana + flexión."""
        _, sig = self.strains_stresses_at(x, y, z, displacements_20)
        return sig[:3] + sig[3:]

    def moments_at(self, x: float, y: float,
                   displacements_20: np.ndarray) -> np.ndarray:
        """Momentos [Mx, My, Mxy] por unidad de longitud (parte de flexión)."""
        _, d_b = self.split_displacements(displacements_20)
        return self.bending.moments_at(x, y, d_b)

    def w_at(self, x: float, y: float, displacements_20: np.ndarray) -> float:
        """Desplazamiento transversal w interpolado en un punto."""
        _, d_b = self.split_displacements(displacements_20)
        return self.bending.w_at(x, y, d_b)
