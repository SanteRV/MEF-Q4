"""Elemento cuadrilátero bilineal Q4 (tensión plana / deformación plana).

Este módulo es el NÚCLEO del método: implementa la sección "ELEMENTOS
PLANE" del documento teórico de la tesis (TEORÍA OK.docx,
cap. 01.01.02), en su mismo orden y con sus mismas fórmulas (ec. 1.2.x):

    K^e = t · ∫∫ Bᵀ·D·B · det J · dξ · dη          (integración 2×2 de Gauss)

    1. Jacobiano J con componentes explícitos J11, J12, J21, J22   (ec. 1.2.17)
    2. Inversa del Jacobiano en forma explícita                    (ec. 1.2.18)
       [∂F/∂x; ∂F/∂y] = (1/det J)·[[J22, -J12], [-J21, J11]]·[∂F/∂ξ; ∂F/∂η]
    3. Cambio de variable  dx·dy = det J · dξ · dη                 (ec. 1.2.19)
    4. Matriz A (3×4), reordena las derivadas naturales            (ec. 1.2.23)
    5. Matriz G (4×8), derivadas de N respecto a ξ y η             (ec. 1.2.24)
    6. Matriz strain-displacement  B = A·G  (3×8)                  (ec. 1.2.25)

Convención de nodos (la del documento, figura 4: CCW desde la esquina
inferior izquierda):

        η
    4 --- 3          N1 = (-1, -1)   inferior izquierda
    |     | → ξ      N2 = (+1, -1)   inferior derecha
    1 --- 2          N3 = (+1, +1)   superior derecha
                     N4 = (-1, +1)   superior izquierda

Funciones de forma bilineales (ec. 1.2.7 a 1.2.10):  Ni = 1/4 (1 + ξ·ξi)(1 + η·ηi)
    N1 = 1/4 (1-ξ)(1-η)      N2 = 1/4 (1+ξ)(1-η)
    N3 = 1/4 (1+ξ)(1+η)      N4 = 1/4 (1-ξ)(1+η)

Cada Ni vale 1 en su propio nodo y 0 en los otros tres (partición de la
unidad). Con ellas se interpola tanto la geometría como los desplazamientos
(planteamiento isoparamétrico, ec. 1.2.11 del documento).
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .gauss import gauss_2d
from .node import Node


# ---------- Coordenadas naturales de los nodos (figura 4 del documento) ----------
# Fila i = (ξi, ηi) del nodo i+1. Este arreglo define la convención de TODO
# el programa: el orden de las funciones de forma, de las columnas de B y
# del ensamblaje de K^e depende de esta numeración.
NATURAL_COORDS = np.array([
    [-1.0, -1.0],   # nodo 1  (--)  inferior izquierda
    [+1.0, -1.0],   # nodo 2  (+-)  inferior derecha
    [+1.0, +1.0],   # nodo 3  (++)  superior derecha
    [-1.0, +1.0],   # nodo 4  (-+)  superior izquierda
])

# ---------- Puntos de Gauss 2×2 (cap. 01.01.05, tabla 1 fila n=2) ----------
# Regla de 2 puntos por dirección tomada de la tabla de Gauss del documento:
# los puntos caen en (±1/√3, ±1/√3) y todos los pesos valen 1 (ec. 1.5.2).
# Integra exacto polinomios de grado ≤ 3, suficiente para el integrando
# Bᵀ·D·B·det J del Q4.
# El orden de los GP sigue el de los nodos (--, +-, ++, -+): así el GP i
# queda en el cuadrante del nodo i, lo que facilita la lectura didáctica.
GAUSS_2X2 = gauss_2d(2)

# Número de puntos por dirección que usa este elemento (fila de la tabla 1).
GAUSS_ORDER = 2


def shape_functions(xi: float, eta: float) -> np.ndarray:
    """Evalúa las 4 funciones de forma en el punto natural (ξ, η).

    Para qué sirve: son los "pesos" con que cada nodo aporta al valor
    interpolado en el interior del elemento. Se usan para interpolar la
    geometría (x = Σ Ni·xi, y = Σ Ni·yi) y los desplazamientos
    (u = Σ Ni·ui, v = Σ Ni·vi) — ec. 1.2.11 del documento (isoparametría).

    Devuelve el vector [N1, N2, N3, N4], donde Ni = 1/4(1 + ξ·ξi)(1 + η·ηi).
    """
    N = np.zeros(4)
    for i in range(4):
        xi_i, eta_i = NATURAL_COORDS[i]
        N[i] = 0.25 * (1.0 + xi * xi_i) * (1.0 + eta * eta_i)
    return N


def shape_function_derivatives(xi: float, eta: float) -> np.ndarray:
    """Derivadas de las funciones de forma respecto a ξ y η.

    Para qué sirve: estas derivadas alimentan dos cosas —
      1. el Jacobiano J (ec. 1.2.17), que mide cómo se deforma el mapeo
         natural → físico, y
      2. la matriz G (ec. 1.2.24), que expresa las derivadas naturales de
         u y v en función de los desplazamientos nodales.

    Devuelve una matriz 2×4: fila 0 = ∂Ni/∂ξ, fila 1 = ∂Ni/∂η.
        ∂Ni/∂ξ = 1/4 · ξi · (1 + η·ηi)
        ∂Ni/∂η = 1/4 · ηi · (1 + ξ·ξi)
    """
    dN = np.zeros((2, 4))
    for i in range(4):
        xi_i, eta_i = NATURAL_COORDS[i]
        dN[0, i] = 0.25 * xi_i * (1.0 + eta * eta_i)   # dNi/dξ
        dN[1, i] = 0.25 * eta_i * (1.0 + xi * xi_i)    # dNi/dη
    return dN


def constitutive_matrix(E: float, nu: float, plane_stress: bool = True) -> np.ndarray:
    """Matriz constitutiva D (3×3) del material para problema plano.

    Para qué sirve: relaciona deformaciones con esfuerzos según la ley de
    Hooke, σ = D·ε. Es el "material" dentro de K^e = t·∫∫ Bᵀ·D·B·det J·dξdη.

    Tensión plana (σz = 0, placas delgadas):
        D = E/(1-ν²) · [[1, ν, 0], [ν, 1, 0], [0, 0, (1-ν)/2]]
    Deformación plana (εz = 0, sólidos largos):
        D = E/((1+ν)(1-2ν)) · [[1-ν, ν, 0], [ν, 1-ν, 0], [0, 0, (1-2ν)/2]]
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
    """Resultados intermedios calculados en UN punto de Gauss.

    Para qué sirve: el aplicativo es didáctico, por eso no basta con K^e —
    se guarda cada pieza del cálculo (J, A, G, B...) para poder mostrarla
    en el paso a paso, exportarla a Excel y al PDF del reporte.
    """
    xi: float                # coordenada ξ del punto de Gauss
    eta: float               # coordenada η del punto de Gauss
    weight: float            # peso w de la cuadratura (= 1 en la regla 2×2)
    N: np.ndarray            # (4,)   funciones de forma evaluadas en el GP
    dN_natural: np.ndarray   # (2,4)  ∂N/∂ξ y ∂N/∂η en el GP
    J: np.ndarray            # (2,2)  Jacobiano (ec. 1.2.17)
    detJ: float              # determinante de J (cambio de área, ec. 1.2.19)
    A: np.ndarray            # (3,4)  matriz A (ec. 1.2.23)
    G: np.ndarray            # (4,8)  matriz G (ec. 1.2.24)
    dN_xy: np.ndarray        # (2,4)  ∂N/∂x y ∂N/∂y (vía inversa explícita, ec. 1.2.18)
    B: np.ndarray            # (3,8)  strain-displacement B = A·G (ec. 1.2.25)


@dataclass
class Q4Element:
    """Elemento finito Q4: 4 nodos, 2 GDL por nodo (8 GDL en total).

    Para qué sirve: representa UNA pieza de la malla. Sabe calcular su
    propia matriz de rigidez K^e (8×8) y sus esfuerzos/deformaciones a
    partir de los desplazamientos que el solver le devuelve.
    """

    id: int
    nodes: list[Node]        # 4 nodos en orden antihorario (CCW)
    E: float                 # módulo de elasticidad (Pa)
    nu: float                # coeficiente de Poisson
    t: float                 # espesor (m)
    plane_stress: bool = True   # True = tensión plana, False = deformación plana

    def __post_init__(self) -> None:
        if len(self.nodes) != 4:
            raise ValueError("Q4Element requiere exactamente 4 nodos.")

    # ---------- propiedades geométricas ----------
    @property
    def coords(self) -> np.ndarray:
        """Matriz 4×2 con las coordenadas físicas (x, y) de los 4 nodos.

        La fila i corresponde al nodo i+1; el nodo i+1 ocupa la esquina
        natural NATURAL_COORDS[i].
        """
        return np.array([[n.x, n.y] for n in self.nodes])

    def global_dofs(self) -> list[int]:
        """Índices de los 8 GDL del elemento dentro del sistema global.

        Para qué sirve: el ensamblaje. La entrada K^e[a, b] se suma en
        K_global[dofs[a], dofs[b]]. Orden: [u1x, u1y, u2x, u2y, ..., u4y].
        """
        dofs = []
        for n in self.nodes:
            dofs.extend(n.dofs)
        return dofs

    # ---------- 1. Jacobiano (ec. 1.2.16-1.2.17 del documento) ----------
    def jacobian(self, xi: float, eta: float) -> tuple[np.ndarray, float]:
        """Jacobiano J (2×2) del mapeo natural → físico, con componentes explícitos.

        Para qué sirve: J conecta las derivadas en (ξ, η) con las derivadas
        en (x, y) (regla de la cadena, ec. 1.2.16) y su determinante realiza el
        cambio de variable de la integral: dx·dy = det J·dξ·dη (ec. 1.2.19).

            J = | J11  J12 |  =  | ∂x/∂ξ  ∂y/∂ξ |              (ec. 1.2.17)
                | J21  J22 |     | ∂x/∂η  ∂y/∂η |

        Componentes según la ec. 1.2.17 del documento (transcripción literal):
            J11 = 1/4·[-(1-η)·x1 + (1-η)·x2 + (1+η)·x3 - (1+η)·x4]
            J12 = 1/4·[-(1-η)·y1 + (1-η)·y2 + (1+η)·y3 - (1+η)·y4]
            J21 = 1/4·[-(1-ξ)·x1 - (1+ξ)·x2 + (1+ξ)·x3 + (1-ξ)·x4]
            J22 = 1/4·[-(1-ξ)·y1 - (1+ξ)·y2 + (1+ξ)·y3 + (1-ξ)·y4]
        """
        (x1, y1), (x2, y2), (x3, y3), (x4, y4) = self.coords

        J11 = 0.25 * (-(1 - eta) * x1 + (1 - eta) * x2 + (1 + eta) * x3 - (1 + eta) * x4)
        J12 = 0.25 * (-(1 - eta) * y1 + (1 - eta) * y2 + (1 + eta) * y3 - (1 + eta) * y4)
        J21 = 0.25 * (-(1 - xi) * x1 - (1 + xi) * x2 + (1 + xi) * x3 + (1 - xi) * x4)
        J22 = 0.25 * (-(1 - xi) * y1 - (1 + xi) * y2 + (1 + xi) * y3 + (1 - xi) * y4)

        J = np.array([
            [J11, J12],
            [J21, J22],
        ])
        # Determinante 2×2 en forma explícita (sin np.linalg.det, como la guía)
        detJ = J11 * J22 - J12 * J21
        return J, detJ

    # ---------- 4. Matriz A (ec. 1.2.23 del documento) ----------
    @staticmethod
    def A_matrix(J: np.ndarray, detJ: float) -> np.ndarray:
        """Matriz A (3×4): convierte derivadas naturales de u y v en deformaciones.

        Para qué sirve: las deformaciones físicas ε = [εx, εy, γxy] requieren
        derivadas respecto a (x, y), pero lo que sabemos derivar fácil es
        respecto a (ξ, η). A hace esa conversión usando la inversa explícita
        del Jacobiano (ec. 1.2.18) aplicada a u y v (ec. 1.2.20 y 1.2.21):

            ε = [∂u/∂x; ∂v/∂y; ∂u/∂y + ∂v/∂x] = A·[∂u/∂ξ; ∂u/∂η; ∂v/∂ξ; ∂v/∂η]

            A = (1/det J) · | J22  -J12   0     0   |             (ec. 1.2.23)
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

    # ---------- 5. Matriz G (ec. 1.2.24 del documento) ----------
    @staticmethod
    def G_matrix(xi: float, eta: float) -> np.ndarray:
        """Matriz G (4×8): deriva el campo de desplazamientos respecto a ξ y η.

        Para qué sirve: como u = Σ Ni·ui y v = Σ Ni·vi (isoparametría),
        sus derivadas naturales son combinaciones de las ∂Ni/∂ξ y ∂Ni/∂η.
        G organiza esos coeficientes para escribirlo en forma matricial:

            [∂u/∂ξ; ∂u/∂η; ∂v/∂ξ; ∂v/∂η] = G·q                   (ec. 1.2.24)

        con q = [q1..q8] = [u1, v1, u2, v2, u3, v3, u4, v4].

        Transcripción literal de la ec. 1.2.24 del documento:

            G = 1/4 · | -(1-η)   0    (1-η)   0    (1+η)   0   -(1+η)   0   |
                      | -(1-ξ)   0   -(1+ξ)   0    (1+ξ)   0    (1-ξ)   0   |
                      |   0   -(1-η)   0    (1-η)   0    (1+η)   0   -(1+η) |
                      |   0   -(1-ξ)   0   -(1+ξ)   0    (1+ξ)   0    (1-ξ) |
        """
        return 0.25 * np.array([
            [-(1 - eta), 0.0, +(1 - eta), 0.0, +(1 + eta), 0.0, -(1 + eta), 0.0],
            [-(1 - xi),  0.0, -(1 + xi),  0.0, +(1 + xi),  0.0, +(1 - xi),  0.0],
            [0.0, -(1 - eta), 0.0, +(1 - eta), 0.0, +(1 + eta), 0.0, -(1 + eta)],
            [0.0, -(1 - xi),  0.0, -(1 + xi),  0.0, +(1 + xi),  0.0, +(1 - xi)],
        ])

    # ---------- 6. Matriz B = A·G (ec. 1.2.25 del documento) ----------
    def B_matrix(self, xi: float, eta: float) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        """Matriz strain-displacement B (3×8) siguiendo la guía: B = A·G.

        Para qué sirve: B es el puente entre desplazamientos nodales y
        deformaciones (ε = B·q). Es la pieza central de la rigidez
        K^e = t·∫∫ Bᵀ·D·B·det J·dξ·dη y del post-proceso de esfuerzos.

        Devuelve (B, J, detJ, dN_xy). Las derivadas cartesianas dN_xy se
        obtienen con la inversa explícita del Jacobiano (ec. 1.2.18):

            [∂Ni/∂x; ∂Ni/∂y] = (1/det J)·[[J22, -J12], [-J21, J11]]·[∂Ni/∂ξ; ∂Ni/∂η]
        """
        J, detJ = self.jacobian(xi, eta)
        # det J > 0 garantiza que el mapeo natural → físico no está invertido
        # ni degenerado (área positiva). Si sale ≤ 0, los nodos del elemento
        # fueron ingresados en orden horario o cruzado.
        if detJ <= 0:
            raise ValueError(
                f"Jacobiano no positivo (det={detJ}) en (ξ={xi}, η={eta}). "
                "Revisa el orden de los nodos (debe ser antihorario)."
            )

        A = self.A_matrix(J, detJ)          # (3,4)  ec. 1.2.23
        G = self.G_matrix(xi, eta)          # (4,8)  ec. 1.2.24
        B = A @ G                           # (3,8)  ec. 1.2.25: ε = A·G·q = B·q

        # Derivadas cartesianas con la inversa explícita (ec. 1.2.18) — se usan
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
        """Matriz constitutiva D del elemento (según su material e hipótesis)."""
        return constitutive_matrix(self.E, self.nu, self.plane_stress)

    # ---------- integración numérica ----------
    def gauss_data(self) -> list[Q4GaussData]:
        """Evalúa TODAS las matrices intermedias en los 4 puntos de Gauss.

        Para qué sirve: alimenta el paso a paso didáctico (paso 6 del
        aplicativo) y los exports. El solver solo necesita B y detJ, pero
        aquí se conserva cada pieza (N, dN, J, A, G...) para poder mostrarla.
        """
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
        """Matriz de rigidez del elemento según la fórmula del documento:

            K^e = t · ∫∫ Bᵀ·D·B · det J · dξ · dη  ≈  Σᵢ Bᵢᵀ·D·Bᵢ·t·det Jᵢ·wᵢ

        Para qué sirve: K^e relaciona los 8 desplazamientos nodales del
        elemento con las 8 fuerzas nodales (equilibrio elástico). La
        integral se evalúa con cuadratura de Gauss 2×2 (wᵢ = 1); el cambio
        de variable dx·dy = det J·dξ·dη es la ec. 1.2.19 del documento.

        Devuelve la matriz K (8×8, simétrica) y los datos de cada punto
        de Gauss para el paso a paso.
        """
        D = self.D_matrix()
        K = np.zeros((8, 8))
        gp_data = self.gauss_data()
        for gp in gp_data:
            # Contribución de este punto de Gauss a la rigidez del elemento
            K += gp.B.T @ D @ gp.B * self.t * gp.detJ * gp.weight
        return K, gp_data

    # ---------- post-proceso ----------
    def strains_stresses_at(self, xi: float, eta: float,
                            displacements_8: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Deformaciones y esfuerzos en un punto natural (ξ, η) del elemento.

        Para qué sirve: post-proceso. Una vez que el solver entrega los
        desplazamientos q del elemento, aquí se recuperan
            ε = B·q   (deformaciones, guía: ε = A·G·q)
            σ = D·ε   (esfuerzos, ley de Hooke)
        """
        B, _, _, _ = self.B_matrix(xi, eta)
        strain = B @ displacements_8
        stress = self.D_matrix() @ strain
        return strain, stress

    def strains_stresses_at_corners(
        self, displacements_8: np.ndarray
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Deformaciones y esfuerzos evaluados en los 4 nodos (esquinas).

        Para qué sirve: los esfuerzos se calculan de forma natural en los
        puntos de Gauss (interiores), pero al usuario le interesan también
        en los nodos. Aquí se evalúa B directamente en cada esquina
        (ξ, η) = (±1, ±1), en el orden de los nodos: --, +-, ++, -+.
        """
        return [
            self.strains_stresses_at(xi_i, eta_i, displacements_8)
            for xi_i, eta_i in NATURAL_COORDS
        ]
