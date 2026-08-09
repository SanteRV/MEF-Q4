"""Elemento PLATE rectangular de 12 GDL (flexión de placas, Kirchhoff).

Implementa la sección "ELEMENTOS PLATE" del documento teórico de la tesis
(TEORÍA OK.docx, cap. 01.01.03), por el método de la FUNCIÓN DE
DESPLAZAMIENTOS, en su mismo orden y con sus mismas fórmulas:

    1. Polinomio de 12 términos para w(x, y)                      (ec. 1.3.11)
       w = α1 + α2·x + α3·y + α4·x² + α5·xy + α6·y² + α7·x³
           + α8·x²y + α9·xy² + α10·y³ + α11·x³y + α12·xy³
    2. w = [P]·[α]  con P = fila de los 12 monomios               (ec. 1.3.13)
    3. δ = [C]·[α]  evaluando (w, θx, θy) en los 4 nodos          (ec. 1.3.14)
       con θx = ∂w/∂y (ec. 1.3.3) y θy = -∂w/∂x (ec. 1.3.4)
    4. Deformaciones:  ε = -z·[Q]·[α]                             (ec. 1.3.16)
       (curvaturas: ∂²w/∂x², ∂²w/∂y², 2·∂²w/∂x∂y)
    5. Matriz B:       B = -z·[Q]·[C⁻¹]                           (ec. 1.3.17)
    6. Rigidez:  K^e = [C⁻¹]ᵀ·[∫∫ Qᵀ·D·Q dx dy]·[C⁻¹]·(t³/12)     (ec. 1.3.18)
       con D del estado plano de esfuerzos (ec. 1.3.9)
    7. Carga distribuida q → F nodal = q·A/4 en los GDL w         (ec. 1.3.19)

Hipótesis de Kirchhoff (placas delgadas): la deformación dominante es el
desplazamiento transversal w; las normales a la superficie media permanecen
rectas y perpendiculares (sin cortante transversal); εz = γyz = γzx = 0.
Rango de validez: 8..10 <= a/h <= 80..100 (Ventsel & Krauthammer).

GDL por nodo: (w, θx, θy) — 4 nodos → 12 GDL por elemento.
El ORDEN DE LOS NODOS es el mismo que el del elemento plane (el documento
lo exige): N1=(--), N2=(+-), N3=(++), N4=(-+), CCW desde la esquina
inferior izquierda. El elemento debe ser RECTANGULAR alineado con los ejes
(la formulación usa coordenadas físicas x, y directamente, no hay mapeo
isoparamétrico general).
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from .gauss import gauss_1d
from .node_plate import NodePlate
from .q4_element import constitutive_matrix


# Puntos de Gauss 1D de 3 puntos (Tabla 1 del documento, fila n=3).
# La integral de QᵀDQ tiene monomios de hasta grado 4 por dirección;
# 3 puntos integran exacto polinomios de grado 5 — suficiente y exacto.
GAUSS_ORDER = 3
GAUSS_1D_3 = gauss_1d(GAUSS_ORDER)


def P_row(x: float, y: float) -> np.ndarray:
    """Fila P de los 12 monomios del polinomio de w (ec. 1.3.12).

    Para qué sirve: es la 'base' con la que se construye todo — evaluada
    en los nodos arma la matriz C, y sus segundas derivadas arman Q.
    """
    return np.array([
        1.0, x, y, x * x, x * y, y * y,
        x ** 3, x * x * y, x * y * y, y ** 3,
        x ** 3 * y, x * y ** 3,
    ])


def dP_dx_row(x: float, y: float) -> np.ndarray:
    """Derivada ∂P/∂x — se usa (negada) para la fila de θy en C (ec. 1.3.4)."""
    return np.array([
        0.0, 1.0, 0.0, 2.0 * x, y, 0.0,
        3.0 * x * x, 2.0 * x * y, y * y, 0.0,
        3.0 * x * x * y, y ** 3,
    ])


def dP_dy_row(x: float, y: float) -> np.ndarray:
    """Derivada ∂P/∂y — es la fila de θx en C (ec. 1.3.3)."""
    return np.array([
        0.0, 0.0, 1.0, 0.0, x, 2.0 * y,
        0.0, x * x, 2.0 * x * y, 3.0 * y * y,
        x ** 3, 3.0 * x * y * y,
    ])


def Q_matrix(x: float, y: float) -> np.ndarray:
    """Matriz Q (3×12): curvaturas del polinomio w (ec. 1.3.16).

    Para qué sirve: las deformaciones de flexión son proporcionales a las
    curvaturas de la placa,  ε = -z·[∂²w/∂x²; ∂²w/∂y²; 2·∂²w/∂x∂y]
    (ec. 1.3.8). Q expresa esas curvaturas en función de los coeficientes α:

        Q = | 0 0 0 2 0 0 6x 2y  0  0  6xy   0   |
            | 0 0 0 0 0 2  0  0 2x 6y   0   6xy  |
            | 0 0 0 0 2 0  0 4x 4y  0  6x²  6y²  |
    """
    return np.array([
        [0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 6.0 * x, 2.0 * y, 0.0, 0.0, 6.0 * x * y, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 2.0 * x, 6.0 * y, 0.0, 6.0 * x * y],
        [0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 4.0 * x, 4.0 * y, 0.0, 6.0 * x * x, 6.0 * y * y],
    ])


@dataclass
class PlateElement:
    """Elemento plate rectangular: 4 nodos, 3 GDL por nodo (12 GDL).

    Para qué sirve: representa un panel de placa a flexión. Sabe calcular
    su matriz de rigidez K^e (12×12, ec. 1.3.18) y recuperar curvaturas,
    momentos y esfuerzos a partir de los desplazamientos nodales.
    """

    id: int
    nodes: list[NodePlate]     # 4 nodos en orden CCW desde inf-izq (como plane)
    E: float                   # módulo de elasticidad (Pa)
    nu: float                  # coeficiente de Poisson
    t: float                   # espesor de la placa (m)

    def __post_init__(self) -> None:
        if len(self.nodes) != 4:
            raise ValueError("PlateElement requiere exactamente 4 nodos.")
        # La formulación por función de desplazamientos (ec. 1.3.11-1.3.18)
        # está desarrollada para un RECTÁNGULO alineado con los ejes.
        xs = sorted({round(n.x, 12) for n in self.nodes})
        ys = sorted({round(n.y, 12) for n in self.nodes})
        if len(xs) != 2 or len(ys) != 2:
            raise ValueError(
                "PlateElement requiere un rectángulo alineado con los ejes "
                "(la formulación de 12 GDL del documento es para rectángulos)."
            )

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
        """Área del rectángulo (a·b) — se usa en el vector de fuerzas."""
        a, b = self.sides
        return a * b

    def global_dofs(self) -> list[int]:
        """Los 12 GDL globales: [w1, θx1, θy1, w2, θx2, θy2, ..., θy4]."""
        dofs: list[int] = []
        for n in self.nodes:
            dofs.extend(n.dofs)
        return dofs

    # ---------- 1-3. Matriz C (ec. 1.3.14) ----------
    def C_matrix(self) -> np.ndarray:
        """Matriz C (12×12): evalúa (w, θx, θy) del polinomio en los 4 nodos.

        Para qué sirve: conecta los coeficientes α del polinomio con los
        desplazamientos nodales δ (δ = C·α). Su inversa convierte
        desplazamientos nodales en coeficientes: α = C⁻¹·δ (ec. 1.3.15).

        Por cada nodo i aporta 3 filas:
            fila w  :  P(xi, yi)                (ec. 1.3.13)
            fila θx :  ∂P/∂y (xi, yi)           (θx = ∂w/∂y, ec. 1.3.3)
            fila θy : -∂P/∂x (xi, yi)           (θy = -∂w/∂x, ec. 1.3.4)
        """
        C = np.zeros((12, 12))
        for i, n in enumerate(self.nodes):
            C[3 * i, :] = P_row(n.x, n.y)
            C[3 * i + 1, :] = dP_dy_row(n.x, n.y)
            C[3 * i + 2, :] = -dP_dx_row(n.x, n.y)
        return C

    def D_matrix(self) -> np.ndarray:
        """Matriz constitutiva D (3×3) del estado plano de esfuerzos (ec. 1.3.9).

        La placa delgada de Kirchhoff trabaja en tensión plana: cada
        'lámina' a distancia z de la superficie media sigue σ = D·ε.
        """
        return constitutive_matrix(self.E, self.nu, plane_stress=True)

    # ---------- 5. Matriz B (ec. 1.3.17) ----------
    def B_matrix(self, x: float, y: float, z: float) -> np.ndarray:
        """Matriz strain-displacement B (3×12) en el punto (x, y, z).

        Para qué sirve: relaciona los 12 desplazamientos nodales con las
        deformaciones en cualquier punto del espesor:
            ε = B·δ   con   B = -z·[Q]·[C⁻¹]                     (ec. 1.3.17)
        La deformación crece linealmente con z (máxima en las caras
        z = ±t/2, nula en la superficie media).
        """
        C_inv = np.linalg.inv(self.C_matrix())
        return -z * Q_matrix(x, y) @ C_inv

    # ---------- 6. Matriz de rigidez (ec. 1.3.18) ----------
    def stiffness_matrix(self) -> np.ndarray:
        """Matriz de rigidez del elemento según la fórmula del documento:

            K^e = [C⁻¹]ᵀ · [∫∫ Qᵀ·D·Q dx dy] · [C⁻¹] · (t³/12)   (ec. 1.3.18)

        Para qué sirve: relaciona los 12 desplazamientos nodales (w, θx,
        θy por nodo) con las 12 fuerzas nodales (Fz, Mx, My por nodo).
        El factor t³/12 proviene de integrar z² en el espesor
        (∫ z² dz de -t/2 a t/2), que es la rigidez a flexión de la placa.

        La integral ∫∫ Qᵀ·D·Q dx dy se evalúa con cuadratura de Gauss
        3×3 (tabla 1 del documento): el integrando tiene monomios de
        hasta grado 4 por dirección y 3 puntos integran exacto grado 5,
        de modo que el resultado es EXACTO.
        """
        xs = [n.x for n in self.nodes]
        ys = [n.y for n in self.nodes]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
        a = x2 - x1     # lado en x
        b = y2 - y1     # lado en y
        xc = (x1 + x2) / 2.0
        yc = (y1 + y2) / 2.0

        D = self.D_matrix()

        # Integral en coordenadas fisicas via mapeo lineal al cuadrado
        # patron: x = xc + (a/2)ξ, y = yc + (b/2)η, dxdy = (a·b/4)dξdη
        K_hat = np.zeros((12, 12))
        jac = (a / 2.0) * (b / 2.0)
        for xi, w_xi in GAUSS_1D_3:
            for eta, w_eta in GAUSS_1D_3:
                x = xc + (a / 2.0) * xi
                y = yc + (b / 2.0) * eta
                Q = Q_matrix(x, y)
                K_hat += (Q.T @ D @ Q) * w_xi * w_eta * jac

        C_inv = np.linalg.inv(self.C_matrix())
        return C_inv.T @ K_hat @ C_inv * (self.t ** 3 / 12.0)

    # ---------- 7. Vector de fuerzas por carga distribuida (ec. 1.3.19) ----------
    def load_vector_uniform(self, q: float) -> np.ndarray:
        """Vector de fuerzas nodales para una carga q uniforme (N/m²).

        1er caso del documento (ec. 1.3.19): la carga total q·A se reparte
        en partes iguales a los 4 nodos, solo en los GDL de fuerza
        vertical w (sin momentos):

            F = [qA/4, 0, 0, qA/4, 0, 0, qA/4, 0, 0, qA/4, 0, 0]ᵀ
        """
        F = np.zeros(12)
        carga_nodal = q * self.area / 4.0
        for i in range(4):
            F[3 * i] = carga_nodal
        return F

    # ---------- 7b. Carga repartida a vigas en 1 dirección (ec. 1.3.20) ----------
    def load_vector_one_way(self, q: float, span: str = "y") -> np.ndarray:
        """Vector de fuerzas del 2do caso del documento (ec. 1.3.20).

        Cuando la placa se apoya sobre vigas en UNA sola dirección, la carga
        de superficie q no se reparte por igual a los 4 nodos: viaja como
        carga de línea hacia las vigas y llega a los nodos con el reparto de
        empotramiento perfecto de una viga (fuerza y momento).

        Con las vigas orientadas en Y (caso literal de la ec. 1.3.20):

            qv = q·a/2                         carga de línea sobre cada viga
            F = [qv·b/2,  qv·b²/12, 0,
                 qv·b/2,  qv·b²/12, 0,
                 qv·b/2, -qv·b²/12, 0,
                 qv·b/2, -qv·b²/12, 0]ᵀ

        Los momentos entran en el GDL θx (= ∂w/∂y), que es el giro de flexión
        de una viga que salva la luz b, y cambian de signo entre el extremo
        inicial (nodos 1 y 2, en y = y_min) y el final (nodos 3 y 4).

        span="x" es la versión simétrica (vigas orientadas en X): la luz pasa
        a ser a, la carga de línea qv = q·b/2 y los momentos entran en θy
        (= -∂w/∂x) con el signo invertido entre x_min y x_max.
        """
        a, b = self.sides
        F = np.zeros(12)
        if span == "y":
            qv = q * a / 2.0                  # carga de línea sobre la viga
            fuerza = qv * b / 2.0
            momento = qv * b * b / 12.0
            # Nodos 1 y 2 están en y_min; nodos 3 y 4 en y_max
            signos = (+1.0, +1.0, -1.0, -1.0)
            for i in range(4):
                F[3 * i] = fuerza
                F[3 * i + 1] = signos[i] * momento     # GDL θx
        elif span == "x":
            qv = q * b / 2.0
            fuerza = qv * a / 2.0
            momento = qv * a * a / 12.0
            # Nodos 1 y 4 están en x_min; nodos 2 y 3 en x_max.
            # θy = -∂w/∂x, de ahí el signo opuesto al caso de θx.
            signos = (-1.0, +1.0, +1.0, -1.0)
            for i in range(4):
                F[3 * i] = fuerza
                F[3 * i + 2] = signos[i] * momento     # GDL θy
        else:
            raise ValueError("span debe ser 'x' o 'y'.")
        return F

    # ---------- post-proceso ----------
    def curvatures_at(self, x: float, y: float,
                      displacements_12: np.ndarray) -> np.ndarray:
        """Curvaturas [∂²w/∂x², ∂²w/∂y², 2∂²w/∂x∂y] en un punto (ec. 1.3.16).

        Para qué sirve: son la 'deformación generalizada' de la placa;
        de ellas salen los momentos y los esfuerzos.
        """
        C_inv = np.linalg.inv(self.C_matrix())
        alpha = C_inv @ displacements_12          # α = C⁻¹·δ (ec. 1.3.15)
        return Q_matrix(x, y) @ alpha

    def moments_at(self, x: float, y: float,
                   displacements_12: np.ndarray) -> np.ndarray:
        """Momentos flectores por unidad de longitud [Mx, My, Mxy].

        M = ∫ σ·z dz = -(t³/12)·D·[curvaturas]  (integración de la ec.
        3.10 en el espesor). Mx y My son momentos de flexión; Mxy es el
        momento torsor.
        """
        kappa = self.curvatures_at(x, y, displacements_12)
        return -(self.t ** 3 / 12.0) * (self.D_matrix() @ kappa)

    def strains_stresses_at(self, x: float, y: float, z: float,
                            displacements_12: np.ndarray
                            ) -> tuple[np.ndarray, np.ndarray]:
        """Deformaciones y esfuerzos [σx, σy, τxy] en un punto (x, y, z).

        ε = B·δ con B = -z·Q·C⁻¹ (ec. 1.3.17) y σ = D·ε (ec. 1.3.10).
        Los valores máximos ocurren en las caras z = ±t/2.
        """
        B = self.B_matrix(x, y, z)
        strain = B @ displacements_12
        stress = self.D_matrix() @ strain
        return strain, stress

    def w_at(self, x: float, y: float, displacements_12: np.ndarray) -> float:
        """Desplazamiento transversal w interpolado en un punto (ec. 1.3.15)."""
        C_inv = np.linalg.inv(self.C_matrix())
        alpha = C_inv @ displacements_12
        return float(P_row(x, y) @ alpha)
