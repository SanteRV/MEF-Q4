"""Cuadratura de Gauss-Legendre — cap. 01.01.05 del documento teórico.

Fuente única de los puntos y pesos que usa TODO el programa. Antes cada
elemento llevaba sus constantes escritas a mano (2x2 en el Q4, 3 puntos en
el plate); al centralizarlas aquí, la Tabla 1 del documento y el cálculo
no pueden desincronizarse.

Contenido:

    Tabla 1 (n = 1..6)          puntos xi_i y pesos W_i en [-1, 1]
    Integración unidimensional  I = Sum_i W_i * Phi(xi_i)              (ec. 1.5.1)
    Integración bidimensional   I = Sum_i Sum_j W_i*W_j*Phi(xi_i, eta_j)  (ec. 1.5.2)

Una regla de n puntos integra de forma exacta polinomios de grado <= 2n-1.
De ahi la eleccion de cada elemento:

    Q4 plane  -> 2x2  (el integrando B^T*D*B*detJ es de grado <= 3)
    Plate 12 GDL -> 3x3 (el integrando Q^T*D*Q llega a grado 4 por direccion)
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Tabla 1 del documento: puntos y pesos de Gauss para n = 1..6
# ---------------------------------------------------------------------------
# Los valores se dan con los mismos decimales que la tabla; para n <= 3 se
# escriben en forma cerrada (0, +-1/raiz(3), +-raiz(0.6)) porque el documento
# los presenta asi y ademas evita error de redondeo.
GAUSS_TABLE: dict[int, list[tuple[float, float]]] = {
    1: [
        (0.0, 2.0),
    ],
    2: [
        (-1.0 / np.sqrt(3.0), 1.0),
        (+1.0 / np.sqrt(3.0), 1.0),
    ],
    3: [
        (-np.sqrt(0.6), 5.0 / 9.0),
        (0.0, 8.0 / 9.0),
        (+np.sqrt(0.6), 5.0 / 9.0),
    ],
    4: [
        (-0.8611363116, 0.3478548451),
        (-0.3399810436, 0.6521451549),
        (+0.3399810436, 0.6521451549),
        (+0.8611363116, 0.3478548451),
    ],
    5: [
        (-0.9061798459, 0.2369268851),
        (-0.5384693101, 0.4786286705),
        (0.0, 0.5688888889),
        (+0.5384693101, 0.4786286705),
        (+0.9061798459, 0.2369268851),
    ],
    6: [
        (-0.9324695142, 0.1713244924),
        (-0.6612093865, 0.3607615730),
        (-0.2386191861, 0.4679139346),
        (+0.2386191861, 0.4679139346),
        (+0.6612093865, 0.3607615730),
        (+0.9324695142, 0.1713244924),
    ],
}

# Orden maximo disponible en la Tabla 1 del documento.
MAX_ORDER = max(GAUSS_TABLE)


def gauss_1d(n: int) -> list[tuple[float, float]]:
    """Devuelve [(xi_i, W_i), ...] de la regla de n puntos (Tabla 1).

    Para que sirve: es la base de la ec. 1.5.1. Cualquier integral llevada
    al intervalo [-1, 1] se aproxima con estos pares punto/peso.
    """
    if n not in GAUSS_TABLE:
        raise ValueError(
            f"La Tabla 1 del documento cubre n = 1..{MAX_ORDER}; se pidio n = {n}."
        )
    return list(GAUSS_TABLE[n])


def gauss_2d(n_xi: int, n_eta: int | None = None) -> list[tuple[float, float, float]]:
    """Producto tensorial de dos reglas 1D: [(xi, eta, W_i*W_j), ...] (ec. 1.5.2).

    Para que sirve: la integracion bidimensional del documento es el producto
    de dos integraciones unidimensionales, con peso combinado W_i*W_j.

    El recorrido sigue el orden de los nodos del elemento (--, +-, ++, -+)
    cuando n = 2, de modo que el punto de Gauss i cae en el cuadrante del
    nodo i (util para la lectura didactica paso a paso).
    """
    if n_eta is None:
        n_eta = n_xi
    pts_xi = gauss_1d(n_xi)
    pts_eta = gauss_1d(n_eta)
    if n_xi == 2 and n_eta == 2:
        # Orden por cuadrantes, igual que la numeracion de nodos del elemento
        (a, wa), (b, wb) = pts_xi
        (c, wc), (d, wd) = pts_eta
        return [
            (a, c, wa * wc),   # (--)
            (b, c, wb * wc),   # (+-)
            (b, d, wb * wd),   # (++)
            (a, d, wa * wd),   # (-+)
        ]
    return [
        (xi, eta, w_xi * w_eta)
        for xi, w_xi in pts_xi
        for eta, w_eta in pts_eta
    ]


def integrate_1d(f, n: int) -> float:
    """Integra f(xi) en [-1, 1] con la regla de n puntos (ec. 1.5.1).

    I = W_1*Phi(xi_1) + W_2*Phi(xi_2) + ... + W_n*Phi(xi_n)
    """
    return float(sum(w * f(xi) for xi, w in gauss_1d(n)))


def integrate_2d(f, n_xi: int, n_eta: int | None = None):
    """Integra f(xi, eta) en [-1, 1]x[-1, 1] con la regla n_xi x n_eta (ec. 1.5.2).

    Acepta integrandos escalares o matriciales (se acumula con el operador +,
    que numpy resuelve elemento a elemento).
    """
    total = None
    for xi, eta, w in gauss_2d(n_xi, n_eta):
        term = w * f(xi, eta)
        total = term if total is None else total + term
    return total


def table_rows() -> list[dict]:
    """Tabla 1 en forma de filas listas para mostrar en la UI o exportar.

    Para que sirve: el paso didactico de integracion numerica muestra la
    tabla completa del documento (n = 1..6), no solo la regla que usa el
    elemento; asi el usuario ve de donde salen +-1/raiz(3) y los pesos 1.
    """
    filas = []
    for n in sorted(GAUSS_TABLE):
        pts = GAUSS_TABLE[n]
        filas.append({
            "n": n,
            "Puntos xi_i": ";  ".join(f"{xi:+.10f}" for xi, _ in pts),
            "Pesos W_i": ";  ".join(f"{w:.10f}" for _, w in pts),
            "Grado exacto": 2 * n - 1,
        })
    return filas
