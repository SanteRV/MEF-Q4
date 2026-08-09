"""Criterios de convergencia — cap. 01.01.08 del documento teórico.

El documento no se conforma con refinar la malla: exige comprobar que la
formulación de cada elemento cumple los criterios que garantizan que ese
refinamiento converja. Este módulo ejecuta esas comprobaciones sobre los
elementos realmente implementados (plane Q4 y plate de 12 GDL):

    01.01.08.02  Criterio de cuerpo rígido        K·q_rigido = 0
    01.01.08.03  Criterio de deformación constante ε (o κ) uniforme
    01.01.08.05  Prueba del parche + autovalores de K^e

Modos de cuerpo rígido esperados (documento, 01.01.08.05):
    plane 3    plate thin 3    flat shell 6

Se añade el elemento frame 3D (cap. 01.02), que el documento no incluye en
el capítulo de convergencia pero admite las mismas comprobaciones: una
barra libre en el espacio tiene 6 movimientos rígidos, y la matriz de
transformación de la ec. 2.3.1 debe ser ortonormal para que K = Tᵀ·k·T no
introduzca energía espuria al cambiar de sistema de coordenadas.

Todo se devuelve como una lista de Check para poder mostrarla en la UI y
exportarla, en lugar de imprimir por consola.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .frame_element import FrameElement
from .node import Node
from .node_frame import NodeFrame
from .node_plate import NodePlate
from .node_shell import NodeShell
from .plate_element import PlateElement
from .q4_element import GAUSS_2X2, Q4Element
from .shell_element import ShellElement
from .solver import solve
from .solver_plate import solve_plate
from .structure import Structure
from .structure_plate import StructurePlate


@dataclass
class Check:
    """Resultado de UNA comprobación de la teoría."""
    criterio: str        # capítulo del documento
    nombre: str          # qué se comprobó
    valor: float         # error o cantidad medida
    tolerancia: float    # umbral de aceptación
    ok: bool
    detalle: str = ""

    @property
    def estado(self) -> str:
        return "CUMPLE" if self.ok else "NO CUMPLE"


# ---------------------------------------------------------------------------
# Elementos de referencia para las comprobaciones
# ---------------------------------------------------------------------------
def _q4_referencia(E: float, nu: float, t: float, plane_stress: bool) -> Q4Element:
    """Q4 DISTORSIONADO (no rectangular): el caso exigente del criterio."""
    nodos = [
        Node(0, 0.00, 0.00),
        Node(1, 2.10, 0.15),
        Node(2, 1.85, 1.30),
        Node(3, 0.20, 1.05),
    ]
    return Q4Element(0, nodos, E=E, nu=nu, t=t, plane_stress=plane_stress)


def _plate_referencia(E: float, nu: float, t: float) -> PlateElement:
    """Plate rectangular (la formulación de 12 GDL solo admite rectángulos)."""
    nodos = [
        NodePlate(0, 0.0, 0.0),
        NodePlate(1, 1.6, 0.0),
        NodePlate(2, 1.6, 1.1),
        NodePlate(3, 0.0, 1.1),
    ]
    return PlateElement(0, nodos, E=E, nu=nu, t=t)


def _frame_referencia(E: float, nu: float) -> FrameElement:
    """Barra frame en una dirección oblicua, para exigir la transformación."""
    nodos = [NodeFrame(0, 0.0, 0.0, 0.0), NodeFrame(1, 2.0, -1.0, 3.0)]
    G = E / (2.0 * (1.0 + nu))
    return FrameElement(0, nodos, E=E, G=G, A=0.01,
                        Iy=8.333e-6, Iz=8.333e-6, J=1.6e-5, psi=0.3)


def _shell_referencia(E: float, nu: float, t: float) -> ShellElement:
    """Flat shell rectangular (hereda la restricción del plate)."""
    nodos = [
        NodeShell(0, 0.0, 0.0),
        NodeShell(1, 1.6, 0.0),
        NodeShell(2, 1.6, 1.1),
        NodeShell(3, 0.0, 1.1),
    ]
    return ShellElement(0, nodos, E=E, nu=nu, t=t)


# ---------------------------------------------------------------------------
# 01.01.08.02 — Criterio de cuerpo rígido
# ---------------------------------------------------------------------------
def cuerpo_rigido_plane(el: Q4Element) -> list[Check]:
    """Los 3 movimientos rígidos del plane no deben generar esfuerzos.

    El documento lo plantea de dos maneras equivalentes y aquí se verifican
    las dos: las deformaciones deben anularse (εx = εy = γxy = 0) y el vector
    de desplazamientos rígidos debe pertenecer al núcleo de K^e (K·q = 0).
    """
    K, _ = el.stiffness_matrix()
    escala = float(np.max(np.abs(K)))
    checks: list[Check] = []

    modos = {
        "Traslación en X": [(1.0, 0.0) for _ in el.nodes],
        "Traslación en Y": [(0.0, 1.0) for _ in el.nodes],
        # Rotación infinitesimal alrededor de Z: u = -y, v = x
        "Rotación alrededor de Z": [(-n.y, n.x) for n in el.nodes],
    }
    for nombre, campo in modos.items():
        q = np.array([c for par in campo for c in par], dtype=float)
        fuerzas = float(np.max(np.abs(K @ q)))
        norma = float(np.max(np.abs(q))) or 1.0
        eps_max = 0.0
        for xi, eta, _ in GAUSS_2X2:
            eps, _sig = el.strains_stresses_at(xi, eta, q)
            eps_max = max(eps_max, float(np.max(np.abs(eps))))
        checks.append(Check(
            criterio="01.01.08.02",
            nombre=f"Cuerpo rígido plane — {nombre}",
            valor=max(fuerzas / (escala * norma), eps_max),
            tolerancia=1e-10,
            ok=(fuerzas < 1e-10 * escala * norma and eps_max < 1e-10),
            detalle=(f"max|K·q| = {fuerzas:.3e} (escala K = {escala:.3e}); "
                     f"max|ε| = {eps_max:.3e}"),
        ))
    return checks


def cuerpo_rigido_plate(el: PlateElement) -> list[Check]:
    """Los 3 movimientos rígidos del plate thin: w constante y los 2 giros.

    Corresponden a w = cte, w = c·y y w = c·x, todos con curvatura nula.
    """
    K = el.stiffness_matrix()
    escala = float(np.max(np.abs(K)))
    checks: list[Check] = []

    # (w, θx = ∂w/∂y, θy = -∂w/∂x) para cada campo rígido
    modos = {
        "Traslación transversal (w = cte)":
            [(1.0, 0.0, 0.0) for _ in el.nodes],
        "Giro global alrededor de X (w = y)":
            [(n.y, 1.0, 0.0) for n in el.nodes],
        "Giro global alrededor de Y (w = x)":
            [(n.x, 0.0, -1.0) for n in el.nodes],
    }
    xc = float(np.mean([n.x for n in el.nodes]))
    yc = float(np.mean([n.y for n in el.nodes]))
    for nombre, campo in modos.items():
        d = np.array([c for terna in campo for c in terna], dtype=float)
        fuerzas = float(np.max(np.abs(K @ d)))
        norma = float(np.max(np.abs(d))) or 1.0
        kappa = float(np.max(np.abs(el.curvatures_at(xc, yc, d))))
        checks.append(Check(
            criterio="01.01.08.02",
            nombre=f"Cuerpo rígido plate — {nombre}",
            valor=max(fuerzas / (escala * norma), kappa),
            tolerancia=1e-10,
            ok=(fuerzas < 1e-10 * escala * norma and kappa < 1e-10),
            detalle=(f"max|K·q| = {fuerzas:.3e} (escala K = {escala:.3e}); "
                     f"max|κ| = {kappa:.3e}"),
        ))
    return checks


# ---------------------------------------------------------------------------
# 01.01.08.03 — Criterio de deformación constante
# ---------------------------------------------------------------------------
def deformacion_constante_plane(el: Q4Element) -> list[Check]:
    """Un campo lineal de desplazamientos debe dar ε constante y exacta.

    Se impone u = a1·x + a2·y, v = b1·x + b2·y, cuyas deformaciones teóricas
    son εx = a1, εy = b2, γxy = a2 + b1 en TODO el elemento.
    """
    a1, a2, b1, b2 = 3.0e-4, -1.2e-4, 7.0e-5, 2.5e-4
    objetivo = np.array([a1, b2, a2 + b1])
    q = np.array([v for n in el.nodes
                  for v in (a1 * n.x + a2 * n.y, b1 * n.x + b2 * n.y)])
    err = 0.0
    for xi, eta, _ in GAUSS_2X2:
        eps, _s = el.strains_stresses_at(xi, eta, q)
        err = max(err, float(np.max(np.abs(eps - objetivo))))
    ref = float(np.max(np.abs(objetivo)))
    return [Check(
        criterio="01.01.08.03",
        nombre="Deformación constante plane (campo lineal de u, v)",
        valor=err / ref,
        tolerancia=1e-12,
        ok=err < 1e-12 * ref,
        detalle=(f"ε objetivo = [{objetivo[0]:.6g}, {objetivo[1]:.6g}, "
                 f"{objetivo[2]:.6g}]; error máximo en los GP = {err:.3e}"),
    )]


def deformacion_constante_plate(el: PlateElement) -> list[Check]:
    """Curvatura constante: el polinomio de 12 términos contiene x², y², xy.

    Se imponen los tres estados que exige el documento — flexión pura
    alrededor de cada eje y torsión pura — con w = A·x² + B·y² + C·xy.
    """
    checks: list[Check] = []
    rng = np.random.default_rng(7)
    xs = [n.x for n in el.nodes]
    ys = [n.y for n in el.nodes]
    casos = {
        "Flexión pura en X (w = x²)": (1.0, 0.0, 0.0),
        "Flexión pura en Y (w = y²)": (0.0, 1.0, 0.0),
        "Torsión pura (w = x·y)": (0.0, 0.0, 1.0),
    }
    for nombre, (A, B, C) in casos.items():
        objetivo = np.array([2.0 * A, 2.0 * B, 2.0 * C])
        d = np.zeros(12)
        for i, n in enumerate(el.nodes):
            d[3 * i] = A * n.x ** 2 + B * n.y ** 2 + C * n.x * n.y
            d[3 * i + 1] = 2.0 * B * n.y + C * n.x        # θx =  ∂w/∂y
            d[3 * i + 2] = -(2.0 * A * n.x + C * n.y)     # θy = -∂w/∂x
        err = 0.0
        for _ in range(25):
            x = rng.uniform(min(xs), max(xs))
            y = rng.uniform(min(ys), max(ys))
            err = max(err, float(np.max(np.abs(
                el.curvatures_at(x, y, d) - objetivo))))
        checks.append(Check(
            criterio="01.01.08.03",
            nombre=f"Curvatura constante plate — {nombre}",
            valor=err,
            tolerancia=1e-9,
            ok=err < 1e-9,
            detalle=(f"κ objetivo = [{objetivo[0]:.6g}, {objetivo[1]:.6g}, "
                     f"{objetivo[2]:.6g}]; error máximo = {err:.3e}"),
        ))
    return checks


# ---------------------------------------------------------------------------
# 01.01.08.05 — Autovalores de K^e (paso previo a la prueba del parche)
# ---------------------------------------------------------------------------
def _check_autovalores(K: np.ndarray, esperados: int, etiqueta: str) -> Check:
    """Cuenta los autovalores nulos de K^e y los compara con la teoría.

    Autovalores nulos de más = modos espurios de energía nula (mecanismos);
    de menos = el elemento no representa todos los movimientos rígidos.
    """
    eig = np.linalg.eigvalsh(K)
    escala = float(np.max(np.abs(eig)))
    nulos = int(np.sum(np.abs(eig) < escala * 1e-10))
    return Check(
        criterio="01.01.08.05",
        nombre=f"Autovalores de K^e — {etiqueta}",
        valor=float(nulos),
        tolerancia=float(esperados),
        ok=(nulos == esperados),
        detalle=(f"{nulos} autovalores nulos (la teoría exige {esperados}); "
                 f"primer autovalor no nulo = "
                 f"{min(v for v in eig if abs(v) >= escala * 1e-10):.6e}"),
    )


def autovalores_plane(el: Q4Element) -> list[Check]:
    K, _ = el.stiffness_matrix()
    return [_check_autovalores(K, 3, "plane Q4 (8 GDL)")]


def autovalores_plate(el: PlateElement) -> list[Check]:
    return [_check_autovalores(el.stiffness_matrix(), 3, "plate thin (12 GDL)")]


def autovalores_frame(el: "FrameElement") -> list[Check]:
    """La barra frame libre en el espacio tiene los 6 movimientos rígidos.

    Se comprueba sobre k en el SCL y sobre K = Tᵀ·k·T en el SCG: la
    transformación de la ec. 2.4.2 es ortogonal, así que no puede cambiar
    el número de modos de energía nula.
    """
    return [
        _check_autovalores(el.stiffness_local(), 6, "frame 3D, k en el SCL"),
        _check_autovalores(el.stiffness_matrix(), 6, "frame 3D, K en el SCG"),
    ]


def transformacion_frame(el: "FrameElement") -> list[Check]:
    """La matriz r de la ec. 2.3.1 debe ser ortonormal y directa.

    Sus filas son los ejes locales expresados en el global: si dejaran de
    ser ortonormales, la ec. 2.4.2 (K = Tᵀ·k·T) introduciría energía
    espuria al cambiar de sistema de coordenadas.
    """
    r = el.rotation_matrix()
    err_ort = float(np.max(np.abs(r @ r.T - np.eye(3))))
    err_det = abs(float(np.linalg.det(r)) - 1.0)
    valor = max(err_ort, err_det)
    return [Check(
        criterio="01.02.01.03",
        nombre="Frame 3D — matriz de transformación ortonormal",
        valor=valor,
        tolerancia=1e-12,
        ok=valor < 1e-12,
        detalle=(f"max|r·rᵀ − I| = {err_ort:.3e}; |det(r) − 1| = {err_det:.3e}"),
    )]


def autovalores_shell(el: ShellElement) -> list[Check]:
    """El flat shell debe presentar los 6 movimientos rígidos del espacio.

    Salen de superponer los 3 del plane (2 traslaciones en el plano y el
    giro alrededor de Z, contenido en el campo u = -y, v = x) con los 3 del
    plate (traslación transversal y los 2 giros).
    """
    return [_check_autovalores(el.stiffness_matrix(), 6,
                               "flat shell (20 GDL)")]


# ---------------------------------------------------------------------------
# 01.01.08.05 — Desacoplamiento membrana / flexión del flat shell
# ---------------------------------------------------------------------------
def desacoplamiento_shell(el: ShellElement) -> list[Check]:
    """Las acciones de membrana no deben producir respuesta fuera del plano.

    El documento lo exige explícitamente (01.01.08.05, último párrafo): al
    ser la matriz de rigidez por bloques desacoplados, una acción de
    membrana solo puede generar respuesta en el plano y una acción de
    flexión solo fuera del plano. Se comprueba sobre K^e: las casillas que
    cruzan (u, v) con (w, θx, θy) deben ser exactamente cero.
    """
    K = el.stiffness_matrix()
    escala = float(np.max(np.abs(K)))
    idx_m = [5 * i + c for i in range(4) for c in (0, 1)]
    idx_b = [5 * i + c for i in range(4) for c in (2, 3, 4)]
    acople = float(np.max(np.abs(K[np.ix_(idx_m, idx_b)])))
    return [Check(
        criterio="01.01.08.05",
        nombre="Flat shell — desacoplamiento membrana / flexión",
        valor=acople / escala,
        tolerancia=0.0,
        ok=(acople == 0.0),
        detalle=(f"máximo acoplamiento cruzado en K^e = {acople:.3e} "
                 f"(escala K = {escala:.3e}); la ec. 1.4.10 exige que sea 0"),
    )]


def cuerpo_rigido_shell(el: ShellElement) -> list[Check]:
    """Los 6 movimientos rígidos del flat shell no deben generar esfuerzos."""
    K = el.stiffness_matrix()
    escala = float(np.max(np.abs(K)))
    checks: list[Check] = []
    # (u, v, w, θx, θy) por nodo para cada movimiento rígido del espacio
    modos = {
        "Traslación en X": lambda n: (1.0, 0.0, 0.0, 0.0, 0.0),
        "Traslación en Y": lambda n: (0.0, 1.0, 0.0, 0.0, 0.0),
        "Traslación en Z": lambda n: (0.0, 0.0, 1.0, 0.0, 0.0),
        "Giro alrededor de Z": lambda n: (-n.y, n.x, 0.0, 0.0, 0.0),
        "Giro alrededor de X": lambda n: (0.0, 0.0, n.y, 1.0, 0.0),
        "Giro alrededor de Y": lambda n: (0.0, 0.0, n.x, 0.0, -1.0),
    }
    for nombre, campo in modos.items():
        q = np.array([c for n in el.nodes for c in campo(n)], dtype=float)
        fuerzas = float(np.max(np.abs(K @ q)))
        norma = float(np.max(np.abs(q))) or 1.0
        checks.append(Check(
            criterio="01.01.08.02",
            nombre=f"Cuerpo rígido flat shell — {nombre}",
            valor=fuerzas / (escala * norma),
            tolerancia=1e-10,
            ok=fuerzas < 1e-10 * escala * norma,
            detalle=f"max|K·q| = {fuerzas:.3e} (escala K = {escala:.3e})",
        ))
    return checks


# ---------------------------------------------------------------------------
# 01.01.08.05 — Prueba del parche
# ---------------------------------------------------------------------------
def prueba_parche_plane(E: float, nu: float, t: float,
                        plane_stress: bool = True) -> list[Check]:
    """Parche de 4 Q4 con un nodo interior DESPLAZADO de su posición regular.

    Se imponen en el contorno los desplazamientos de un campo lineal exacto
    (como desplazamientos conocidos, U_c de la ec. 1.6.1), se resuelve SIN
    cargas y se comprueba que el nodo interior reproduzca el valor teórico y
    que las deformaciones sean constantes en todos los puntos de integración.
    """
    a1, a2, b1, b2 = 2.0e-4, -8.0e-5, 5.0e-5, 1.5e-4
    objetivo = np.array([a1, b2, a2 + b1])

    def campo(x: float, y: float) -> tuple[float, float]:
        return (a1 * x + a2 * y, b1 * x + b2 * y)

    # Malla 2x2 sobre el cuadrado [0,2]x[0,2]; el nodo central (índice 4)
    # se corre para que los elementos queden distorsionados.
    coords = [
        (0.0, 0.0), (1.0, 0.0), (2.0, 0.0),
        (0.0, 1.0), (1.28, 0.83), (2.0, 1.0),
        (0.0, 2.0), (1.0, 2.0), (2.0, 2.0),
    ]
    s = Structure()
    for i, (x, y) in enumerate(coords):
        s.add_node(Node(i, x, y))
    conect = [(0, 1, 4, 3), (1, 2, 5, 4), (4, 5, 8, 7), (3, 4, 7, 6)]
    for eid, (n1, n2, n3, n4) in enumerate(conect):
        s.add_element(Q4Element(
            eid, [s.nodes[n1], s.nodes[n2], s.nodes[n3], s.nodes[n4]],
            E=E, nu=nu, t=t, plane_stress=plane_stress))

    interior = 4
    for n in s.nodes:
        if n.id == interior:
            continue
        n.restraint_x = n.restraint_y = True
        n.prescribed_x, n.prescribed_y = campo(n.x, n.y)

    res = solve(s)
    nodo = s.nodes[interior]
    ux, uy = res.displacements[nodo.dofs[0]], res.displacements[nodo.dofs[1]]
    ux_t, uy_t = campo(nodo.x, nodo.y)
    ref = max(abs(ux_t), abs(uy_t)) or 1.0
    err_u = max(abs(ux - ux_t), abs(uy - uy_t))

    err_eps = 0.0
    for el_res in res.elements:
        for eps in el_res.strains_at_gauss:
            err_eps = max(err_eps, float(np.max(np.abs(eps - objetivo))))
    ref_eps = float(np.max(np.abs(objetivo)))

    return [
        Check(
            criterio="01.01.08.05",
            nombre="Prueba del parche plane — nodo interior",
            valor=err_u / ref,
            tolerancia=1e-10,
            ok=err_u < 1e-10 * ref,
            detalle=(f"u interior = ({ux:.9e}, {uy:.9e}); "
                     f"teórico = ({ux_t:.9e}, {uy_t:.9e})"),
        ),
        Check(
            criterio="01.01.08.05",
            nombre="Prueba del parche plane — ε constante en los GP",
            valor=err_eps / ref_eps,
            tolerancia=1e-10,
            ok=err_eps < 1e-10 * ref_eps,
            detalle=(f"error máximo en los 16 puntos de Gauss = {err_eps:.3e} "
                     f"frente a ε = [{objetivo[0]:.6g}, {objetivo[1]:.6g}, "
                     f"{objetivo[2]:.6g}]"),
        ),
    ]


def prueba_parche_plate(E: float, nu: float, t: float) -> list[Check]:
    """Parche de 3x3 elementos rectangulares con 4 nodos interiores.

    El elemento no conforme de 12 GDL solo supera la prueba con mallas
    RECTANGULARES (01.01.08.04), por eso el parche no se distorsiona; lo que
    se distorsiona es el espaciado, para que los elementos tengan tamaños
    distintos. Se impone en el contorno un estado de curvatura constante
    (flexión en x + flexión en y + torsión) y se comprueba que los nodos
    interiores reproduzcan el campo exacto.
    """
    A, B, C = 1.5e-3, -9.0e-4, 6.0e-4          # w = A·x² + B·y² + C·x·y
    objetivo = np.array([2.0 * A, 2.0 * B, 2.0 * C])

    def campo(x: float, y: float) -> tuple[float, float, float]:
        w = A * x ** 2 + B * y ** 2 + C * x * y
        th_x = 2.0 * B * y + C * x            # θx =  ∂w/∂y
        th_y = -(2.0 * A * x + C * y)         # θy = -∂w/∂x
        return (w, th_x, th_y)

    xs = [0.0, 0.7, 1.9, 2.5]                  # espaciado irregular
    ys = [0.0, 0.5, 1.4, 2.0]
    s = StructurePlate()
    idx: dict[tuple[int, int], int] = {}
    nid = 0
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            s.add_node(NodePlate(nid, x, y))
            idx[(i, j)] = nid
            nid += 1
    eid = 0
    for j in range(len(ys) - 1):
        for i in range(len(xs) - 1):
            s.add_element(PlateElement(
                eid,
                [s.nodes[idx[(i, j)]], s.nodes[idx[(i + 1, j)]],
                 s.nodes[idx[(i + 1, j + 1)]], s.nodes[idx[(i, j + 1)]]],
                E=E, nu=nu, t=t))
            eid += 1

    interiores = [idx[(i, j)] for i in (1, 2) for j in (1, 2)]
    for n in s.nodes:
        if n.id in interiores:
            continue
        n.restraint_w = n.restraint_rx = n.restraint_ry = True
        n.prescribed_w, n.prescribed_rx, n.prescribed_ry = campo(n.x, n.y)

    res = solve_plate(s)
    err_u, ref = 0.0, 0.0
    for i in interiores:
        n = s.nodes[i]
        obtenido = res.displacements[list(n.dofs)]
        teorico = np.array(campo(n.x, n.y))
        err_u = max(err_u, float(np.max(np.abs(obtenido - teorico))))
        ref = max(ref, float(np.max(np.abs(teorico))))
    ref = ref or 1.0

    err_k = 0.0
    for el in s.elements:
        d = res.displacements[el.global_dofs()]
        a_lado, b_lado = el.sides
        x0 = min(n.x for n in el.nodes)
        y0 = min(n.y for n in el.nodes)
        for fx in (0.25, 0.75):
            for fy in (0.25, 0.75):
                kappa = el.curvatures_at(x0 + fx * a_lado, y0 + fy * b_lado, d)
                err_k = max(err_k, float(np.max(np.abs(kappa - objetivo))))
    ref_k = float(np.max(np.abs(objetivo)))

    return [
        Check(
            criterio="01.01.08.05",
            nombre="Prueba del parche plate — nodos interiores",
            valor=err_u / ref,
            tolerancia=1e-9,
            ok=err_u < 1e-9 * ref,
            detalle=(f"error máximo en los {len(interiores)} nodos interiores "
                     f"= {err_u:.3e} (referencia {ref:.3e})"),
        ),
        Check(
            criterio="01.01.08.05",
            nombre="Prueba del parche plate — κ constante",
            valor=err_k / ref_k,
            tolerancia=1e-9,
            ok=err_k < 1e-9 * ref_k,
            detalle=(f"κ objetivo = [{objetivo[0]:.6g}, {objetivo[1]:.6g}, "
                     f"{objetivo[2]:.6g}]; error máximo = {err_k:.3e}"),
        ),
    ]


# ---------------------------------------------------------------------------
# Ejecución completa
# ---------------------------------------------------------------------------
def run_all(E: float = 2.1e11, nu: float = 0.3, t: float = 0.01,
            plane_stress: bool = True) -> list[Check]:
    """Ejecuta todas las comprobaciones del cap. 01.01.08 y devuelve la lista.

    Los criterios son propiedades de la FORMULACIÓN, no del modelo que el
    usuario dibujó: por eso se corren sobre elementos y parches de referencia
    generados aquí, tomando solo el material del proyecto.
    """
    q4 = _q4_referencia(E, nu, t, plane_stress)
    pl = _plate_referencia(E, nu, t)
    sh = _shell_referencia(E, nu, t)
    fr = _frame_referencia(E, nu)
    checks: list[Check] = []
    checks += cuerpo_rigido_plane(q4)
    checks += cuerpo_rigido_plate(pl)
    checks += cuerpo_rigido_shell(sh)
    checks += deformacion_constante_plane(q4)
    checks += deformacion_constante_plate(pl)
    checks += autovalores_plane(q4)
    checks += autovalores_plate(pl)
    checks += autovalores_shell(sh)
    checks += autovalores_frame(fr)
    checks += desacoplamiento_shell(sh)
    checks += transformacion_frame(fr)
    checks += prueba_parche_plane(E, nu, t, plane_stress)
    checks += prueba_parche_plate(E, nu, t)
    return checks
