"""Test del núcleo PLATE (elemento rectangular 12 GDL, Kirchhoff).

Verificaciones:
  1. K^e simétrica y con EXACTAMENTE 3 modos rígidos (w constante,
     inclinación en x, inclinación en y).
  2. Estado de curvatura constante reproducido EXACTO (w = x², w = y²,
     w = xy están contenidos en el polinomio de 12 términos).
  3. Benchmark analítico (Timoshenko): placa cuadrada simplemente
     apoyada con carga uniforme q:
         w_centro = alfa·q·a⁴/D_flex   con alfa = 0.00406 (nu = 0.3)
         Mx_centro = beta·q·a²         con beta = 0.0479  (nu = 0.3)
         D_flex = E·t³/(12(1-nu²))
     La malla se refina 4x4 -> 8x8 -> 12x12 y el error debe DISMINUIR
     y terminar por debajo del 1 % en w.

Ejecutar:  .\\venv\\Scripts\\python.exe test_plate.py
"""
from __future__ import annotations
import sys
import numpy as np

sys.path.insert(0, ".")

from src.fem.node_plate import NodePlate
from src.fem.plate_element import PlateElement
from src.fem.structure_plate import StructurePlate
from src.fem.solver_plate import solve_plate


def make_mesh(a: float, b: float, nx: int, ny: int,
              E: float, nu: float, t: float) -> StructurePlate:
    """Malla rectangular nx x ny de elementos plate (nodos CCW desde inf-izq)."""
    s = StructurePlate()
    dx, dy = a / nx, b / ny
    node_at: dict[tuple[int, int], NodePlate] = {}
    nid = 0
    for j in range(ny + 1):
        for i in range(nx + 1):
            n = NodePlate(id=nid, x=i * dx, y=j * dy)
            s.add_node(n)
            node_at[(i, j)] = n
            nid += 1
    eid = 0
    for j in range(ny):
        for i in range(nx):
            n1 = node_at[(i, j)]           # (--)
            n2 = node_at[(i + 1, j)]       # (+-)
            n3 = node_at[(i + 1, j + 1)]   # (++)
            n4 = node_at[(i, j + 1)]       # (-+)
            s.add_element(PlateElement(id=eid, nodes=[n1, n2, n3, n4],
                                       E=E, nu=nu, t=t))
            eid += 1
    return s


def apply_simply_supported(s: StructurePlate, a: float, b: float) -> None:
    """Apoyo simple 'duro' en los 4 bordes.

    En cada borde: w = 0 y giro alrededor del eje NORMAL al borde = 0
    (consecuencia de w = 0 a lo largo del borde recto). El giro de
    flexión alrededor del eje del borde queda libre.
    """
    tol = 1e-9
    for n in s.nodes:
        en_x0 = abs(n.x - 0.0) < tol
        en_xa = abs(n.x - a) < tol
        en_y0 = abs(n.y - 0.0) < tol
        en_yb = abs(n.y - b) < tol
        if en_x0 or en_xa or en_y0 or en_yb:
            n.restraint_w = True
        # Borde paralelo a Y (x = 0 o x = a): dw/dy = 0 -> theta_x = 0
        if en_x0 or en_xa:
            n.restraint_rx = True
        # Borde paralelo a X (y = 0 o y = b): dw/dx = 0 -> theta_y = 0
        if en_y0 or en_yb:
            n.restraint_ry = True


def main() -> None:
    E, nu, t = 2.1e11, 0.3, 0.01
    ok = True

    # ---------- 1. Modos rigidos y simetria ----------
    nodes = [NodePlate(0, 0.0, 0.0), NodePlate(1, 0.8, 0.0),
             NodePlate(2, 0.8, 0.5), NodePlate(3, 0.0, 0.5)]
    el = PlateElement(0, nodes, E=E, nu=nu, t=t)
    K = el.stiffness_matrix()
    sim = float(np.max(np.abs(K - K.T)))
    eig = np.linalg.eigvalsh(K)
    n_rigidos = int(np.sum(np.abs(eig) < np.max(eig) * 1e-12))
    print(f"[1] K^e (12x12): asimetria max = {sim:.3e}   "
          f"modos rigidos = {n_rigidos} (deben ser 3)")
    ok &= sim < 1e-4 * np.max(np.abs(K)) and n_rigidos == 3

    # ---------- 2. Curvatura constante exacta ----------
    # w = x^2 -> curvaturas [2, 0, 0]; w = xy -> [0, 0, 2]
    rng = np.random.default_rng(3)
    for w_fun, dwdx, dwdy, kappa_exacta, nombre in [
        (lambda x, y: x * x, lambda x, y: 2 * x, lambda x, y: 0.0,
         np.array([2.0, 0.0, 0.0]), "w = x^2"),
        (lambda x, y: y * y, lambda x, y: 0.0, lambda x, y: 2 * y,
         np.array([0.0, 2.0, 0.0]), "w = y^2"),
        (lambda x, y: x * y, lambda x, y: y, lambda x, y: x,
         np.array([0.0, 0.0, 2.0]), "w = x*y"),
    ]:
        delta = np.zeros(12)
        for i, n in enumerate(el.nodes):
            delta[3 * i] = w_fun(n.x, n.y)
            delta[3 * i + 1] = dwdy(n.x, n.y)      # theta_x = dw/dy
            delta[3 * i + 2] = -dwdx(n.x, n.y)     # theta_y = -dw/dx
        err = 0.0
        for _ in range(20):
            x = rng.uniform(0, 0.8)
            y = rng.uniform(0, 0.5)
            kap = el.curvatures_at(x, y, delta)
            err = max(err, float(np.max(np.abs(kap - kappa_exacta))))
        print(f"[2] Curvatura constante {nombre}: error max = {err:.3e}")
        ok &= err < 1e-9

    # ---------- 3. Benchmark placa simplemente apoyada ----------
    a = 1.0
    q = -1000.0                          # N/m2 hacia abajo
    D_flex = E * t ** 3 / (12.0 * (1.0 - nu * nu))
    w_exacto = 0.00406 * q * a ** 4 / D_flex      # Timoshenko, nu = 0.3
    Mx_exacto = 0.0479 * abs(q) * a ** 2

    print(f"[3] Placa cuadrada SS con q uniforme: w_exacto = {w_exacto:.6e} m")
    errores = []
    for N in (4, 8, 12):
        s = make_mesh(a, a, N, N, E, nu, t)
        apply_simply_supported(s, a, a)
        res = solve_plate(s, q_uniform=q)
        # w en el nodo central (N par -> existe nodo en el centro)
        centro = next(n for n in s.nodes
                      if abs(n.x - a / 2) < 1e-9 and abs(n.y - a / 2) < 1e-9)
        w_fem = res.displacements[centro.dofs[0]]
        err = abs(w_fem - w_exacto) / abs(w_exacto) * 100.0
        errores.append(err)
        # Momento Mx en el centro (elemento que contiene al centro)
        el_c = s.elements[(N // 2) * N + N // 2]   # elemento arriba-derecha del centro
        u_e = res.displacements[el_c.global_dofs()]
        Mx_fem = abs(el_c.moments_at(a / 2, a / 2, u_e)[0])
        err_m = abs(Mx_fem - Mx_exacto) / Mx_exacto * 100.0
        print(f"    Malla {N:2d}x{N:<2d}: w = {w_fem:.6e} m "
              f"(err {err:5.2f} %)   Mx = {Mx_fem:.4f} N.m/m (err {err_m:5.2f} %)")
    print(f"    Convergencia del error en w: {errores[0]:.2f} % -> "
          f"{errores[1]:.2f} % -> {errores[2]:.2f} %")
    ok &= errores[2] < 1.0 and errores[2] <= errores[1] <= errores[0] + 1e-9

    # ---------- 4. Simetria del resultado ----------
    # La placa y la carga son simetricas: w en 4 puntos simetricos igual
    s = make_mesh(a, a, 8, 8, E, nu, t)
    apply_simply_supported(s, a, a)
    res = solve_plate(s, q_uniform=q)
    pts = [(0.25, 0.25), (0.75, 0.25), (0.75, 0.75), (0.25, 0.75)]
    ws = []
    for (px, py) in pts:
        n = next(nd for nd in s.nodes
                 if abs(nd.x - px) < 1e-9 and abs(nd.y - py) < 1e-9)
        ws.append(res.displacements[n.dofs[0]])
    asim = max(abs(w - ws[0]) for w in ws)
    print(f"[4] Simetria (4 puntos espejo): dispersion = {asim:.3e} m")
    ok &= asim < 1e-12 * abs(w_exacto) * 1e6

    print()
    if ok:
        print("RESULTADO: todos los checks del nucleo PLATE pasan.")
    else:
        print("RESULTADO: HAY CHECKS FALLIDOS — revisar arriba.")
        sys.exit(1)


if __name__ == "__main__":
    main()
