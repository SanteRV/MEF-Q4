"""Test del nucleo SHELL (flat shell rectangular de 20 GDL, cap. 01.01.04).

Verificaciones:
  1. Tamano del sistema del ejemplo de la figura 15 del documento:
     4 elementos, 9 nodos, 5 GDL por nodo -> matriz global de 45x45.
  2. K^e simetrica, 6 modos rigidos y bloques membrana/flexion desacoplados
     (ec. 1.4.10 y criterio 01.01.08.05).
  3. Superposicion exacta: una carga SOLO en el plano debe reproducir el
     resultado del modelo plane y no mover nada fuera del plano; una carga
     SOLO transversal debe reproducir el modelo plate y no mover nada en el
     plano. Es la comprobacion que pide el ultimo parrafo de 01.01.08.05.
  4. Benchmark de Timoshenko para la parte de flexion (placa cuadrada
     simplemente apoyada), heredado del plate.

Ejecutar:  .\\venv\\Scripts\\python.exe test_shell.py
"""
from __future__ import annotations
import sys
import numpy as np

sys.path.insert(0, ".")

from src.fem.node import Node
from src.fem.node_plate import NodePlate
from src.fem.node_shell import NodeShell
from src.fem.plate_element import PlateElement
from src.fem.q4_element import Q4Element
from src.fem.shell_element import ShellElement
from src.fem.solver import solve
from src.fem.solver_plate import solve_plate
from src.fem.solver_shell import solve_shell
from src.fem.structure import Structure
from src.fem.structure_plate import StructurePlate
from src.fem.structure_shell import StructureShell


def make_shell_mesh(a, b, nx, ny, E, nu, t) -> StructureShell:
    """Malla nx x ny de elementos flat shell (nodos CCW desde inf-izq)."""
    s = StructureShell()
    dx, dy = a / nx, b / ny
    at: dict[tuple[int, int], NodeShell] = {}
    nid = 0
    for j in range(ny + 1):
        for i in range(nx + 1):
            n = NodeShell(id=nid, x=i * dx, y=j * dy)
            s.add_node(n)
            at[(i, j)] = n
            nid += 1
    eid = 0
    for j in range(ny):
        for i in range(nx):
            s.add_element(ShellElement(
                id=eid,
                nodes=[at[(i, j)], at[(i + 1, j)],
                       at[(i + 1, j + 1)], at[(i, j + 1)]],
                E=E, nu=nu, t=t))
            eid += 1
    return s


def main() -> None:
    E, nu, t = 2.1e11, 0.3, 0.01
    ok = True

    # ---------- 1. Ejemplo de la figura 15: 4 elementos, 9 nodos ----------
    s = make_shell_mesh(2.0, 2.0, 2, 2, E, nu, t)
    n_dofs = s.n_dofs
    print(f"[1] Figura 15: {len(s.elements)} elementos, {len(s.nodes)} nodos, "
          f"matriz global {n_dofs}x{n_dofs} (deben ser 45x45)")
    ok &= (len(s.elements) == 4 and len(s.nodes) == 9 and n_dofs == 45)

    # ---------- 2. K^e: simetria, modos rigidos y desacoplamiento ----------
    el = s.elements[0]
    K = el.stiffness_matrix()
    sim = float(np.max(np.abs(K - K.T)))
    eig = np.linalg.eigvalsh(K)
    n_rigidos = int(np.sum(np.abs(eig) < np.max(eig) * 1e-10))
    idx_m = [5 * i + c for i in range(4) for c in (0, 1)]
    idx_b = [5 * i + c for i in range(4) for c in (2, 3, 4)]
    acople = float(np.max(np.abs(K[np.ix_(idx_m, idx_b)])))
    print(f"[2] K^e (20x20): asimetria = {sim:.3e}   modos rigidos = "
          f"{n_rigidos} (deben ser 6)   acople membrana-flexion = {acople:.3e}")
    ok &= (sim < 1e-4 * np.max(np.abs(K)) and n_rigidos == 6 and acople == 0.0)

    # ---------- 3a. Carga en el plano == modelo plane ----------
    a, nx = 2.0, 4
    sh = make_shell_mesh(a, 1.0, nx, 2, E, nu, t)
    pl = Structure()
    for n in sh.nodes:
        pl.add_node(Node(id=n.id, x=n.x, y=n.y))
    for e in sh.elements:
        pl.add_element(Q4Element(
            id=e.id, nodes=[pl.nodes[m.id] for m in e.nodes],
            E=E, nu=nu, t=t, plane_stress=True))
    for ns, np_ in zip(sh.nodes, pl.nodes):
        if abs(ns.x) < 1e-12:                       # borde izquierdo empotrado
            ns.restraint_u = ns.restraint_v = True
            ns.restraint_w = ns.restraint_rx = ns.restraint_ry = True
            np_.restraint_x = np_.restraint_y = True
        if abs(ns.x - a) < 1e-12:                   # traccion en el extremo
            ns.load_fx = 1000.0
            np_.load_x = 1000.0
    r_sh = solve_shell(sh)
    r_pl = solve(pl)
    err_plano, fuera = 0.0, 0.0
    for ns, np_ in zip(sh.nodes, pl.nodes):
        d = r_sh.displacements[list(ns.dofs)]
        err_plano = max(err_plano,
                        abs(d[0] - r_pl.displacements[np_.dofs[0]]),
                        abs(d[1] - r_pl.displacements[np_.dofs[1]]))
        fuera = max(fuera, float(np.max(np.abs(d[2:]))))
    ref = float(np.max(np.abs(r_pl.displacements)))
    print(f"[3a] Carga en el plano: diferencia vs modelo plane = "
          f"{err_plano:.3e} m (u_max = {ref:.3e});  respuesta fuera del "
          f"plano = {fuera:.3e}")
    ok &= (err_plano < 1e-14 * ref and fuera == 0.0)

    # ---------- 3b. Carga transversal == modelo plate ----------
    q = -1000.0
    sh2 = make_shell_mesh(1.0, 1.0, 4, 4, E, nu, t)
    pl2 = StructurePlate()
    for n in sh2.nodes:
        pl2.add_node(NodePlate(id=n.id, x=n.x, y=n.y))
    for e in sh2.elements:
        pl2.add_element(PlateElement(
            id=e.id, nodes=[pl2.nodes[m.id] for m in e.nodes], E=E, nu=nu, t=t))
    for ns, npl in zip(sh2.nodes, pl2.nodes):
        en_x = abs(ns.x) < 1e-12 or abs(ns.x - 1.0) < 1e-12
        en_y = abs(ns.y) < 1e-12 or abs(ns.y - 1.0) < 1e-12
        if en_x or en_y:
            ns.restraint_w = npl.restraint_w = True
        if en_x:
            ns.restraint_rx = npl.restraint_rx = True
        if en_y:
            ns.restraint_ry = npl.restraint_ry = True
        # el shell necesita fijar la membrana para que K_ff no sea singular
        ns.restraint_u = ns.restraint_v = True
    r_sh2 = solve_shell(sh2, q_uniform=q)
    r_pl2 = solve_plate(pl2, q_uniform=q)
    err_flex, en_plano = 0.0, 0.0
    for ns, npl in zip(sh2.nodes, pl2.nodes):
        d = r_sh2.displacements[list(ns.dofs)]
        dp = r_pl2.displacements[list(npl.dofs)]
        err_flex = max(err_flex, float(np.max(np.abs(d[2:] - dp))))
        en_plano = max(en_plano, abs(d[0]), abs(d[1]))
    ref2 = float(np.max(np.abs(r_pl2.displacements)))
    print(f"[3b] Carga transversal: diferencia vs modelo plate = "
          f"{err_flex:.3e} (w_max = {ref2:.3e});  respuesta en el plano = "
          f"{en_plano:.3e}")
    ok &= (err_flex < 1e-12 * ref2 and en_plano == 0.0)

    # ---------- 4. Benchmark de Timoshenko a traves del shell ----------
    D_flex = E * t ** 3 / (12.0 * (1.0 - nu * nu))
    w_exacto = 0.00406 * q * 1.0 ** 4 / D_flex
    errores = []
    for N in (4, 8, 12):
        sm = make_shell_mesh(1.0, 1.0, N, N, E, nu, t)
        for ns in sm.nodes:
            en_x = abs(ns.x) < 1e-12 or abs(ns.x - 1.0) < 1e-12
            en_y = abs(ns.y) < 1e-12 or abs(ns.y - 1.0) < 1e-12
            ns.restraint_u = ns.restraint_v = True
            if en_x or en_y:
                ns.restraint_w = True
            if en_x:
                ns.restraint_rx = True
            if en_y:
                ns.restraint_ry = True
        res = solve_shell(sm, q_uniform=q)
        centro = next(n for n in sm.nodes
                      if abs(n.x - 0.5) < 1e-9 and abs(n.y - 0.5) < 1e-9)
        w_fem = res.displacements[centro.dofs[2]]
        err = abs(w_fem - w_exacto) / abs(w_exacto) * 100.0
        errores.append(err)
        print(f"    Malla {N:2d}x{N:<2d}: w = {w_fem:.6e} m (err {err:5.2f} %)")
    print(f"[4] Convergencia del error en w: {errores[0]:.2f} % -> "
          f"{errores[1]:.2f} % -> {errores[2]:.2f} %")
    ok &= errores[2] < 1.0 and errores[2] <= errores[1] <= errores[0] + 1e-9

    print()
    if ok:
        print("RESULTADO: todos los checks del nucleo SHELL pasan.")
    else:
        print("RESULTADO: HAY CHECKS FALLIDOS - revisar arriba.")
        sys.exit(1)


if __name__ == "__main__":
    main()
