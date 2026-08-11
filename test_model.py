"""Test del MODELO UNIFICADO (frame + plane + plate + shell en un modelo).

La verificación clave: el solver unificado debe reproducir EXACTAMENTE los
resultados de los solvers separados ya validados. Si un modelo de puras
losas da lo mismo por las dos vías, el mapeo de GDL es correcto.

Checks:
  1. Secciones: FrameSection.rectangular calcula A, Iy, Iz, J bien.
  2. Modelo de puras PLACAS  == solver_plate (placa SS con carga uniforme).
  3. Modelo de puras LÁMINAS == solver_shell.
  4. Modelo de puro PÓRTICO  == solver_frame (voladizo con carga en punta).
  5. Modelo MIXTO (pórtico + losa) resuelve, comparte nodos, y los GDL
     inactivos quedan restringidos sin volver singular el sistema.
  6. Equilibrio global: suma de reacciones = suma de cargas aplicadas.

Ejecutar:  .\\venv\\Scripts\\python.exe test_model.py
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, ".")

from src.fem.materials import get_material
from src.fem.sections import (AreaSection, FrameSection, SectionLibrary,
                              default_library)
from src.fem.model import Model
from src.fem.solver_model import solve_model

# Núcleos separados, para comparar
from src.fem.node_plate import NodePlate
from src.fem.plate_element import PlateElement
from src.fem.structure_plate import StructurePlate
from src.fem.solver_plate import solve_plate
from src.fem.node_shell import NodeShell
from src.fem.shell_element import ShellElement
from src.fem.structure_shell import StructureShell
from src.fem.solver_shell import solve_shell
from src.fem.node_frame import NodeFrame
from src.fem.frame_element import FrameElement
from src.fem.structure_frame import StructureFrame
from src.fem.solver_frame import solve_frame


E, NU, T = 2.1e11, 0.3, 0.01
A_LADO, Q = 1.0, -1000.0
N = 4


def _lib() -> SectionLibrary:
    lib = SectionLibrary()
    mat = get_material("Acero estructural")
    # Material con E y nu exactos para que la comparación sea 1:1
    mat = type(mat)(name="test", E=E, nu=NU, density=mat.density,
                    description="", color=mat.color)
    lib.add(AreaSection("LOSA", mat, t=T, tipo="plate"))
    lib.add(AreaSection("LAMINA", mat, t=T, tipo="shell"))
    lib.add(AreaSection("MURO", mat, t=T, tipo="plane"))
    lib.add(FrameSection.rectangular("VIGA", mat, 0.3, 0.5))
    return lib


def _grid_model(tipo: str, seccion: str, n: int = N) -> Model:
    """Modelo unificado: malla n×n de elementos de área, apoyo SS, carga q."""
    mod = Model(sections=_lib())
    d = A_LADO / n
    ids = {}
    for j in range(n + 1):
        for i in range(n + 1):
            nd = mod.add_node(i * d, j * d, 0.0)
            ids[(i, j)] = nd.id
    for j in range(n):
        for i in range(n):
            mod.add_member(tipo,
                           [ids[(i, j)], ids[(i + 1, j)],
                            ids[(i + 1, j + 1)], ids[(i, j + 1)]],
                           seccion, q=Q)
    # Apoyo simplemente apoyado 'duro' en los 4 bordes
    tol = 1e-9
    for nd in mod.nodes:
        bx = abs(nd.x) < tol or abs(nd.x - A_LADO) < tol
        by = abs(nd.y) < tol or abs(nd.y - A_LADO) < tol
        if bx or by:
            nd.restraints[2] = True          # w
            if bx:
                nd.restraints[3] = True      # θx
            if by:
                nd.restraints[4] = True      # θy
    return mod


def main() -> None:
    ok = True

    # ---------- 1. Secciones ----------
    lib = default_library()
    viga = lib.get("VIGA 30x50")
    A_ok = abs(viga.A - 0.15) < 1e-12
    Iz_ok = abs(viga.Iz - 0.30 * 0.50 ** 3 / 12) < 1e-15
    print(f"[1] Secciones: {len(lib)} en biblioteca; VIGA 30x50 "
          f"A={viga.A:.4f} m2, Iz={viga.Iz:.6e} m4, J={viga.J:.6e} m4")
    ok &= A_ok and Iz_ok and len(lib) >= 6

    # ---------- 2. Placas: unificado vs solver_plate ----------
    mod = _grid_model("plate", "LOSA")
    res_u = solve_model(mod)
    s = StructurePlate()
    d = A_LADO / N
    at = {}
    nid = 0
    for j in range(N + 1):
        for i in range(N + 1):
            nd = NodePlate(id=nid, x=i * d, y=j * d)
            s.add_node(nd)
            at[(i, j)] = nd
            nid += 1
    for j in range(N):
        for i in range(N):
            s.add_element(PlateElement(
                id=len(s.elements),
                nodes=[at[(i, j)], at[(i + 1, j)],
                       at[(i + 1, j + 1)], at[(i, j + 1)]],
                E=E, nu=NU, t=T))
    tol = 1e-9
    for nd in s.nodes:
        bx = abs(nd.x) < tol or abs(nd.x - A_LADO) < tol
        by = abs(nd.y) < tol or abs(nd.y - A_LADO) < tol
        if bx or by:
            nd.restraint_w = True
            if bx:
                nd.restraint_rx = True
            if by:
                nd.restraint_ry = True
    res_s = solve_plate(s, q_uniform=Q)
    # w del nodo central por ambas vías
    c_u = next(n for n in mod.nodes
               if abs(n.x - 0.5) < tol and abs(n.y - 0.5) < tol)
    c_s = next(n for n in s.nodes
               if abs(n.x - 0.5) < tol and abs(n.y - 0.5) < tol)
    w_u = res_u.displacements[c_u.dofs[2]]
    w_s = res_s.displacements[c_s.dofs[0]]
    err2 = abs(w_u - w_s) / abs(w_s)
    print(f"[2] PLATE unificado vs separado: w = {w_u:.9e} / {w_s:.9e} "
          f"-> error relativo {err2:.2e}")
    ok &= err2 < 1e-12

    # ---------- 3. Láminas: unificado vs solver_shell ----------
    mod3 = _grid_model("shell", "LAMINA")
    res_u3 = solve_model(mod3)
    ss = StructureShell()
    at = {}
    nid = 0
    for j in range(N + 1):
        for i in range(N + 1):
            nd = NodeShell(id=nid, x=i * d, y=j * d)
            ss.add_node(nd)
            at[(i, j)] = nd
            nid += 1
    for j in range(N):
        for i in range(N):
            ss.add_element(ShellElement(
                id=len(ss.elements),
                nodes=[at[(i, j)], at[(i + 1, j)],
                       at[(i + 1, j + 1)], at[(i, j + 1)]],
                E=E, nu=NU, t=T))
    for nd in ss.nodes:
        bx = abs(nd.x) < tol or abs(nd.x - A_LADO) < tol
        by = abs(nd.y) < tol or abs(nd.y - A_LADO) < tol
        if bx or by:
            nd.restraint_w = True
            nd.restraint_u = True
            nd.restraint_v = True
            if bx:
                nd.restraint_rx = True
            if by:
                nd.restraint_ry = True
    # el unificado necesita las mismas restricciones de membrana
    for nd in mod3.nodes:
        bx = abs(nd.x) < tol or abs(nd.x - A_LADO) < tol
        by = abs(nd.y) < tol or abs(nd.y - A_LADO) < tol
        if bx or by:
            nd.restraints[0] = True
            nd.restraints[1] = True
    res_u3 = solve_model(mod3)
    res_s3 = solve_shell(ss, q_uniform=Q)
    c_u3 = next(n for n in mod3.nodes
                if abs(n.x - 0.5) < tol and abs(n.y - 0.5) < tol)
    c_s3 = next(n for n in ss.nodes
                if abs(n.x - 0.5) < tol and abs(n.y - 0.5) < tol)
    w_u3 = res_u3.displacements[c_u3.dofs[2]]
    w_s3 = res_s3.displacements[c_s3.dofs[2]]
    err3 = abs(w_u3 - w_s3) / abs(w_s3)
    print(f"[3] SHELL unificado vs separado: w = {w_u3:.9e} / {w_s3:.9e} "
          f"-> error relativo {err3:.2e}")
    ok &= err3 < 1e-12

    # ---------- 4. Pórtico: unificado vs solver_frame ----------
    L, P = 3.0, -5000.0
    mod4 = Model(sections=_lib())
    n0 = mod4.add_node(0.0, 0.0, 0.0)
    n1 = mod4.add_node(L, 0.0, 0.0)
    mod4.add_member("frame", [n0.id, n1.id], "VIGA")
    mod4.node(n0.id).fix_all()
    mod4.node(n1.id).loads[2] = P          # Fz en la punta
    res_u4 = solve_model(mod4)
    sf = StructureFrame()
    f0 = NodeFrame(id=0, x=0.0, y=0.0, z=0.0)
    f1 = NodeFrame(id=1, x=L, y=0.0, z=0.0)
    for a in ("restraint_ux", "restraint_uy", "restraint_uz",
              "restraint_rx", "restraint_ry", "restraint_rz"):
        setattr(f0, a, True)
    f1.load_fz = P
    sf.add_node(f0)
    sf.add_node(f1)
    sec = _lib().get("VIGA")
    sf.add_element(FrameElement(id=0, nodes=[f0, f1], E=sec.E, G=sec.G,
                                A=sec.A, Iy=sec.Iy, Iz=sec.Iz, J=sec.J))
    res_s4 = solve_frame(sf)
    wz_u = res_u4.displacements[mod4.node(n1.id).dofs[2]]
    wz_s = res_s4.displacements[f1.dofs[2]]
    # Solución analítica del voladizo: P·L³/(3·E·I)
    wz_teo = P * L ** 3 / (3.0 * sec.E * sec.Iy)
    err4 = abs(wz_u - wz_s) / abs(wz_s)
    err4t = abs(wz_u - wz_teo) / abs(wz_teo)
    print(f"[4] FRAME unificado vs separado: w = {wz_u:.9e} / {wz_s:.9e} "
          f"-> error {err4:.2e}   (vs analítico P·L³/3EI: {err4t:.2e})")
    ok &= err4 < 1e-12 and err4t < 1e-9

    # ---------- 5. Modelo MIXTO: pórtico + losa compartiendo nodos ----------
    mixto = Model(sections=_lib())
    # Losa 2x2 en z = 0
    ids = {}
    for j in range(3):
        for i in range(3):
            nd = mixto.add_node(i * 0.5, j * 0.5, 0.0)
            ids[(i, j)] = nd.id
    for j in range(2):
        for i in range(2):
            mixto.add_member("shell",
                             [ids[(i, j)], ids[(i + 1, j)],
                              ids[(i + 1, j + 1)], ids[(i, j + 1)]],
                             "LAMINA", q=Q)
    # Columnas bajo las 4 esquinas de la losa
    base_ids = []
    for (i, j) in [(0, 0), (2, 0), (2, 2), (0, 2)]:
        nb = mixto.add_node(i * 0.5, j * 0.5, -3.0)
        base_ids.append(nb.id)
        mixto.add_member("frame", [nb.id, ids[(i, j)]], "VIGA")
    mixto.assign_support(base_ids, "empotrado")
    res_m = solve_model(mixto)
    n_libres = len(mixto.free_dofs())
    w_losa = res_m.displacements[mixto.node(ids[(1, 1)]).dofs[2]]
    print(f"[5] MIXTO (4 shell + 4 frame): {mixto.describe()}")
    print(f"    w centro de losa = {w_losa:.6e} m")
    ok &= n_libres > 0 and np.isfinite(w_losa) and w_losa < 0

    # ---------- 6. Equilibrio global del modelo mixto ----------
    carga_total_z = float(np.sum(res_m.F_global[2::6]))
    reaccion_total_z = float(np.sum(res_m.reactions[2::6]))
    desbalance = abs(carga_total_z + reaccion_total_z)
    rel = desbalance / max(abs(carga_total_z), 1.0)
    print(f"[6] Equilibrio en Z: carga {carga_total_z:.4f} N, "
          f"reacción {reaccion_total_z:.4f} N -> desbalance relativo {rel:.2e}")
    ok &= rel < 1e-9

    print()
    if ok:
        print("RESULTADO: todos los checks del MODELO UNIFICADO pasan.")
    else:
        print("RESULTADO: HAY CHECKS FALLIDOS — revisar arriba.")
        sys.exit(1)


if __name__ == "__main__":
    main()
