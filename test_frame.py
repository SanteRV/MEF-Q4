"""Test del nucleo FRAME 3D (elemento de 12 GDL, cap. 01.02).

Verificaciones contra solucion analitica conocida:
  1. k local: simetria y 6 modos rigidos (una barra libre en el espacio
     tiene los 6 movimientos rigidos del solido).
  2. Matriz de transformacion T (ec. 2.3.1): r ortonormal, det(r) = +1, y
     K = T^t k T invariante al girar la barra en el espacio.
  3. Voladizo con carga puntual en el extremo:  d = P L^3/(3EI),
     giro = P L^2/(2EI).
  4. Voladizo con carga distribuida:  d = w L^4/(8EI),  giro = w L^3/(6EI),
     y el momento de empotramiento debe ser w L^2/2.
  5. Viga biempotrada con carga distribuida: momentos de extremo w L^2/12
     y flecha central w L^4/(384 EI).
  6. Torsion pura: giro = M_T L /(GJ).
  7. Portico plano: equilibrio global de reacciones.

Ejecutar:  .\\venv\\Scripts\\python.exe test_frame.py
"""
from __future__ import annotations
import sys
import numpy as np

sys.path.insert(0, ".")

from src.fem.frame_element import FrameElement, FrameLoad
from src.fem.node_frame import NodeFrame
from src.fem.solver_frame import solve_frame
from src.fem.structure_frame import StructureFrame


E, G = 2.1e11, 8.077e10
A, Iy, Iz, J = 0.01, 8.333e-6, 8.333e-6, 1.6e-5


def barra(L, *, ejes=(0.0, 0.0, 0.0), loads=None, psi=0.0):
    """Una barra de longitud L desde el origen en la direccion 'ejes'."""
    d = np.array(ejes, dtype=float)
    d = d / np.linalg.norm(d) * L
    s = StructureFrame()
    s.add_node(NodeFrame(0, 0.0, 0.0, 0.0))
    s.add_node(NodeFrame(1, float(d[0]), float(d[1]), float(d[2])))
    s.add_element(FrameElement(
        0, [s.nodes[0], s.nodes[1]], E=E, G=G, A=A, Iy=Iy, Iz=Iz, J=J,
        psi=psi, loads=list(loads or [])))
    return s


def main() -> None:
    ok = True
    L = 3.0
    tol = 1e-9

    # ---------- 1. k local: simetria y modos rigidos ----------
    s = barra(L, ejes=(1, 0, 0))
    el = s.elements[0]
    k = el.stiffness_local()
    sim = float(np.max(np.abs(k - k.T)))
    eig = np.linalg.eigvalsh(k)
    n_rig = int(np.sum(np.abs(eig) < np.max(eig) * 1e-10))
    print(f"[1] k local (12x12): asimetria = {sim:.3e}   "
          f"modos rigidos = {n_rig} (deben ser 6)")
    ok &= sim < 1e-6 * np.max(np.abs(k)) and n_rig == 6

    # ---------- 2. Transformacion: ortonormalidad e invariancia ----------
    errores_r, autoval = [], []
    for ejes in [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (2, -1, 3)]:
        e = barra(L, ejes=ejes).elements[0]
        r = e.rotation_matrix()
        errores_r.append(float(np.max(np.abs(r @ r.T - np.eye(3)))))
        errores_r.append(abs(float(np.linalg.det(r)) - 1.0))
        autoval.append(np.sort(np.linalg.eigvalsh(e.stiffness_matrix())))
    err_r = max(errores_r)
    # Los autovalores de K no dependen de la orientacion de la barra
    err_inv = max(float(np.max(np.abs(a - autoval[0]))) for a in autoval)
    ref_k = float(np.max(np.abs(autoval[0])))
    print(f"[2] Matriz r: error de ortonormalidad y det = {err_r:.3e};  "
          f"invariancia de los autovalores de K = {err_inv:.3e}")
    ok &= err_r < 1e-12 and err_inv < 1e-6 * ref_k

    # ---------- 3. Voladizo con carga puntual en el extremo ----------
    P = -5000.0
    s = barra(L, ejes=(1, 0, 0))
    s.nodes[0].empotrar()
    s.nodes[1].restraint_uz = s.nodes[1].restraint_rx = True
    s.nodes[1].restraint_ry = True
    s.nodes[1].load_fy = P
    r3 = solve_frame(s)
    d_fem = r3.displacements[s.nodes[1].dofs[1]]
    g_fem = r3.displacements[s.nodes[1].dofs[5]]
    d_ex = P * L ** 3 / (3.0 * E * Iz)
    g_ex = P * L ** 2 / (2.0 * E * Iz)
    e_d = abs(d_fem - d_ex) / abs(d_ex)
    e_g = abs(g_fem - g_ex) / abs(g_ex)
    print(f"[3] Voladizo + P: d = {d_fem:.9e} m (exacto {d_ex:.9e}, "
          f"err {e_d:.2e});  giro err {e_g:.2e}")
    ok &= e_d < tol and e_g < tol

    # ---------- 4. Voladizo con carga distribuida ----------
    w = -2000.0
    s = barra(L, ejes=(1, 0, 0),
              loads=[FrameLoad("distribuida", w, eje="y")])
    s.nodes[0].empotrar()
    s.nodes[1].restraint_uz = s.nodes[1].restraint_rx = True
    s.nodes[1].restraint_ry = True
    r4 = solve_frame(s)
    d_fem = r4.displacements[s.nodes[1].dofs[1]]
    g_fem = r4.displacements[s.nodes[1].dofs[5]]
    d_ex = w * L ** 4 / (8.0 * E * Iz)
    g_ex = w * L ** 3 / (6.0 * E * Iz)
    M_emp = r4.elements[0].end_forces[5]
    M_ex = w * L ** 2 / 2.0
    e_d = abs(d_fem - d_ex) / abs(d_ex)
    e_g = abs(g_fem - g_ex) / abs(g_ex)
    e_m = abs(abs(M_emp) - abs(M_ex)) / abs(M_ex)
    print(f"[4] Voladizo + w: d err {e_d:.2e};  giro err {e_g:.2e};  "
          f"momento de empotramiento {M_emp:.4f} (exacto {M_ex:.4f}, "
          f"err {e_m:.2e})")
    ok &= e_d < tol and e_g < tol and e_m < tol

    # ---------- 5. Viga biempotrada con carga distribuida ----------
    s = StructureFrame()
    for i in range(3):
        s.add_node(NodeFrame(i, i * L / 2.0, 0.0, 0.0))
    for i in range(2):
        s.add_element(FrameElement(
            i, [s.nodes[i], s.nodes[i + 1]], E=E, G=G, A=A, Iy=Iy, Iz=Iz, J=J,
            loads=[FrameLoad("distribuida", w, eje="y")]))
    s.nodes[0].empotrar()
    s.nodes[2].empotrar()
    s.nodes[1].restraint_uz = s.nodes[1].restraint_rx = True
    s.nodes[1].restraint_ry = True
    r5 = solve_frame(s)
    d_centro = r5.displacements[s.nodes[1].dofs[1]]
    d_ex = w * L ** 4 / (384.0 * E * Iz)
    M_emp = r5.elements[0].end_forces[5]
    M_ex = w * L ** 2 / 12.0
    e_d = abs(d_centro - d_ex) / abs(d_ex)
    e_m = abs(abs(M_emp) - abs(M_ex)) / abs(M_ex)
    print(f"[5] Biempotrada + w: flecha central err {e_d:.2e};  "
          f"momento de extremo {M_emp:.4f} (exacto {M_ex:.4f}, err {e_m:.2e})")
    ok &= e_d < 1e-8 and e_m < 1e-8

    # ---------- 6. Torsion pura ----------
    MT = 1500.0
    s = barra(L, ejes=(1, 0, 0))
    s.nodes[0].empotrar()
    for atr in ("restraint_uy", "restraint_uz", "restraint_ry",
                "restraint_rz", "restraint_ux"):
        setattr(s.nodes[1], atr, True)
    s.nodes[1].load_mx = MT
    r6 = solve_frame(s)
    phi = r6.displacements[s.nodes[1].dofs[3]]
    phi_ex = MT * L / (G * J)
    e_p = abs(phi - phi_ex) / abs(phi_ex)
    print(f"[6] Torsion pura: giro = {phi:.9e} rad (exacto {phi_ex:.9e}, "
          f"err {e_p:.2e})")
    ok &= e_p < tol

    # ---------- 7. Portico plano: equilibrio de reacciones ----------
    H, Lv = 3.0, 4.0
    s = StructureFrame()
    s.add_node(NodeFrame(0, 0.0, 0.0, 0.0))
    s.add_node(NodeFrame(1, 0.0, 0.0, H))
    s.add_node(NodeFrame(2, Lv, 0.0, H))
    s.add_node(NodeFrame(3, Lv, 0.0, 0.0))
    s.add_element(FrameElement(0, [s.nodes[0], s.nodes[1]], E=E, G=G, A=A,
                               Iy=Iy, Iz=Iz, J=J))
    s.add_element(FrameElement(1, [s.nodes[1], s.nodes[2]], E=E, G=G, A=A,
                               Iy=Iy, Iz=Iz, J=J,
                               loads=[FrameLoad("distribuida", w, eje="z")]))
    s.add_element(FrameElement(2, [s.nodes[2], s.nodes[3]], E=E, G=G, A=A,
                               Iy=Iy, Iz=Iz, J=J))
    s.nodes[0].empotrar()
    s.nodes[3].empotrar()
    # Portico contenido en el plano XZ: se bloquea lo que sale del plano
    for n in (s.nodes[1], s.nodes[2]):
        n.restraint_uy = n.restraint_rx = n.restraint_rz = True
    Fx = 8000.0
    s.nodes[1].load_fx = Fx
    r7 = solve_frame(s)
    Rx = sum(r7.reactions[n.dofs[0]] for n in s.nodes)
    Rz = sum(r7.reactions[n.dofs[2]] for n in s.nodes)
    carga_z = w * Lv          # la distribuida actua en el eje local z de la viga
    e_x = abs(Rx + Fx)
    e_z = abs(Rz + carga_z)
    print(f"[7] Portico: suma Rx = {Rx:.6f} (carga {Fx:.1f}, desbalance "
          f"{e_x:.2e});  suma Rz = {Rz:.6f} (carga {carga_z:.1f}, "
          f"desbalance {e_z:.2e})")
    ok &= e_x < 1e-6 * abs(Fx) and e_z < 1e-6 * abs(carga_z)

    print()
    if ok:
        print("RESULTADO: todos los checks del nucleo FRAME pasan.")
    else:
        print("RESULTADO: HAY CHECKS FALLIDOS - revisar arriba.")
        sys.exit(1)


if __name__ == "__main__":
    main()
