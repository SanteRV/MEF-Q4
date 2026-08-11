"""Test de la VENTANA DE MODELADO unificada (Corrección 2, pasos 2 a 5, 9).

Verifica el flujo real de trabajo simulando los clics del usuario sobre el
lienzo 3D: definir grilla, dibujar en 3D, asignar apoyos en lote, cargar,
resolver y leer las tablas de resultados.

Lo más delicado que se prueba aquí es la conversión clic 2D -> punto 3D:
se proyecta un punto conocido a pantalla y se comprueba que al "clicar"
ahí se recupera exactamente ese punto (ida y vuelta), incluido el ajuste
a la grilla medido en píxeles.

Ejecutar:  .\\venv\\Scripts\\python.exe test_model_ui.py
"""
from __future__ import annotations

import sys

import numpy as np
from PySide6.QtCore import QPointF
from PySide6.QtWidgets import QApplication

sys.path.insert(0, ".")

from src.fem.grid import GridSystem
from src.ui.theme import apply_theme


def main() -> None:
    app = QApplication.instance() or QApplication(sys.argv)
    apply_theme(app)
    from src.ui.model_mode import ModelModeWidget
    from src.ui.model_canvas_3d import Mode

    ok = True
    w = ModelModeWidget()
    w.resize(1200, 800)
    w.show()
    app.processEvents()
    c = w.canvas

    # ---------- 1. Grilla ----------
    w.ed_gx.setText("0, 5, 10")
    w.ed_gy.setText("0, 5")
    w.ed_gz.setText("0, 3")
    w._apply_grid()
    app.processEvents()
    g = w.grid
    print(f"[1] Grilla: {g.describe()}")
    ok &= g.x == [0.0, 5.0, 10.0] and g.y == [0.0, 5.0] and g.z == [0.0, 3.0]
    ok &= len(g.intersections()) == 12
    ok &= w.cmb_plano.count() == 7      # 2 planos XY + 2 XZ + 3 YZ

    # ---------- 2. Ida y vuelta del clic ----------
    c.set_work_plane("xy", 0.0)
    c.set_view("3d")
    c.fit_view()
    app.processEvents()
    err = 0.0
    for p3 in [(0, 0, 0), (5, 0, 0), (10, 5, 0), (5, 5, 0)]:
        s = c._project_point(*p3)
        b = c._point_on_work_plane(QPointF(*s))
        err = max(err, max(abs(b[i] - p3[i]) for i in range(3)))
    print(f"[2] Clic 2D -> punto 3D: error máximo {err:.2e} m")
    ok &= err < 1e-9

    # El ajuste se mide en píxeles: cerca engancha, lejos no
    s = c._project_point(5.0, 5.0, 0.0)
    cerca = c._point_on_work_plane(QPointF(s[0] + 9, s[1] + 9))
    lejos = c._point_on_work_plane(QPointF(s[0] + 40, s[1] + 40))
    print(f"    Ajuste a 12.7 px -> {tuple(round(v, 6) for v in cerca)}; "
          f"a 56 px (libre) -> ({lejos[0]:.3f}, {lejos[1]:.3f}, {lejos[2]:.3f})")
    ok &= abs(cerca[0] - 5.0) < 1e-9 and abs(cerca[1] - 5.0) < 1e-9
    ok &= abs(lejos[0] - 5.0) > 1e-6

    # ---------- 3. Dibujar una losa (4 clics) ----------
    w.cmb_tipo.setCurrentText("shell")
    app.processEvents()
    c.set_mode(Mode.AREA)
    for px, py in [(0, 0), (5, 0), (5, 5), (0, 5)]:
        c._click(QPointF(*c._project_point(px, py, 0.0)), False)
    for px, py in [(5, 0), (10, 0), (10, 5), (5, 5)]:
        c._click(QPointF(*c._project_point(px, py, 0.0)), False)
    app.processEvents()
    n_shell = len(w.model.members_of_type("shell"))
    print(f"[3] Losa dibujada con clics: {len(w.model.nodes)} nodos, "
          f"{n_shell} elementos shell (los 2 comparten arista)")
    ok &= n_shell == 2 and len(w.model.nodes) == 6

    # ---------- 4. Dibujar columnas en un plano vertical ----------
    w.cmb_tipo.setCurrentText("frame")
    app.processEvents()
    c.set_mode(Mode.FRAME)
    # Columna bajo (0,0): del nivel Z=0 al Z=3, dibujada en el plano YZ X=0
    c.set_work_plane("yz", 0.0)
    for py, pz in [(0, 0), (0, 3)]:
        c._click(QPointF(*c._project_point(0.0, py, pz)), False)
    # Columna bajo (10,0), en el plano YZ X=10
    c.set_work_plane("yz", 10.0)
    for py, pz in [(0, 0), (0, 3)]:
        c._click(QPointF(*c._project_point(10.0, py, pz)), False)
    app.processEvents()
    n_frame = len(w.model.members_of_type("frame"))
    print(f"[4] Columnas dibujadas en planos verticales: {n_frame} frames. "
          f"{w.model.describe()}")
    ok &= n_frame == 2

    # ---------- 5. Apoyos en lote ----------
    c.set_mode(Mode.SELECT)
    c.set_work_plane("xy", 3.0)
    # Los nodos superiores de las columnas están en Z = 3
    altos = [n.id for n in w.model.nodes if abs(n.z - 3.0) < 1e-9]
    c.selected = list(altos)
    w._assign_support("empotrado")
    app.processEvents()
    n_emp = sum(1 for n in w.model.nodes if all(n.restraints))
    print(f"[5] Apoyos en lote: {len(altos)} nodos seleccionados -> "
          f"{n_emp} empotrados")
    ok &= n_emp == len(altos) == 2

    # ---------- 6. Cargas y presión ----------
    c.selected = [n.id for n in w.model.nodes
                  if abs(n.z) < 1e-9 and abs(n.x - 5.0) < 1e-9]
    w.cmb_gdl.setCurrentIndex(2)            # w (Fz)
    w.sp_valor.setValue(-2000.0)
    w._assign_load()
    w.sp_q.setValue(-500.0)
    w._assign_pressure()
    app.processEvents()
    cargados = sum(1 for n in w.model.nodes if abs(n.loads[2]) > 0)
    con_q = sum(1 for m in w.model.members if getattr(m, "q", 0.0) != 0.0)
    print(f"[6] Cargas: {cargados} nodo(s) con Fz, {con_q} área(s) con q")
    ok &= cargados >= 1 and con_q == 2

    # ---------- 7. Resolver y leer tablas ----------
    w.solve()
    app.processEvents()
    r = w.result
    if r is None:
        print("[7] FALLO: el modelo no se resolvió")
        ok = False
    else:
        u_max = float(np.max(np.abs(r.displacements)))
        # Equilibrio global en Z
        carga_z = float(np.sum(r.F_global[2::6]))
        reac_z = float(np.sum(r.reactions[2::6]))
        desb = abs(carga_z + reac_z) / max(abs(carga_z), 1.0)
        filas_u = w.tbl_nodos.model().rowCount()
        filas_r = w.tbl_reac.model().rowCount()
        filas_e = w.tbl_elem.model().rowCount()
        print(f"[7] Resuelto: u_max = {u_max:.6e} m; "
              f"equilibrio Z: carga {carga_z:.2f} N, reacción {reac_z:.2f} N "
              f"(desbalance {desb:.2e})")
        print(f"    Tablas: {filas_u} desplazamientos, {filas_r} reacciones, "
              f"{filas_e} elementos")
        ok &= np.isfinite(u_max) and u_max > 0
        ok &= desb < 1e-9
        ok &= filas_u == len(w.model.nodes) and filas_r == 2 and filas_e == 4

    # ---------- 8. Vistas y cancelación ----------
    for preset in ("3d", "xy", "xz", "yz"):
        c.set_view(preset)
    c.set_mode(Mode.AREA)
    c._click(QPointF(*c._project_point(0.0, 0.0, 0.0)), False)
    n_pend = len(c._pending)
    c.keyPressEvent(type("E", (), {"key": lambda s=None: 0x01000000,
                                   "modifiers": lambda s=None: None})())
    print(f"[8] Vistas 3D/XY/XZ/YZ aplicadas; elemento en curso: "
          f"{n_pend} nodo(s) pendientes antes de cancelar")
    ok &= n_pend == 1

    print()
    if ok:
        print("RESULTADO: todos los checks de la VENTANA DE MODELADO pasan.")
    else:
        print("RESULTADO: HAY CHECKS FALLIDOS — revisar arriba.")
        sys.exit(1)


if __name__ == "__main__":
    main()
