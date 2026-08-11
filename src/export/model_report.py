"""Reporte de análisis del modelo unificado — Corrección 2, puntos 7 y 9.

Para qué sirve: el usuario modela una estructura completa (frames, muros,
placas y losas mezclados) y necesita un documento que resuma qué modeló y
qué obtuvo. Este módulo genera ese reporte en PDF:

    1. Resumen del modelo y de los grados de libertad
    2. Materiales y secciones empleados
    3. Nodos: coordenadas, apoyos y cargas
    4. Elementos: tipo, nodos y sección
    5. Resultados: desplazamientos, reacciones y fuerzas/momentos
    6. Nota metodológica con los capítulos del documento teórico

El desarrollo paso a paso de cada formulación NO se repite aquí: vive en el
reporte del modo didáctico (src/export/pdf_export.py), que es donde tiene
sentido mostrar las matrices intermedias de un solo elemento.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

from ..fem.model import DOF_NAMES, Model
from ..fem.sections import AreaSection, FrameSection
from ..fem.solver_model import ModelResult

PRIMARY = colors.HexColor("#1F4E78")
_MAX_FILAS = 400          # corte de seguridad para modelos muy grandes


def _fmt(v) -> str:
    """Precisión completa (15 dígitos); ceros residuales como 0."""
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, float):
        if not np.isfinite(v):
            return str(v)
        if abs(v) < 1e-13:
            return "0"
        return f"{v:.15g}"
    return str(v)


def _tabla(headers: list[str], filas: list[list[str]],
           anchos: list[float] | None = None) -> Table:
    """Tabla con el estilo del aplicativo: encabezado navy, filas alternas."""
    data = [headers] + filas
    t = Table(data, hAlign="LEFT", colWidths=anchos, repeatRows=1)
    estilo = [
        ("BACKGROUND", (0, 0), (-1, 0), PRIMARY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Courier"),
        ("FONTSIZE", (0, 0), (-1, -1), 7.5),
        ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
        ("ALIGN", (0, 0), (-1, 0), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#BFCBD6")),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]
    for i in range(1, len(data)):
        if i % 2 == 0:
            estilo.append(
                ("BACKGROUND", (0, i), (-1, i), colors.HexColor("#F2F6FA")))
    t.setStyle(TableStyle(estilo))
    return t


def _apoyo_txt(restraints: list[bool]) -> str:
    """Describe el apoyo de un nodo de forma compacta."""
    if all(restraints):
        return "empotrado"
    if not any(restraints):
        return "libre"
    return "+".join(DOF_NAMES[i] for i, r in enumerate(restraints) if r)


def export_model_report(model: Model, result: ModelResult | None,
                        path: Path) -> None:
    """Genera el reporte PDF del modelo unificado."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(path), pagesize=landscape(A4),
        leftMargin=1.4 * cm, rightMargin=1.4 * cm,
        topMargin=1.4 * cm, bottomMargin=1.4 * cm,
        title="Reporte de análisis estructural",
    )
    s = getSampleStyleSheet()
    H1 = ParagraphStyle("H1", parent=s["Title"], textColor=PRIMARY,
                        fontSize=18, spaceAfter=10)
    H2 = ParagraphStyle("H2", parent=s["Heading2"], textColor=PRIMARY,
                        fontSize=12, spaceBefore=14, spaceAfter=6)
    P = ParagraphStyle("P", parent=s["BodyText"], fontSize=9, leading=13)

    story: list = []
    story.append(Paragraph("Reporte de análisis estructural", H1))

    # ---------- 1. Resumen ----------
    story.append(Paragraph("1. Resumen del modelo", H2))
    por_tipo = {t: len(model.members_of_type(t))
                for t in ("frame", "plane", "plate", "shell")}
    filas = [
        ["Nodos", _fmt(len(model.nodes))],
        ["Elementos", _fmt(len(model.members))],
        ["  frame (barras)", _fmt(por_tipo["frame"])],
        ["  plane (muros)", _fmt(por_tipo["plane"])],
        ["  plate (placas)", _fmt(por_tipo["plate"])],
        ["  shell (láminas)", _fmt(por_tipo["shell"])],
        ["Grados de libertad totales", _fmt(model.n_dofs)],
        ["Grados de libertad libres", _fmt(len(model.free_dofs()))],
        ["Grados de libertad restringidos", _fmt(len(model.restrained_dofs()))],
    ]
    story.append(_tabla(["Concepto", "Valor"], filas, [9 * cm, 4 * cm]))

    # ---------- 2. Secciones ----------
    story.append(Paragraph("2. Materiales y secciones", H2))
    usadas = sorted({m.section for m in model.members})
    filas = []
    for nombre in usadas:
        sec = model.sections.get(nombre)
        tipo = getattr(sec, "tipo", "?")
        if isinstance(sec, FrameSection):
            props = (f"A={_fmt(sec.A)}  Iy={_fmt(sec.Iy)}  "
                     f"Iz={_fmt(sec.Iz)}  J={_fmt(sec.J)}")
        elif isinstance(sec, AreaSection):
            props = f"t={_fmt(sec.t)} m"
        else:
            props = "—"
        filas.append([nombre, tipo, sec.material.name,
                      _fmt(sec.E), _fmt(sec.nu), props])
    if filas:
        story.append(_tabla(
            ["Sección", "Tipo", "Material", "E (Pa)", "ν", "Propiedades"],
            filas))
    else:
        story.append(Paragraph("El modelo no tiene elementos.", P))

    # ---------- 3. Nodos ----------
    story.append(Paragraph("3. Nodos: coordenadas, apoyos y cargas", H2))
    filas = []
    for n in model.nodes[:_MAX_FILAS]:
        cargas = ", ".join(f"{DOF_NAMES[i]}={_fmt(v)}"
                           for i, v in enumerate(n.loads) if v != 0.0)
        filas.append([f"N{n.id + 1}", _fmt(n.x), _fmt(n.y), _fmt(n.z),
                      _apoyo_txt(n.restraints), cargas or "—"])
    story.append(_tabla(["Nodo", "x (m)", "y (m)", "z (m)", "Apoyo", "Cargas"],
                        filas))
    if len(model.nodes) > _MAX_FILAS:
        story.append(Paragraph(
            f"Se listan los primeros {_MAX_FILAS} nodos de "
            f"{len(model.nodes)}.", P))

    # ---------- 4. Elementos ----------
    story.append(Paragraph("4. Elementos", H2))
    filas = []
    for mem in model.members[:_MAX_FILAS]:
        nodos = " - ".join(f"N{i + 1}" for i in mem.node_ids)
        extra = f"q={_fmt(mem.q)} N/m²" if getattr(mem, "q", 0.0) else "—"
        filas.append([f"E{mem.id + 1}", mem.tipo, nodos, mem.section, extra])
    story.append(_tabla(["Elemento", "Tipo", "Nodos", "Sección", "Carga"],
                        filas))

    if result is None:
        story.append(Paragraph(
            "El modelo aún no ha sido resuelto: el reporte no incluye "
            "resultados.", P))
        doc.build(story)
        return

    # ---------- 5. Resultados ----------
    story.append(PageBreak())
    story.append(Paragraph("5. Desplazamientos nodales", H2))
    filas = []
    for n in model.nodes[:_MAX_FILAS]:
        filas.append([f"N{n.id + 1}"] +
                     [_fmt(result.displacements[g]) for g in n.dofs])
    story.append(_tabla(["Nodo"] + [f"{d}" for d in DOF_NAMES], filas))

    story.append(Paragraph("6. Reacciones en los apoyos", H2))
    activo = result.active_dofs
    filas = []
    for n in model.nodes:
        if not any(n.restraints):
            continue
        vals = []
        for k, g in enumerate(n.dofs):
            ok = n.restraints[k] and (activo is None or activo[g])
            vals.append(_fmt(result.reactions[g]) if ok else "—")
        filas.append([f"N{n.id + 1}"] + vals)
    if filas:
        story.append(_tabla(
            ["Nodo", "Fx (N)", "Fy (N)", "Fz (N)",
             "Mx (N·m)", "My (N·m)", "Mz (N·m)"], filas))
        # Verificación de equilibrio: la suma de reacciones debe compensar
        # la suma de cargas aplicadas en cada dirección.
        story.append(Spacer(1, 0.3 * cm))
        eq = []
        for k, nombre in enumerate(("Fx", "Fy", "Fz")):
            carga = float(np.sum(result.F_global[k::6]))
            reac = float(np.sum(result.reactions[k::6]))
            eq.append([nombre, _fmt(carga), _fmt(reac), _fmt(carga + reac)])
        story.append(_tabla(
            ["Dirección", "Σ cargas (N)", "Σ reacciones (N)", "Desbalance"],
            eq, [4 * cm, 5 * cm, 5 * cm, 5 * cm]))
    else:
        story.append(Paragraph("El modelo no tiene apoyos definidos.", P))

    story.append(Paragraph("7. Fuerzas y esfuerzos por elemento", H2))
    filas = []
    for mr in result.members[:_MAX_FILAS]:
        if mr.tipo == "frame" and mr.end_forces is not None:
            f = mr.end_forces
            filas.append([f"E{mr.member_id + 1}", "frame",
                          _fmt(float(f[0])), _fmt(float(f[1])),
                          _fmt(float(f[2])), _fmt(float(f[4]))])
        elif mr.moments:
            m = mr.moments[0]
            w = _fmt(mr.w_center) if mr.w_center is not None else "—"
            filas.append([f"E{mr.member_id + 1}", mr.tipo,
                          _fmt(float(m[0])), _fmt(float(m[1])),
                          _fmt(float(m[2])), w])
        elif mr.stresses:
            sg = mr.stresses[0]
            filas.append([f"E{mr.member_id + 1}", mr.tipo,
                          _fmt(float(sg[0])), _fmt(float(sg[1])),
                          _fmt(float(sg[2])), "—"])
    story.append(_tabla(
        ["Elemento", "Tipo", "(1) N / Mx / σx", "(2) Vy / My / σy",
         "(3) Vz / Mxy / τxy", "(4) M / w"], filas))
    story.append(Paragraph(
        "En elementos frame las columnas son N, Vy, Vz y M en el nodo "
        "inicial; en plate y shell son los momentos Mx, My, Mxy y el "
        "desplazamiento w en el centro; en plane son los esfuerzos "
        "σx, σy y τxy.", P))

    # ---------- 6. Nota metodológica ----------
    story.append(Paragraph("8. Nota metodológica", H2))
    story.append(Paragraph(
        "El análisis resuelve un único sistema K·U = F que reúne los cuatro "
        "tipos de elemento. Cada elemento aporta su matriz de rigidez según "
        "la formulación del documento teórico: elemento plane isoparamétrico "
        "de 4 nodos (cap. 01.01.02), placa rectangular de 12 GDL con teoría "
        "de Kirchhoff (cap. 01.01.03), lámina plana de 20 GDL por "
        "superposición de membrana y flexión (cap. 01.01.04) y barra "
        "tridimensional de 12 GDL (cap. 01.02). El sistema se particiona "
        "según las condiciones de borde y se resuelve el bloque libre; las "
        "reacciones se recuperan a posteriori. Los grados de libertad que "
        "ningún elemento activa se restringen automáticamente para evitar "
        "la singularidad de la matriz.", P))

    doc.build(story)
