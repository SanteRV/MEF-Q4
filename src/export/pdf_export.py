"""Exportador a PDF para Q4: reporte paso a paso con tablas e imágenes."""
from __future__ import annotations
from pathlib import Path
import tempfile

import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Polygon
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
)

from ..fem.steps import Procedure
from ..fem.structure import Structure
from ..fem.solver import FEMResult
from ..fem.q4_element import GAUSS_2X2, shape_functions


def _char_size(structure: Structure) -> float:
    xs = [n.x for n in structure.nodes]
    ys = [n.y for n in structure.nodes]
    if not xs:
        return 1.0
    return max(max(xs) - min(xs), max(ys) - min(ys)) or 1.0


def _draw_element(ax, el, **kw):
    xy = [(n.x, n.y) for n in el.nodes]
    poly = Polygon(xy, closed=True, **kw)
    ax.add_patch(poly)


def _figure_for_plot(plot_key, structure: Structure,
                     result: FEMResult | None) -> Figure | None:
    fig = Figure(figsize=(6, 4), tight_layout=True)
    ax = fig.add_subplot(111)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    size = _char_size(structure)
    margin = size * 0.2
    xs = [n.x for n in structure.nodes]
    ys = [n.y for n in structure.nodes]

    if plot_key == "mesh":
        ax.set_title("Geometría del elemento Q4")
        for el in structure.elements:
            _draw_element(ax, el, facecolor="#cfe2f3",
                          edgecolor="#1f77b4", alpha=0.6, lw=2)
        for n in structure.nodes:
            ax.plot(n.x, n.y, "o", color="black", markersize=6)
            ax.annotate(f"N{n.id + 1}", (n.x, n.y),
                        textcoords="offset points", xytext=(8, 8), fontsize=9)
    elif plot_key == "bc":
        ax.set_title("Apoyos y cargas")
        for el in structure.elements:
            _draw_element(ax, el, facecolor="#eeeeee",
                          edgecolor="#aaaaaa", alpha=0.4, lw=1)
        for n in structure.nodes:
            ax.plot(n.x, n.y, "o", color="black", markersize=5)
            ax.annotate(f"N{n.id + 1}", (n.x, n.y),
                        textcoords="offset points", xytext=(8, 8), fontsize=9)
            if n.restraint_x or n.restraint_y:
                ax.plot(n.x, n.y, "^", color="green", markersize=14,
                        markerfacecolor="none")
            if n.load_x or n.load_y:
                norm = np.hypot(n.load_x, n.load_y) or 1.0
                dx = (n.load_x / norm) * size * 0.3
                dy = (n.load_y / norm) * size * 0.3
                ax.annotate("", xy=(n.x + dx, n.y + dy), xytext=(n.x, n.y),
                            arrowprops=dict(arrowstyle="->", color="red", lw=2))
                ax.annotate(f"F=({n.load_x:.0f},{n.load_y:.0f})N",
                            (n.x + dx, n.y + dy), fontsize=8, color="red")
    elif plot_key == "deformed" and result is not None:
        max_disp = float(np.max(np.abs(result.displacements))) or 1.0
        scale = 0.15 * size / max_disp if max_disp > 0 else 1.0
        ax.set_title(f"Deformada (escala x{scale:.1f})")
        for el in structure.elements:
            _draw_element(ax, el, facecolor="none",
                          edgecolor="#aaa", lw=1)
            xy_def = []
            for n in el.nodes:
                ux = result.displacements[n.dofs[0]]
                uy = result.displacements[n.dofs[1]]
                xy_def.append((n.x + scale * ux, n.y + scale * uy))
            _draw_element_xy(ax, xy_def, facecolor="#f4cccc",
                             edgecolor="#d62728", alpha=0.6, lw=2)
    elif plot_key == "stress" and result is not None:
        ax.set_title("σx en puntos de Gauss")
        for el, el_res in zip(structure.elements, result.elements):
            _draw_element(ax, el, facecolor="#eeeeee",
                          edgecolor="#888", alpha=0.4, lw=1)
            for (xi, eta, _), sigma in zip(GAUSS_2X2, el_res.stresses_at_gauss):
                N = shape_functions(xi, eta)
                x_gp = sum(N[i] * el.nodes[i].x for i in range(4))
                y_gp = sum(N[i] * el.nodes[i].y for i in range(4))
                color = "#1f77b4" if sigma[0] >= 0 else "#d62728"
                ax.plot(x_gp, y_gp, "o", color=color, markersize=9)
                ax.annotate(f"{sigma[0]:.2e}", (x_gp, y_gp),
                            textcoords="offset points", xytext=(6, 6),
                            fontsize=7, color=color)
    elif plot_key == "stress_corners" and result is not None:
        ax.set_title("σx en las esquinas (nodos)")
        for el, el_res in zip(structure.elements, result.elements):
            _draw_element(ax, el, facecolor="#eeeeee",
                          edgecolor="#888", alpha=0.4, lw=1)
            for node, sigma in zip(el.nodes, el_res.stresses_at_corners):
                color = "#1f77b4" if sigma[0] >= 0 else "#d62728"
                ax.plot(node.x, node.y, "s", color=color, markersize=11)
                ax.annotate(f"{sigma[0]:.2f}", (node.x, node.y),
                            textcoords="offset points", xytext=(6, 6),
                            fontsize=7, color=color)
    else:
        return None

    if xs:
        ax.set_xlim(min(xs) - margin, max(xs) + margin)
        ax.set_ylim(min(ys) - margin, max(ys) + margin)
    return fig


def _draw_element_xy(ax, xy, **kw):
    poly = Polygon(xy, closed=True, **kw)
    ax.add_patch(poly)


def _fmt(v):
    """Formato para PDF: 15 dígitos sig (precisión completa float64)."""
    if isinstance(v, float):
        if not np.isfinite(v):
            return str(v)
        if abs(v) < 1e-13:
            return "0"
        return f"{v:.15g}"
    return str(v)


def _df_to_table(df) -> Table:
    data = [list(df.columns)]
    for _, row in df.iterrows():
        formatted = [_fmt(v) for v in row.values]
        data.append(formatted)
    tbl = Table(data, hAlign="LEFT")
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F4E78")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
    ]))
    return tbl


def _matrix_to_table(name: str, M: np.ndarray, styles) -> list:
    """Render una matriz como Table. Para matrices grandes (8×8) ajusta la
    fuente y el padding para que entre en el ancho A4 sin distorsión.
    """
    flow = [Paragraph(f"<b>{name}</b>", styles["Normal"])]
    if M.ndim == 1:
        M = M.reshape(-1, 1)
    rows, cols = M.shape
    # Ajustar precisión según el ancho de la matriz para que entre en A4
    if cols >= 8:
        # Matrices anchas (K^e, K_global): 3 dígitos significativos
        fmt = lambda v: ("0" if abs(float(v)) < 1e-13
                          else f"{float(v):.3g}")
        font_size = 6
    elif cols >= 4:
        fmt = lambda v: ("0" if abs(float(v)) < 1e-13
                          else f"{float(v):.4g}")
        font_size = 7
    else:
        fmt = _fmt
        font_size = 8
    data = [[fmt(float(v)) for v in row] for row in M]
    # Para matrices con 9+ columnas, restringir el ancho total = ancho A4 útil
    # (17 cm con márgenes de 1.5 cm) y repartir entre columnas para que entre.
    table_kwargs = dict(hAlign="LEFT")
    if cols >= 9:
        col_w = (17 * cm) / cols   # ancho útil A4 dividido por columnas
        table_kwargs["colWidths"] = [col_w] * cols
    tbl = Table(data, **table_kwargs)
    tbl.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), "Courier"),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
        ("ALIGN", (0, 0), (-1, -1), "RIGHT"),
        ("LEFTPADDING", (0, 0), (-1, -1), 2),
        ("RIGHTPADDING", (0, 0), (-1, -1), 2),
        ("TOPPADDING", (0, 0), (-1, -1), 1),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
    ]))
    flow.append(tbl)
    flow.append(Spacer(1, 0.3 * cm))
    return flow


def export_to_pdf(procedure: Procedure, structure: Structure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(path), pagesize=A4,
        leftMargin=1.5*cm, rightMargin=1.5*cm,
        topMargin=1.5*cm, bottomMargin=1.5*cm,
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "t1", parent=styles["Title"], textColor=colors.HexColor("#1F4E78"))
    h2 = ParagraphStyle(
        "h2", parent=styles["Heading2"], textColor=colors.HexColor("#1F4E78"))

    story = []
    story.append(Paragraph("Análisis por Método de Elementos Finitos — Elemento Q4",
                           title_style))
    story.append(Spacer(1, 0.5*cm))
    story.append(Paragraph(
        "Reporte generado automáticamente con el desarrollo paso a paso "
        "del problema, incluyendo geometría, propiedades, funciones de forma, "
        "puntos de Gauss, matrices D, J, B, ensamblaje de K, solución del "
        "sistema y resultados (desplazamientos, reacciones, deformaciones y esfuerzos).",
        styles["Normal"]))
    story.append(PageBreak())

    tmp_files: list[Path] = []
    try:
        for step in procedure.steps:
            story.append(Paragraph(step.title, h2))
            story.append(Paragraph(step.description.replace("\n", "<br/>"),
                                   styles["Normal"]))
            story.append(Spacer(1, 0.3*cm))

            if step.plot_key:
                fig = _figure_for_plot(step.plot_key, structure, procedure.result)
                if fig is not None:
                    tmp = Path(tempfile.mkstemp(suffix=".png")[1])
                    fig.savefig(str(tmp), dpi=150, bbox_inches="tight")
                    tmp_files.append(tmp)
                    # Preservar aspect ratio: calcular height a partir del width
                    # ancho fijo 14 cm, alto proporcional según el aspect real.
                    try:
                        from PIL import Image as _PILImage
                        with _PILImage.open(str(tmp)) as im:
                            w_px, h_px = im.size
                        target_w = 14 * cm
                        target_h = target_w * h_px / w_px
                        # Limitar a 14 cm de alto máximo (para que entre en pág)
                        if target_h > 14 * cm:
                            target_h = 14 * cm
                            target_w = target_h * w_px / h_px
                        story.append(Image(str(tmp), width=target_w, height=target_h))
                    except Exception:
                        # Fallback: tamaño fijo proporcional
                        story.append(Image(str(tmp), width=14 * cm, height=9 * cm))
                    story.append(Spacer(1, 0.3 * cm))

            for name, df in step.tables.items():
                if df.empty:
                    continue
                story.append(Paragraph(f"<b>{name}</b>", styles["Normal"]))
                story.append(_df_to_table(df))
                story.append(Spacer(1, 0.3*cm))

            for name, M in step.matrices.items():
                story.extend(_matrix_to_table(name, M, styles))

            story.append(PageBreak())

        doc.build(story)
    finally:
        for f in tmp_files:
            try:
                f.unlink()
            except OSError:
                pass
