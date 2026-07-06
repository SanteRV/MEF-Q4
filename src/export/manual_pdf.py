"""Manual teórico del MEF Q4 — PDF estático de referencia.

Contiene la teoría del elemento Q4 SIN valores específicos:
- Funciones de forma N1..N4 + derivadas
- Cuadratura de Gauss 2×2
- Derivación del Jacobiano J y matriz B
- Matrices D (tensión y deformación plana) + K^e

Útil como anexo de tesis / material de estudio.
"""
from __future__ import annotations
from pathlib import Path
import tempfile
import io

import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Polygon, FancyArrowPatch
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image,
    PageBreak, KeepTogether,
)


PRIMARY = colors.HexColor("#1F4E78")
PRIMARY_LIGHT = colors.HexColor("#D8E4F0")


# =========================================================================
# Diagramas (matplotlib → png temporal)
# =========================================================================
def _fig_natural_q4_element() -> Path:
    """Diagrama del elemento Q4 en coordenadas naturales (ξ, η)."""
    fig = Figure(figsize=(5.5, 5), tight_layout=True)
    ax = fig.add_subplot(111)
    ax.set_aspect("equal")
    # Cuadrado en coord naturales
    poly = Polygon([(1, 1), (-1, 1), (-1, -1), (1, -1)],
                   closed=True, facecolor="#cfe2f3",
                   edgecolor="#1f77b4", linewidth=2)
    ax.add_patch(poly)
    # Nodos numerados (convención ++,-+,--,+-)
    nodes = [(1, 1, "N₁ (+,+)"), (-1, 1, "N₂ (-,+)"),
             (-1, -1, "N₃ (-,-)"), (1, -1, "N₄ (+,-)")]
    for x, y, lbl in nodes:
        ax.plot(x, y, "o", color="black", markersize=10)
        ax.annotate(lbl, (x, y), textcoords="offset points",
                    xytext=(10*np.sign(x) if x != 0 else 10,
                            10*np.sign(y) if y != 0 else 10),
                    fontsize=10, fontweight="bold")
    # Ejes ξ y η
    ax.annotate("", xy=(1.5, 0), xytext=(-1.5, 0),
                arrowprops=dict(arrowstyle="->", color="#888", lw=1.2))
    ax.annotate("ξ", (1.55, -0.05), fontsize=14, color="#444")
    ax.annotate("", xy=(0, 1.5), xytext=(0, -1.5),
                arrowprops=dict(arrowstyle="->", color="#888", lw=1.2))
    ax.annotate("η", (0.05, 1.55), fontsize=14, color="#444")
    ax.plot(0, 0, "x", color="red", markersize=8)
    ax.annotate("origen", (0.05, 0.1), color="red", fontsize=9)
    ax.set_xlim(-1.8, 1.8)
    ax.set_ylim(-1.8, 1.8)
    ax.grid(True, alpha=0.3)
    ax.set_title("Elemento Q4 en coordenadas naturales (ξ, η) ∈ [−1, 1]")
    ax.set_xlabel("ξ")
    ax.set_ylabel("η")
    p = Path(tempfile.mkstemp(suffix=".png")[1])
    fig.savefig(str(p), dpi=140, bbox_inches="tight")
    return p


def _fig_gauss_points() -> Path:
    """Diagrama de los 4 puntos de Gauss en el elemento."""
    fig = Figure(figsize=(5.5, 5), tight_layout=True)
    ax = fig.add_subplot(111)
    ax.set_aspect("equal")
    poly = Polygon([(1, 1), (-1, 1), (-1, -1), (1, -1)],
                   closed=True, facecolor="#eeeeee",
                   edgecolor="#888", linewidth=1.5)
    ax.add_patch(poly)
    g = 1 / np.sqrt(3)
    gps = [(+g, +g, "GP₁"), (-g, +g, "GP₂"), (-g, -g, "GP₃"), (+g, -g, "GP₄")]
    for x, y, lbl in gps:
        ax.plot(x, y, "*", color="#d62728", markersize=18)
        ax.annotate(lbl, (x, y), textcoords="offset points",
                    xytext=(10, 10), fontsize=10, fontweight="bold",
                    color="#d62728")
    # Nodos solo como referencia
    for x, y in [(1, 1), (-1, 1), (-1, -1), (1, -1)]:
        ax.plot(x, y, "ko", markersize=6, alpha=0.4)
    ax.set_xlim(-1.4, 1.4); ax.set_ylim(-1.4, 1.4)
    ax.grid(True, alpha=0.3)
    ax.set_title("4 puntos de Gauss 2×2 en (±1/√3, ±1/√3), wᵢ = 1")
    ax.set_xlabel("ξ"); ax.set_ylabel("η")
    p = Path(tempfile.mkstemp(suffix=".png")[1])
    fig.savefig(str(p), dpi=140, bbox_inches="tight")
    return p


def _fig_mapping() -> Path:
    """Mapeo del elemento de coordenadas naturales a físicas."""
    fig = Figure(figsize=(8, 4), tight_layout=True)
    # Izquierda: elemento natural
    ax1 = fig.add_subplot(121)
    ax1.set_aspect("equal")
    poly = Polygon([(1, 1), (-1, 1), (-1, -1), (1, -1)], closed=True,
                   facecolor="#cfe2f3", edgecolor="#1f77b4", linewidth=2)
    ax1.add_patch(poly)
    for x, y, lbl in [(1, 1, "1"), (-1, 1, "2"), (-1, -1, "3"), (1, -1, "4")]:
        ax1.plot(x, y, "ko", markersize=7)
        ax1.annotate(lbl, (x, y), textcoords="offset points",
                     xytext=(8, 8), fontsize=11, fontweight="bold")
    ax1.set_xlim(-1.5, 1.5); ax1.set_ylim(-1.5, 1.5)
    ax1.set_title("Coords naturales (ξ, η)")
    ax1.set_xlabel("ξ"); ax1.set_ylabel("η")
    ax1.grid(True, alpha=0.3)

    # Flecha de mapeo
    ax_mid = fig.add_axes([0.46, 0.40, 0.08, 0.2])
    ax_mid.axis("off")
    ax_mid.annotate("", xy=(1, 0.5), xytext=(0, 0.5),
                    arrowprops=dict(arrowstyle="->", color=PRIMARY.rgb(),
                                    lw=2.5))
    ax_mid.annotate("x = Σ Nᵢ·xᵢ\ny = Σ Nᵢ·yᵢ",
                    (0.5, 0.7), ha="center", fontsize=8)

    # Derecha: elemento físico (cualquier quad)
    ax2 = fig.add_subplot(122)
    ax2.set_aspect("equal")
    physical = [(2.5, 1.8), (0.5, 1.5), (0.2, 0.2), (2.0, 0.4)]
    poly2 = Polygon(physical, closed=True, facecolor="#fce5cd",
                    edgecolor="#d97706", linewidth=2)
    ax2.add_patch(poly2)
    for (x, y), lbl in zip(physical, ["1", "2", "3", "4"]):
        ax2.plot(x, y, "ko", markersize=7)
        ax2.annotate(lbl, (x, y), textcoords="offset points",
                     xytext=(8, 8), fontsize=11, fontweight="bold")
    ax2.set_xlim(-0.3, 3); ax2.set_ylim(-0.3, 2.5)
    ax2.set_title("Coords físicas (x, y)")
    ax2.set_xlabel("x"); ax2.set_ylabel("y")
    ax2.grid(True, alpha=0.3)

    p = Path(tempfile.mkstemp(suffix=".png")[1])
    fig.savefig(str(p), dpi=140, bbox_inches="tight")
    return p


# =========================================================================
# Generación del PDF
# =========================================================================
def export_manual(path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(path), pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm,
    )
    styles = getSampleStyleSheet()
    H1 = ParagraphStyle("H1", parent=styles["Title"], textColor=PRIMARY,
                        fontSize=22, alignment=1, spaceAfter=12)
    H2 = ParagraphStyle("H2", parent=styles["Heading2"], textColor=PRIMARY,
                        spaceBefore=18, spaceAfter=8)
    H3 = ParagraphStyle("H3", parent=styles["Heading3"], textColor=PRIMARY,
                        spaceBefore=10, spaceAfter=4)
    Body = ParagraphStyle("Body", parent=styles["BodyText"],
                          fontSize=10, leading=14, alignment=4)
    Math = ParagraphStyle("Math", parent=styles["BodyText"],
                          fontName="Courier", fontSize=10, leading=14,
                          alignment=1, backColor=PRIMARY_LIGHT,
                          borderColor=PRIMARY, borderWidth=0.5, borderPadding=8,
                          spaceBefore=6, spaceAfter=6)

    story = []
    tmp_files: list[Path] = []

    # ===== Portada =====
    story.append(Spacer(1, 4*cm))
    story.append(Paragraph(
        "Manual Teórico<br/>Método de Elementos Finitos<br/>Elemento Q4", H1))
    story.append(Spacer(1, 1*cm))
    story.append(Paragraph(
        "Referencia teórica del elemento cuadrilátero bilineal para problemas "
        "planos de elasticidad lineal.", Body))
    story.append(Spacer(1, 3*cm))
    story.append(Paragraph("<i>Trabajo de Tesis · Ingeniería Civil</i>",
                           ParagraphStyle("subtitle", parent=Body,
                                          alignment=1, textColor=PRIMARY)))
    story.append(PageBreak())

    # ===== Sección 1: Introducción al Q4 =====
    story.append(Paragraph("1. El elemento cuadrilátero bilineal (Q4)", H2))
    story.append(Paragraph(
        "El elemento finito Q4 es un cuadrilátero de 4 nodos con 2 grados de "
        "libertad (GDL) por nodo (uₓ, u_y), totalizando 8 GDL. Se formula en "
        "<b>coordenadas naturales</b> (ξ, η) ∈ [−1, 1], lo que permite "
        "trabajar con cualquier cuadrilátero físico mediante un mapeo isoparamétrico.",
        Body))

    img_natural = _fig_natural_q4_element()
    tmp_files.append(img_natural)
    story.append(Image(str(img_natural), width=11*cm, height=10*cm))
    story.append(Paragraph(
        "<i>Figura 1: Elemento Q4 en coordenadas naturales con la convención "
        "de numeración de nodos usada en este aplicativo: N₁ en (+,+), "
        "N₂ en (−,+), N₃ en (−,−), N₄ en (+,−), recorriendo en sentido "
        "antihorario desde la esquina superior derecha.</i>", Body))
    story.append(Spacer(1, 0.4*cm))

    img_map = _fig_mapping()
    tmp_files.append(img_map)
    story.append(Image(str(img_map), width=16*cm, height=8*cm))
    story.append(Paragraph(
        "<i>Figura 2: Mapeo isoparamétrico — el cuadrado patrón en (ξ, η) "
        "se transforma a cualquier cuadrilátero físico en (x, y) usando las "
        "funciones de forma como interpoladores.</i>", Body))

    # ===== Sección 2: Funciones de forma =====
    story.append(Paragraph("2. Funciones de forma N₁ … N₄", H2))
    story.append(Paragraph(
        "Las funciones de forma son polinomios bilineales que cumplen la "
        "propiedad de Kronecker: Nᵢ(ξⱼ, ηⱼ) = δᵢⱼ. Es decir, valen 1 en su "
        "nodo asociado y 0 en los otros tres. Esto permite interpolar "
        "cualquier campo f dentro del elemento como:", Body))
    story.append(Paragraph(
        "f(ξ, η) = N₁(ξ,η)·f₁ + N₂(ξ,η)·f₂ + N₃(ξ,η)·f₃ + N₄(ξ,η)·f₄", Math))
    story.append(Paragraph("Forma compacta general:", Body))
    story.append(Paragraph(
        "Nᵢ(ξ, η) = ¼ · (1 + ξ·ξᵢ)(1 + η·ηᵢ)", Math))
    story.append(Paragraph(
        "Expandiendo para cada nodo con la convención (++, −+, −−, +−):", Body))
    data_N = [
        ["Nodo", "(ξᵢ, ηᵢ)", "Función de forma"],
        ["N₁", "(+1, +1)", "¼ · (1 + ξ)(1 + η)"],
        ["N₂", "(−1, +1)", "¼ · (1 − ξ)(1 + η)"],
        ["N₃", "(−1, −1)", "¼ · (1 − ξ)(1 − η)"],
        ["N₄", "(+1, −1)", "¼ · (1 + ξ)(1 − η)"],
    ]
    tbl = Table(data_N, hAlign="LEFT")
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), PRIMARY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PRIMARY_LIGHT]),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 0.4*cm))

    story.append(Paragraph("2.1. Derivadas respecto a coordenadas naturales", H3))
    story.append(Paragraph(
        "∂Nᵢ/∂ξ = ¼ · ξᵢ · (1 + η·ηᵢ)<br/>"
        "∂Nᵢ/∂η = ¼ · ηᵢ · (1 + ξ·ξᵢ)", Math))
    story.append(Paragraph(
        "Estas derivadas son las que se usan para construir el Jacobiano "
        "(sección 4) y la matriz B (sección 5).", Body))

    story.append(PageBreak())

    # ===== Sección 3: Cuadratura de Gauss =====
    story.append(Paragraph("3. Cuadratura de Gauss 2×2", H2))
    story.append(Paragraph(
        "Las integrales sobre el elemento Q4 no se pueden evaluar "
        "analíticamente para geometrías arbitrarias. Se aproximan con "
        "cuadratura numérica de Gauss-Legendre, que evalúa el integrando en "
        "puntos cuidadosamente elegidos:", Body))
    story.append(Paragraph(
        "∫₋₁¹ ∫₋₁¹ f(ξ, η) dξ dη ≈ Σᵢ wᵢ · f(ξᵢ, ηᵢ)", Math))
    story.append(Paragraph(
        "Para 2 puntos por dirección (2×2 = 4 puntos en total) en el "
        "elemento Q4, los puntos están en (±1/√3, ±1/√3) ≈ (±0.5774, ±0.5774) "
        "con peso wᵢ = 1 cada uno.", Body))
    story.append(Paragraph(
        "Esta regla es <b>exacta</b> para polinomios de grado ≤ 3 en cada "
        "variable, que coincide con el grado del integrando Bᵀ · D · B "
        "para un Q4 bilineal con D constante. Por eso 2×2 es óptimo "
        "(ni sobre-integración ni sub-integración).", Body))
    img_gp = _fig_gauss_points()
    tmp_files.append(img_gp)
    story.append(Image(str(img_gp), width=11*cm, height=10*cm))
    story.append(Paragraph(
        "<i>Figura 3: Ubicación de los 4 puntos de Gauss en el elemento patrón.</i>",
        Body))

    story.append(PageBreak())

    # ===== Sección 4: Jacobiano =====
    story.append(Paragraph("4. Jacobiano del mapeo natural → físico", H2))
    story.append(Paragraph(
        "El mapeo isoparamétrico transforma (ξ, η) → (x, y):", Body))
    story.append(Paragraph(
        "x(ξ, η) = Σ Nᵢ(ξ, η) · xᵢ,    y(ξ, η) = Σ Nᵢ(ξ, η) · yᵢ", Math))
    story.append(Paragraph(
        "El Jacobiano J es la matriz 2×2 de derivadas parciales:", Body))
    story.append(Paragraph(
        "J = | ∂x/∂ξ   ∂y/∂ξ |   con   ∂x/∂ξ = Σ (∂Nᵢ/∂ξ) · xᵢ<br/>"
        "    | ∂x/∂η   ∂y/∂η |        ∂y/∂ξ = Σ (∂Nᵢ/∂ξ) · yᵢ", Math))
    story.append(Paragraph(
        "Su determinante |J| es el factor de cambio de área entre el dominio "
        "natural y el físico: <b>dx·dy = |J|·dξ·dη</b>. Por tanto las "
        "integrales se transforman así:", Body))
    story.append(Paragraph(
        "∫∫_físico f(x,y) dx dy = ∫₋₁¹ ∫₋₁¹ f(x(ξ,η), y(ξ,η)) · |J| dξ dη", Math))
    story.append(Paragraph(
        "<b>Casos especiales:</b><br/>"
        "• Elemento rectangular alineado a/2 × b/2: J = diag(a/2, b/2), |J| = ab/4, constante.<br/>"
        "• Parallelogramo: J constante pero con términos off-diagonales.<br/>"
        "• Cuadrilátero general: J varía con (ξ, η); por eso se evalúa en cada GP.",
        Body))

    story.append(Paragraph("4.1. Inversa del Jacobiano y derivadas físicas", H3))
    story.append(Paragraph(
        "Las derivadas de las shape functions respecto a (x, y) se obtienen "
        "mediante la regla de la cadena:", Body))
    story.append(Paragraph(
        "| ∂Nᵢ/∂x |  =  J⁻¹ · | ∂Nᵢ/∂ξ |<br/>"
        "| ∂Nᵢ/∂y |          | ∂Nᵢ/∂η |", Math))

    story.append(PageBreak())

    # ===== Sección 5: Matriz B =====
    story.append(Paragraph("5. Matriz B (strain-displacement)", H2))
    story.append(Paragraph(
        "Las deformaciones en problema plano son:", Body))
    story.append(Paragraph(
        "ε = [εx, εy, γxy]ᵀ", Math))
    story.append(Paragraph(
        "y se obtienen a partir de los desplazamientos nodales u^e via la "
        "matriz B (3 × 8):", Body))
    story.append(Paragraph(
        "ε(ξ, η) = B(ξ, η) · u^e", Math))
    story.append(Paragraph("donde:", Body))
    story.append(Paragraph(
        "B = | ∂N₁/∂x   0    ∂N₂/∂x   0    ∂N₃/∂x   0    ∂N₄/∂x   0   |<br/>"
        "    | 0    ∂N₁/∂y   0    ∂N₂/∂y   0    ∂N₃/∂y   0    ∂N₄/∂y |<br/>"
        "    | ∂N₁/∂y ∂N₁/∂x ∂N₂/∂y ∂N₂/∂x ∂N₃/∂y ∂N₃/∂x ∂N₄/∂y ∂N₄/∂x |", Math))

    # ===== Sección 6: Matriz D =====
    story.append(Paragraph("6. Matriz constitutiva D", H2))
    story.append(Paragraph("6.1. Tensión plana (placas delgadas)", H3))
    story.append(Paragraph(
        "Hipótesis: σz = τxz = τyz = 0. Apropiada para placas cuyo espesor "
        "es mucho menor que las dimensiones en el plano.", Body))
    story.append(Paragraph(
        "D_tp = E/(1−ν²) · | 1   ν   0     |<br/>"
        "                  | ν   1   0     |<br/>"
        "                  | 0   0  (1−ν)/2 |", Math))

    story.append(Paragraph("6.2. Deformación plana (sólidos largos)", H3))
    story.append(Paragraph(
        "Hipótesis: εz = γxz = γyz = 0. Apropiada para sólidos cuya "
        "dimensión normal al plano es mucho mayor (ej. túneles, presas).",
        Body))
    story.append(Paragraph(
        "D_dp = E/((1+ν)(1−2ν)) · | 1−ν   ν    0      |<br/>"
        "                          | ν    1−ν   0      |<br/>"
        "                          | 0    0   (1−2ν)/2 |", Math))

    # ===== Sección 7: K^e =====
    story.append(Paragraph("7. Matriz de rigidez del elemento K^e", H2))
    story.append(Paragraph(
        "La matriz de rigidez del elemento se obtiene integrando sobre su "
        "dominio:", Body))
    story.append(Paragraph(
        "K^e = ∫∫_Ωe Bᵀ · D · B · t dx dy", Math))
    story.append(Paragraph(
        "Aplicando el cambio de variable al dominio natural y usando la "
        "cuadratura de Gauss 2×2:", Body))
    story.append(Paragraph(
        "K^e ≈ Σᵢ Bᵢᵀ · D · Bᵢ · t · |Jᵢ| · wᵢ      (i = 1..4)", Math))
    story.append(Paragraph(
        "donde Bᵢ y |Jᵢ| son las matrices evaluadas en el punto de Gauss i, "
        "t es el espesor del elemento y wᵢ = 1 son los pesos de la cuadratura.",
        Body))
    story.append(Paragraph(
        "K^e resulta una matriz 8 × 8 simétrica y semi-definida positiva.",
        Body))

    story.append(PageBreak())

    # ===== Sección 8: Sistema global y solución =====
    story.append(Paragraph("8. Ensamblaje y solución del sistema", H2))
    story.append(Paragraph(
        "La matriz de rigidez global K se obtiene por suma directa de los "
        "K^e de cada elemento en las posiciones correspondientes a sus GDL "
        "(método de los códigos LM en la literatura).", Body))
    story.append(Paragraph(
        "K · u = F", Math))
    story.append(Paragraph(
        "Una vez aplicadas las condiciones de borde (eliminando filas y "
        "columnas de GDL restringidos), se resuelve el sistema reducido:",
        Body))
    story.append(Paragraph(
        "K_ff · u_f = F_f", Math))
    story.append(Paragraph(
        "Los desplazamientos en GDL restringidos son cero, y las reacciones "
        "se calculan a posteriori:", Body))
    story.append(Paragraph("R = K · u − F", Math))

    # ===== Sección 9: Post-proceso =====
    story.append(Paragraph("9. Post-proceso: deformaciones y esfuerzos", H2))
    story.append(Paragraph(
        "Con los desplazamientos conocidos, las deformaciones y esfuerzos "
        "en cualquier punto del elemento (típicamente en los puntos de Gauss "
        "por ser super-convergentes) se calculan como:", Body))
    story.append(Paragraph(
        "ε = B(ξ, η) · u^e<br/>"
        "σ = D · ε", Math))
    story.append(Paragraph(
        "Para visualización suelen extrapolarse a las esquinas / nodos. "
        "También se pueden calcular esfuerzos principales (σ₁, σ₂) y "
        "von Mises (σ_VM) a partir de σ.", Body))

    # ===== Referencias =====
    story.append(PageBreak())
    story.append(Paragraph("Referencias bibliográficas", H2))
    refs = [
        "Bathe, K.-J. (2014). <i>Finite Element Procedures</i>. 2nd ed. Prentice Hall.",
        "Zienkiewicz, O. C.; Taylor, R. L.; Zhu, J. Z. (2013). "
        "<i>The Finite Element Method: Its Basis and Fundamentals</i>. 7th ed. Butterworth-Heinemann.",
        "Cook, R. D.; Malkus, D. S.; Plesha, M. E.; Witt, R. J. (2002). "
        "<i>Concepts and Applications of Finite Element Analysis</i>. 4th ed. Wiley.",
        "Oñate, E. (2009). <i>Structural Analysis with the Finite Element Method</i>. Vol. 1. Springer.",
    ]
    for r in refs:
        story.append(Paragraph(f"• {r}", Body))
        story.append(Spacer(1, 0.2*cm))

    try:
        doc.build(story)
    finally:
        for f in tmp_files:
            try:
                f.unlink()
            except OSError:
                pass
