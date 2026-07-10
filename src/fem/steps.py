"""Generador de pasos didácticos del análisis MEF con elementos Q4.

Cada paso contiene tablas, matrices y, opcionalmente, una gráfica.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

from .structure import Structure
from .q4_element import GAUSS_2X2, NATURAL_COORDS, Q4Element
from .solver import (
    assemble_global_stiffness,
    assemble_load_vector,
    solve,
    FEMResult,
)

# Import opcional (no rompe si steps.py se usa sin UI)
try:
    from ..ui.user_level import is_novato as _is_novato
except Exception:
    def _is_novato() -> bool:
        return True


def _desc(novato: str, experto: str) -> str:
    """Devuelve la descripción adecuada según el nivel actual del usuario."""
    return novato if _is_novato() else experto


def _is_axis_aligned_rectangle(el: Q4Element, tol: float = 1e-9) -> bool:
    """Detecta si el Q4 es un rectángulo con lados paralelos a los ejes.

    Sólo en este caso la fórmula J = diag(a/2, b/2) es válida; en parallelogramos
    rotados J es constante pero con términos fuera de la diagonal; en cuadriláteros
    generales J varía con (ξ, η).
    """
    coords = el.coords
    xs = sorted({round(c[0] / tol) for c in coords})
    ys = sorted({round(c[1] / tol) for c in coords})
    # Sólo 2 valores únicos de x y de y → rectángulo alineado
    return len(xs) == 2 and len(ys) == 2


def _is_parallelogram(el: Q4Element, tol: float = 1e-9) -> bool:
    """Detecta si el Q4 es un parallelogramo (J constante pero quizá con off-diagonal)."""
    p = el.coords
    # Para parallelogramo: vector (p1→p2) == vector (p4→p3)  (o equivalente)
    v_top = p[1] - p[0]
    v_bot = p[2] - p[3]
    return bool(np.allclose(v_top, v_bot, atol=tol))


@dataclass
class Step:
    title: str
    description: str
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    matrices: dict[str, np.ndarray] = field(default_factory=dict)
    plot_key: str | None = None


@dataclass
class Procedure:
    steps: list[Step]
    result: FEMResult


def _shape_function_text() -> pd.DataFrame:
    return pd.DataFrame([
        {"Función": "N1", "Expresión": "1/4 · (1 + ξ)(1 + η)", "ξi": +1, "ηi": +1},
        {"Función": "N2", "Expresión": "1/4 · (1 − ξ)(1 + η)", "ξi": -1, "ηi": +1},
        {"Función": "N3", "Expresión": "1/4 · (1 − ξ)(1 − η)", "ξi": -1, "ηi": -1},
        {"Función": "N4", "Expresión": "1/4 · (1 + ξ)(1 − η)", "ξi": +1, "ηi": -1},
    ])


def _gauss_points_text() -> pd.DataFrame:
    g = 1.0 / np.sqrt(3.0)
    return pd.DataFrame([
        {"Punto": "GP1", "ξ": +g, "η": +g, "Peso": 1.0},
        {"Punto": "GP2", "ξ": -g, "η": +g, "Peso": 1.0},
        {"Punto": "GP3", "ξ": -g, "η": -g, "Peso": 1.0},
        {"Punto": "GP4", "ξ": +g, "η": -g, "Peso": 1.0},
    ])


def build_procedure(structure: Structure) -> Procedure:
    steps: list[Step] = []

    # ============= 1. Datos de entrada =============
    df_nodes = pd.DataFrame(
        [{"Nodo": n.id + 1, "x (m)": n.x, "y (m)": n.y} for n in structure.nodes]
    )
    df_props_rows = []
    for el in structure.elements:
        df_props_rows.append({
            "Elemento": el.id + 1,
            "Nodos": " - ".join(str(n.id + 1) for n in el.nodes),
            "E (Pa)": el.E,
            "ν": el.nu,
            "t (m)": el.t,
            "Hipótesis": "Tensión plana" if el.plane_stress else "Deformación plana",
        })
    df_props = pd.DataFrame(df_props_rows)

    # Lista de hipótesis presentes (puede haber elementos con distinta hipótesis)
    hyps = {("Tensión plana" if el.plane_stress else "Deformación plana")
            for el in structure.elements}
    hyp_str = " y ".join(sorted(hyps)) if hyps else "—"
    steps.append(Step(
        title="1. Datos de entrada (geometría y propiedades)",
        description=_desc(
            novato=(
                "Se ingresan las coordenadas (x, y) de los nodos del elemento Q4 "
                "y las propiedades del material:\n"
                "  • E: módulo de elasticidad (rigidez del material, en Pa)\n"
                "  • ν: coeficiente de Poisson (contracción lateral, adimensional)\n"
                "  • t: espesor de la placa fuera del plano (en m)\n"
                "\n"
                f"Hipótesis del problema plano: {hyp_str}.\n"
                "\n"
                "El paso siguiente define qué nodos están restringidos (apoyos) "
                "y qué cargas externas se aplican."
            ),
            experto=f"Geometría + (E, ν, t) + hipótesis: {hyp_str}.",
        ),
        tables={"Nodos": df_nodes, "Propiedades del elemento": df_props},
        plot_key="mesh",
    ))

    # ============= 2. Condiciones de borde y cargas =============
    df_bc = pd.DataFrame([
        {
            "Nodo": n.id + 1,
            "Restringido X": "Sí" if n.restraint_x else "No",
            "Restringido Y": "Sí" if n.restraint_y else "No",
            "Carga Fx (N)": n.load_x,
            "Carga Fy (N)": n.load_y,
        }
        for n in structure.nodes
    ])
    steps.append(Step(
        title="2. Condiciones de borde y cargas",
        description=_desc(
            novato=(
                "Para que el sistema K·u = F tenga solución única, hay que:\n"
                "  1. Restringir suficientes grados de libertad para evitar "
                "movimientos de cuerpo rígido (traslación + rotación en 2D = 3 GDL).\n"
                "  2. Aplicar las cargas externas como fuerzas concentradas en los nodos.\n"
                "\n"
                "Convenciones:\n"
                "  • Restringido X = el nodo no puede moverse en dirección horizontal.\n"
                "  • Restringido Y = el nodo no puede moverse en dirección vertical.\n"
                "  • Fx, Fy = componentes de la fuerza aplicada en el nodo (N).\n"
                "\n"
                "Estos GDL restringidos se eliminarán del sistema en el paso 11."
            ),
            experto="Restricciones (Fix X/Y) + cargas Fx, Fy por nodo.",
        ),
        tables={"Apoyos y cargas": df_bc},
        plot_key="bc",
    ))

    # ============= 3. Funciones de forma =============
    steps.append(Step(
        title="3. Funciones de forma del Q4",
        description=_desc(
            novato=(
                "El elemento cuadrilátero bilineal Q4 se formula en coordenadas "
                "naturales (ξ, η) ∈ [-1, 1]. Las funciones de forma son "
                "polinomios bilineales que valen 1 en su nodo asociado y 0 en "
                "los otros tres:\n"
                "  Ni(ξ, η) = ¼ · (1 + ξ·ξi)(1 + η·ηi)\n"
                "donde (ξi, ηi) son las coordenadas naturales del nodo i.\n"
                "\n"
                "Estas funciones interpolan cualquier campo (geometría, "
                "desplazamiento) dentro del elemento: f(ξ,η) = Σ Ni(ξ,η) · fi.\n"
                "\n"
                "El diagrama muestra el cuadrado patrón en (ξ, η) ∈ [-1, 1] con "
                "la convención de numeración de la guía: N₁ en (-,-), N₂ en "
                "(+,-), N₃ en (+,+), N₄ en (-,+), recorriendo CCW desde la "
                "esquina inferior izquierda."
            ),
            experto=(
                "Funciones de forma bilineales:  Ni = ¼(1+ξ·ξi)(1+η·ηi)"
            ),
        ),
        tables={"Funciones de forma": _shape_function_text()},
        plot_key="shape_functions_diagram",
    ))

    # ============= 4. Puntos de Gauss =============
    steps.append(Step(
        title="4. Puntos de Gauss (integración 2×2)",
        plot_key="gauss_points_diagram",
        description=_desc(
            novato=(
                "La integral K^e = ∫∫ Bᵀ·D·B·t dx dy no se puede resolver "
                "analíticamente para un Q4 general. Se aproxima con cuadratura "
                "de Gauss, que evalúa el integrando en pocos puntos cuidadosamente "
                "elegidos:\n"
                "  ∫₋₁¹∫₋₁¹ f(ξ,η) dξ dη ≈ Σᵢ wᵢ · f(ξᵢ, ηᵢ)\n"
                "\n"
                "Con 2×2 puntos (4 totales) en ±1/√3 ≈ ±0.5774 y peso w=1, "
                "la cuadratura es exacta para polinomios de hasta grado 3 en (ξ,η), "
                "que es justo lo que aparece en Bᵀ·B para un Q4 bilineal."
            ),
            experto=(
                "Cuadratura Gauss 2×2: 4 puntos en (±1/√3, ±1/√3), w=1."
            ),
        ),
        tables={"Puntos de Gauss": _gauss_points_text()},
    ))

    # ============= 5. Matriz constitutiva D =============
    # Para cada elemento: matriz simbólica (estructura) + factor escalar
    # + matriz resultante (factor × estructura). Es la presentación pedida
    # en la corrección B6 del documento.
    el0 = structure.elements[0]
    D_matrices: dict[str, np.ndarray] = {}
    desc_lines: list[str] = []
    for el in structure.elements:
        suffix = f" (elemento {el.id + 1})" if len(structure.elements) > 1 else ""
        if el.plane_stress:
            hyp = "Tensión plana"
            scale = el.E / (1.0 - el.nu * el.nu)
            D_struct = np.array([
                [1.0,    el.nu, 0.0],
                [el.nu,  1.0,   0.0],
                [0.0,    0.0,   (1.0 - el.nu) / 2.0],
            ])
            formula = "D = E/(1−ν²) · structure(ν)"
            sustitution = (
                f"  E = {el.E:.6g} Pa,   ν = {el.nu:.6g}\n"
                f"  1 − ν² = 1 − {el.nu:.6g}² = {1 - el.nu**2:.6f}\n"
                f"  factor = E/(1−ν²) = {scale:.6g} Pa\n"
                f"  (1−ν)/2 = {(1 - el.nu)/2:.6g}"
            )
        else:
            hyp = "Deformación plana"
            scale = el.E / ((1.0 + el.nu) * (1.0 - 2.0 * el.nu))
            D_struct = np.array([
                [1.0 - el.nu, el.nu,       0.0],
                [el.nu,       1.0 - el.nu, 0.0],
                [0.0,         0.0,         (1.0 - 2.0 * el.nu) / 2.0],
            ])
            formula = "D = E/[(1+ν)(1−2ν)] · structure(ν)"
            sustitution = (
                f"  E = {el.E:.6g} Pa,   ν = {el.nu:.6g}\n"
                f"  (1+ν)(1−2ν) = {(1+el.nu)*(1-2*el.nu):.6f}\n"
                f"  factor = E/[(1+ν)(1−2ν)] = {scale:.6g} Pa\n"
                f"  (1−2ν)/2 = {(1-2*el.nu)/2:.6g}"
            )

        # 3 matrices por elemento para mostrar la "descomposición" simbólica:
        D_matrices[f"1) Estructura simbólica de D{suffix}"] = D_struct
        D_matrices[f"2) Factor escalar de D{suffix}"] = np.array([[scale]])
        D_matrices[f"3) D resultante = factor × estructura{suffix}"] = el.D_matrix()

        desc_lines.append(
            f"{hyp}{suffix}\n"
            f"Fórmula:  {formula}\n"
            f"Sustitución con los valores del proyecto:\n{sustitution}"
        )

    steps.append(Step(
        title="5. Matriz constitutiva D",
        description=_desc(
            novato=(
                "La matriz constitutiva D relaciona deformaciones con esfuerzos "
                "según la ley de Hooke generalizada para problema plano:\n"
                "  σ = D · ε     con   σ = [σx, σy, τxy]ᵀ,  ε = [εx, εy, γxy]ᵀ\n"
                "\n"
                "Se descompone en:\n"
                "  • Estructura simbólica (matriz 3×3 con ν, adimensional)\n"
                "  • Factor escalar (depende de E y ν, en Pa)\n"
                "  • D resultante = factor × estructura\n"
                "\n"
                + "\n\n".join(desc_lines)
                + "\n\nLa matriz D resultante es la que se usa en B^T·D·B durante "
                "el ensamblaje de K^e en el paso 7."
            ),
            experto="\n\n".join(desc_lines),
        ),
        matrices=D_matrices,
    ))

    # ============= 6. Jacobiano y matriz B en cada punto de Gauss =============
    # Se muestran las matrices en el mismo orden que el documento guía:
    # J (ec. 31) → det J → A (ec. 37) → G (ec. 38) → B = A·G (ec. 39).
    K_el, gp_list = el0.stiffness_matrix()
    matrices_gauss: dict[str, np.ndarray] = {}
    for i, gp in enumerate(gp_list, start=1):
        matrices_gauss[f"GP{i} — J (Jacobiano, ec. 31)"] = gp.J
        matrices_gauss[f"GP{i} — det J"] = np.array([[gp.detJ]])
        matrices_gauss[f"GP{i} — A (3×4, ec. 37)"] = gp.A
        matrices_gauss[f"GP{i} — G (4×8, ec. 38)"] = gp.G
        matrices_gauss[f"GP{i} — B = A·G (3×8, ec. 39)"] = gp.B

    # Descripción adaptativa según la geometría del elemento
    desc_geom = ""
    suffix = " del primer elemento" if len(structure.elements) > 1 else ""
    if _is_axis_aligned_rectangle(el0):
        xs0 = [n.x for n in el0.nodes]
        ys0 = [n.y for n in el0.nodes]
        a = max(xs0) - min(xs0)
        b = max(ys0) - min(ys0)
        desc_geom = (
            f"\nEste elemento{suffix} es un rectángulo {a:.15g}×{b:.15g} m alineado "
            f"con los ejes → J es diagonal y constante en todo el dominio, con "
            f"|J11| = a/2 = {a/2:.15g}, |J22| = b/2 = {b/2:.15g} y "
            f"det J = a·b/4 = {a*b/4:.15g}. (El signo de J11 y J22 depende de "
            f"qué esquina física ocupa el nodo 1; det J es siempre positivo.)"
        )
    elif _is_parallelogram(el0):
        desc_geom = (
            f"\nEste elemento{suffix} es un parallelogramo: J es constante "
            "en todo el dominio (puede tener términos fuera de la diagonal "
            "si está rotado o inclinado)."
        )
    else:
        desc_geom = (
            f"\nEste elemento{suffix} es un cuadrilátero general: J varía con "
            "la posición (ξ, η), por eso se evalúa en cada punto de Gauss."
        )

    steps.append(Step(
        title="6. Jacobiano J y matriz B = A·G en los puntos de Gauss",
        description=(
            "En cada punto de Gauss se calcula, siguiendo el orden del "
            "documento guía:\n"
            "• El Jacobiano J (2×2) con sus componentes J11, J12, J21, J22 "
            "(ec. 31):  J11 = ∂x/∂ξ,  J12 = ∂y/∂ξ,  J21 = ∂x/∂η,  J22 = ∂y/∂η\n"
            "• Su determinante det J, que realiza el cambio de variable "
            "dx·dy = det J·dξ·dη (ec. 33)\n"
            "• La matriz A (3×4), construida con la inversa explícita del "
            "Jacobiano (ec. 32 y 37):\n"
            "   A = (1/det J)·[ J22  −J12   0    0  ;"
            "   0   0  −J21  J11 ;  −J21  J11  J22  −J12 ]\n"
            "• La matriz G (4×8), con las derivadas de las funciones de forma "
            "respecto a ξ y η (ec. 38):  [∂u/∂ξ; ∂u/∂η; ∂v/∂ξ; ∂v/∂η] = G·q\n"
            "• La matriz strain-displacement B = A·G (3×8) (ec. 39), tal que "
            "ε = A·G·q = B·q"
            + desc_geom
        ),
        matrices=matrices_gauss,
    ))

    # ============= 7. Contribución K^e en cada punto de Gauss =============
    # Una matriz K_GPi = Bᵀ·D·B·t·|J|·w por cada uno de los 4 GP.
    # Si hay varios elementos se muestran los del primero.
    D0 = el0.D_matrix()
    gp_K_matrices: dict[str, np.ndarray] = {}
    sum_check = np.zeros((8, 8))
    for i, gp in enumerate(gp_list, start=1):
        K_i = gp.B.T @ D0 @ gp.B * el0.t * gp.detJ * gp.weight
        gp_K_matrices[f"K^e contribución GP{i}  (Bᵀ·D·B·t·|J|·w)"] = K_i
        sum_check += K_i
    elem_label = f" del elemento {el0.id + 1}" if len(structure.elements) > 1 else ""
    steps.append(Step(
        title="7. Contribución de cada punto de Gauss a K^e",
        description=_desc(
            novato=(
                f"En cada uno de los 4 puntos de Gauss{elem_label} se calcula la "
                "contribución a la matriz de rigidez del elemento:\n"
                "  K_GPi = Bᵢᵀ · D · Bᵢ · t · |Jᵢ| · wᵢ      (matriz 8×8)\n"
                "Donde Bᵢ y |Jᵢ| son los del paso 6, D es del paso 5, t es el espesor "
                "y wᵢ = 1 son los pesos de la cuadratura 2×2.\n"
                "\n"
                "Cada matriz tiene 8 filas y 8 columnas, correspondientes a los "
                "8 grados de libertad del elemento (u1x[1], u1y[2], u2x[3], ..., u4y[8]).\n"
                "\n"
                "La matriz K^e del elemento (paso 8) se obtiene sumando estas 4 "
                "contribuciones."
            ),
            experto=f"K_GPi = Bᵢᵀ·D·Bᵢ·t·|Jᵢ|·wᵢ (8×8), una por GP.",
        ),
        matrices=gp_K_matrices,
    ))

    # ============= 8. Matriz de rigidez del elemento =============
    Ke_matrices: dict[str, np.ndarray] = {}
    for el in structure.elements:
        if el.id == el0.id:
            Ke_matrices[f"K^e elemento {el.id + 1} (8×8) = ΣK_GPi"] = K_el
        else:
            Ke_el, _ = el.stiffness_matrix()
            Ke_matrices[f"K^e elemento {el.id + 1} (8×8) = ΣK_GPi"] = Ke_el
    steps.append(Step(
        title="8. Matriz de rigidez por elemento K^e (8×8)",
        description=_desc(
            novato=(
                "Según la fórmula de la guía:\n"
                "  K^e = t·∫₋₁¹∫₋₁¹ Bᵀ·D·B · det J · dξ·dη ≈ Σᵢ K_GPi   "
                "(suma de las 4 contribuciones del paso 7).\n"
                "\n"
                "La integral se plantea en coordenadas naturales (ξ, η) gracias "
                "al cambio de variable dx·dy = det J·dξ·dη (ec. 33). La "
                "cuadratura de Gauss 2×2 es exacta para polinomios de hasta "
                "grado 3 en (ξ, η), que es lo que aparece en el integrando "
                "para un Q4 bilineal.\n"
                "\n"
                "K^e es simétrica y semidefinida positiva. Sus filas y columnas "
                "están indexadas por los 8 GDL del elemento."
            ),
            experto=(
                "K^e = t·∫∫ Bᵀ·D·B·det J·dξ·dη = Σᵢ K_GPi "
                "(suma de las 4 contribuciones)."
            ),
        ),
        matrices=Ke_matrices,
    ))

    # ============= 8. Matriz global K =============
    K_global = assemble_global_stiffness(structure)
    n_el = len(structure.elements)
    if n_el == 1:
        desc_assembly = (
            "La matriz K^e se ubica en la matriz global K según los grados "
            "de libertad de los nodos del elemento. Con un único elemento Q4, "
            "K coincide con K^e (rellenando con ceros los GDL ajenos al elemento)."
        )
    else:
        desc_assembly = (
            f"La matriz global K ({2 * len(structure.nodes)}×"
            f"{2 * len(structure.nodes)}) se forma por suma directa: para cada "
            "uno de los " + str(n_el) + " elementos se suman los términos de "
            "K^e en las posiciones de la matriz global correspondientes a sus GDL."
        )
    steps.append(Step(
        title="9. Ensamblaje de la matriz de rigidez global K",
        description=desc_assembly,
        matrices={"K global": K_global},
    ))

    # ============= 10. Vector de cargas =============
    F = assemble_load_vector(structure)
    steps.append(Step(
        title="10. Vector de cargas global F",
        description=_desc(
            novato=(
                "El vector F (de tamaño igual al número total de GDL) contiene "
                "las cargas nodales aplicadas en cada componente:\n"
                "  F[1] = Fx en nodo 1\n"
                "  F[2] = Fy en nodo 1\n"
                "  F[3] = Fx en nodo 2\n"
                "  ...\n"
                "Los GDL sin carga aplicada son cero.\n"
                "\n"
                "Junto con K, este vector forma el sistema global K · u = F que "
                "se resuelve en los pasos siguientes."
            ),
            experto="F con tamaño 2·N — cargas por GDL.",
        ),
        matrices={"F": F.reshape(-1, 1)},
    ))

    # ============= 11. Sistema reducido =============
    free = structure.free_dofs()
    K_ff = K_global[np.ix_(free, free)]
    F_f = F[free]
    # Método de penalización (alternativa pedagógica al método de reducción):
    # K_pen[restringido, restringido] += α  con α muy grande (típicamente 1e10·max(K))
    K_pen = K_global.copy()
    F_pen = F.copy()
    alpha = 1e10 * float(np.max(np.abs(K_global))) if K_global.size else 1e15
    for d in structure.restrained_dofs():
        K_pen[d, d] += alpha
        F_pen[d] = 0.0   # restringimos a 0 (apoyo fijo)
    steps.append(Step(
        title="11. Aplicación de condiciones de borde (sistema reducido)",
        description=_desc(
            novato=(
                "Existen DOS métodos clásicos para imponer las condiciones de borde:\n"
                "\n"
                "**Método 1 — Reducción del sistema (el que usamos):**\n"
                "Se eliminan las filas y columnas correspondientes a GDL "
                "restringidos. Queda el sistema K_ff · u_f = F_f de tamaño menor.\n"
                "Ventaja: el sistema queda más pequeño y mejor condicionado.\n"
                "Desventaja: requiere reindexar los GDL.\n"
                "\n"
                "**Método 2 — Penalización (alternativa):**\n"
                "Se suma un valor α muy grande (≈ 10¹⁰ · max(K)) en la diagonal "
                "de los GDL restringidos. Esto los 'fuerza' a quedar en cero sin "
                "eliminarlos.\n"
                f"En este problema, α = {alpha:.3e}.\n"
                "Ventaja: el sistema mantiene su tamaño original (más simple de implementar).\n"
                "Desventaja: el sistema queda mal condicionado numéricamente.\n"
                "\n"
                "Las matrices mostradas son: K_ff (reducción) y K_pen (penalización). "
                "Ambos métodos producen los mismos desplazamientos (cuando α es "
                "suficientemente grande)."
            ),
            experto=(
                "Reducción: eliminar filas/cols restringidas → K_ff · u_f = F_f.\n"
                "Alternativa: penalización K[d,d] += α (α≈10¹⁰·max(K))."
            ),
        ),
        matrices={
            "K_ff (reducción)": K_ff,
            "F_f": F_f.reshape(-1, 1),
            "K_pen (penalización)": K_pen,
            "F_pen": F_pen.reshape(-1, 1),
        },
    ))

    # ============= 12. Solución (desplazamientos) =============
    result = solve(structure)
    df_disp = pd.DataFrame([
        {
            "Nodo": n.id + 1,
            "ux (m)": result.displacements[n.dofs[0]],
            "uy (m)": result.displacements[n.dofs[1]],
        }
        for n in structure.nodes
    ])
    steps.append(Step(
        title="12. Solución: desplazamientos nodales",
        description=_desc(
            novato=(
                "Se resuelve el sistema lineal K_ff · u_f = F_f mediante "
                "factorización LU (numpy.linalg.solve).\n"
                "\n"
                "Los desplazamientos en GDL restringidos son automáticamente 0 "
                "(no aparecen en el sistema reducido).\n"
                "\n"
                "La gráfica de la derecha muestra la deformada con un factor "
                "de amplificación visual (los desplazamientos reales suelen ser "
                "microscópicos: ~10⁻⁶ m). Usá el slider para cambiar la escala."
            ),
            experto="K_ff · u_f = F_f resuelto con LU (numpy.linalg.solve).",
        ),
        tables={"Desplazamientos": df_disp},
        plot_key="deformed",
    ))

    # ============= 13. Reacciones =============
    df_reac = pd.DataFrame([
        {
            "Nodo": n.id + 1,
            "Rx (N)": result.reactions[n.dofs[0]] if n.restraint_x else 0.0,
            "Ry (N)": result.reactions[n.dofs[1]] if n.restraint_y else 0.0,
        }
        for n in structure.nodes if n.restraint_x or n.restraint_y
    ])
    steps.append(Step(
        title="13. Reacciones en los apoyos",
        description=_desc(
            novato=(
                "Una vez conocidos los desplazamientos, las reacciones en los "
                "GDL restringidos se calculan como:\n"
                "  R = K · u − F   (en los GDL restringidos)\n"
                "\n"
                "Estas reacciones deben equilibrar las cargas externas aplicadas "
                "(ΣR + ΣF_aplicado = 0 para cada componente).\n"
                "\n"
                "Es una verificación útil: si la suma de reacciones no equilibra "
                "las cargas, hay un error en el cálculo."
            ),
            experto="R = K·u − F (en GDL restringidos).",
        ),
        tables={"Reacciones": df_reac if not df_reac.empty else pd.DataFrame()},
    ))

    # ============= 13. Esfuerzos en los puntos de Gauss =============
    rows = []
    for el_res, el in zip(result.elements, structure.elements):
        for i, (eps, sig) in enumerate(
            zip(el_res.strains_at_gauss, el_res.stresses_at_gauss), start=1
        ):
            rows.append({
                "Elemento": el.id + 1,
                "Punto Gauss": f"GP{i}",
                "εx": eps[0], "εy": eps[1], "γxy": eps[2],
                "σx (Pa)": sig[0], "σy (Pa)": sig[1], "τxy (Pa)": sig[2],
            })
    df_stress = pd.DataFrame(rows)
    steps.append(Step(
        title="14. Deformaciones y esfuerzos en puntos de Gauss",
        description=_desc(
            novato=(
                "Una vez conocidos los desplazamientos nodales, se calculan las "
                "deformaciones y esfuerzos en cada punto de Gauss:\n"
                "  ε(GP) = B(GP) · u^e         (deformaciones)\n"
                "  σ(GP) = D · ε(GP)           (esfuerzos)\n"
                "\n"
                "Los puntos de Gauss son los puntos óptimos para evaluar "
                "esfuerzos (super-convergencia), mejor que los nodos o cualquier "
                "otro punto interior.\n"
                "\n"
                "**Mapa de contornos:** la gráfica evalúa σ en una grilla densa "
                "(20×20 puntos por elemento) usando las shape functions para "
                "interpolar entre los GP, y lo dibuja como gradiente de color."
            ),
            experto="ε = B·u^e, σ = D·ε en cada GP. Mapa = interpolación con N.",
        ),
        tables={"Deformaciones y esfuerzos": df_stress},
        plot_key="stress",
    ))

    # ============= 14. Esfuerzos en las ESQUINAS (nodos) =============
    corner_labels = ["N1 (--)", "N2 (+-)", "N3 (++)", "N4 (-+)"]
    rows = []
    corner_matrices: dict[str, np.ndarray] = {}
    for el_res, el in zip(result.elements, structure.elements):
        # Matrices B en las 4 esquinas (para verificación manual)
        for label, (xi_n, eta_n) in zip(corner_labels, NATURAL_COORDS):
            B_corner, _, _, _ = el.B_matrix(float(xi_n), float(eta_n))
            key = f"B en {label}"
            if len(structure.elements) > 1:
                key = f"B en {label} (elem {el.id + 1})"
            corner_matrices[key] = B_corner
        # Tabla de esfuerzos
        for eps, sig, label in zip(
            el_res.strains_at_corners, el_res.stresses_at_corners, corner_labels
        ):
            rows.append({
                "Elemento": el.id + 1,
                "Esquina": label,
                "εx": eps[0], "εy": eps[1], "γxy": eps[2],
                "σx (Pa)": sig[0], "σy (Pa)": sig[1], "τxy (Pa)": sig[2],
            })
    df_corner = pd.DataFrame(rows)
    steps.append(Step(
        title="15. Deformaciones y esfuerzos en las esquinas (nodos)",
        description=(
            "Los esfuerzos se evalúan directamente en cada esquina del "
            "elemento usando B(ξ, η) en las coordenadas naturales del nodo:\n"
            "  N1 = (+1, +1)   N2 = (−1, +1)   N3 = (−1, −1)   N4 = (+1, −1)\n"
            "Luego  ε = B · u^e   y   σ = D · ε.\n"
            "Las matrices B(esquina) se muestran abajo para verificación "
            "manual: ε = B · u^e seguido de σ = D · ε reproduce los valores "
            "de la tabla.\n"
            "Nota: en MEF estándar los esfuerzos más precisos están en los "
            "puntos de Gauss; los valores en esquinas son válidos pero "
            "menos precisos."
        ),
        tables={"σ en las esquinas": df_corner},
        matrices=corner_matrices,
        plot_key="stress_corners",
    ))

    return Procedure(steps=steps, result=result)
