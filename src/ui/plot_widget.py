"""Widget de Matplotlib embebido en Qt — gráficas para Q4."""
from __future__ import annotations
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.patches import Polygon
from matplotlib.tri import Triangulation
from PySide6.QtWidgets import QWidget, QVBoxLayout

from ..fem.structure import Structure
from ..fem.solver import FEMResult
from ..fem.q4_element import GAUSS_2X2, shape_functions


# Campos de esfuerzo disponibles para mapa de contornos
STRESS_FIELDS = {
    "σx":         lambda sig: sig[0],
    "σy":         lambda sig: sig[1],
    "τxy":        lambda sig: sig[2],
    "σ_VM":       lambda sig: float(np.sqrt(sig[0] ** 2 - sig[0] * sig[1]
                                            + sig[1] ** 2 + 3 * sig[2] ** 2)),
    "σ1 (princ)": lambda sig: 0.5 * (sig[0] + sig[1]) + np.sqrt(
        (0.5 * (sig[0] - sig[1])) ** 2 + sig[2] ** 2),
    "σ2 (princ)": lambda sig: 0.5 * (sig[0] + sig[1]) - np.sqrt(
        (0.5 * (sig[0] - sig[1])) ** 2 + sig[2] ** 2),
}


class PlotWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(6, 4), tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)

    def clear(self):
        self.figure.clear()
        self.canvas.draw_idle()

    def _new_axes(self, title: str):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        return ax

    def _draw_element(self, ax, el, facecolor="#cfe2f3", edgecolor="#1f77b4",
                      alpha=0.6, lw=2):
        xy = [(n.x, n.y) for n in el.nodes]
        poly = Polygon(xy, closed=True, facecolor=facecolor,
                       edgecolor=edgecolor, alpha=alpha, linewidth=lw)
        ax.add_patch(poly)

    def plot_mesh(self, structure: Structure):
        ax = self._new_axes("Geometría del elemento Q4")
        for el in structure.elements:
            self._draw_element(ax, el)
            cx = np.mean([n.x for n in el.nodes])
            cy = np.mean([n.y for n in el.nodes])
            ax.annotate(f"E{el.id + 1}", (cx, cy),
                        color="#1f77b4", fontsize=10,
                        ha="center", va="center", fontweight="bold")
        for n in structure.nodes:
            ax.plot(n.x, n.y, "o", color="black", markersize=7)
            ax.annotate(f"N{n.id + 1}", (n.x, n.y),
                        textcoords="offset points", xytext=(8, 8),
                        fontsize=10, fontweight="bold")
        self._fit_view(ax, structure)
        self.canvas.draw_idle()

    def plot_boundary_conditions(self, structure: Structure):
        ax = self._new_axes("Apoyos y cargas")
        for el in structure.elements:
            self._draw_element(ax, el, facecolor="#eeeeee",
                               edgecolor="#aaaaaa", alpha=0.4)
        for n in structure.nodes:
            ax.plot(n.x, n.y, "o", color="black", markersize=6)
            ax.annotate(f"N{n.id + 1}", (n.x, n.y),
                        textcoords="offset points", xytext=(8, 8), fontsize=9)
            if n.restraint_x and n.restraint_y:
                ax.plot(n.x, n.y, "s", color="green", markersize=18,
                        markerfacecolor="none", markeredgewidth=2)
            elif n.restraint_x or n.restraint_y:
                ax.plot(n.x, n.y, "^", color="green", markersize=15,
                        markerfacecolor="none", markeredgewidth=2)
            if n.load_x or n.load_y:
                # Vector de carga
                size = self._char_size(structure) * 0.5
                norm = np.hypot(n.load_x, n.load_y) or 1.0
                dx = (n.load_x / norm) * size
                dy = (n.load_y / norm) * size
                ax.annotate(
                    "", xy=(n.x + dx, n.y + dy), xytext=(n.x, n.y),
                    arrowprops=dict(arrowstyle="->", color="red", lw=2))
                ax.annotate(f"F=({n.load_x:.0f},{n.load_y:.0f}) N",
                            (n.x + dx, n.y + dy),
                            fontsize=8, color="red")
        self._fit_view(ax, structure)
        self.canvas.draw_idle()

    def plot_deformed(self, structure: Structure, result: FEMResult,
                      scale: float | None = None):
        """Si `scale` es None se calcula automáticamente para visualización."""
        ax = self._new_axes("Configuración deformada")
        max_disp = float(np.max(np.abs(result.displacements))) or 1.0
        size = self._char_size(structure)
        if scale is None:
            scale = 0.15 * size / max_disp if max_disp > 0 else 1.0

        for el in structure.elements:
            self._draw_element(ax, el, facecolor="none", edgecolor="#aaa",
                               alpha=1.0, lw=1)
            # Polígono deformado
            xy_def = []
            for n in el.nodes:
                ux = result.displacements[n.dofs[0]]
                uy = result.displacements[n.dofs[1]]
                xy_def.append((n.x + scale * ux, n.y + scale * uy))
            poly = Polygon(xy_def, closed=True, facecolor="#f4cccc",
                           edgecolor="#d62728", alpha=0.6, linewidth=2)
            ax.add_patch(poly)

        for n in structure.nodes:
            ux = result.displacements[n.dofs[0]]
            uy = result.displacements[n.dofs[1]]
            ax.plot(n.x, n.y, "o", color="#aaa", markersize=4)
            ax.plot(n.x + scale * ux, n.y + scale * uy,
                    "o", color="#d62728", markersize=6)

        ax.set_title(f"Configuración deformada (escala x{scale:.1f})")
        self._fit_view(ax, structure, margin=size * 0.25)
        self.canvas.draw_idle()

    def plot_stress(self, structure: Structure, result: FEMResult):
        ax = self._new_axes("Esfuerzo σx en los puntos de Gauss")
        for el, el_res in zip(structure.elements, result.elements):
            self._draw_element(ax, el, facecolor="#eeeeee",
                               edgecolor="#888888", alpha=0.4)
            for (xi, eta, _), sigma in zip(GAUSS_2X2, el_res.stresses_at_gauss):
                # Posición global del punto de Gauss
                N = shape_functions(xi, eta)
                x_gp = sum(N[i] * el.nodes[i].x for i in range(4))
                y_gp = sum(N[i] * el.nodes[i].y for i in range(4))
                sx = sigma[0]
                color = "#1f77b4" if sx >= 0 else "#d62728"
                ax.plot(x_gp, y_gp, "o", color=color, markersize=10)
                ax.annotate(f"{sx:.2e}", (x_gp, y_gp),
                            textcoords="offset points", xytext=(6, 6),
                            fontsize=8, color=color)
        for n in structure.nodes:
            ax.plot(n.x, n.y, "ks", markersize=5)
        self._fit_view(ax, structure)
        self.canvas.draw_idle()

    # ---------- utilitarios ----------
    def _char_size(self, structure: Structure) -> float:
        xs = [n.x for n in structure.nodes]
        ys = [n.y for n in structure.nodes]
        if not xs:
            return 1.0
        return max(max(xs) - min(xs), max(ys) - min(ys)) or 1.0

    def _fit_view(self, ax, structure: Structure, margin: float | None = None):
        xs = [n.x for n in structure.nodes]
        ys = [n.y for n in structure.nodes]
        if not xs:
            return
        if margin is None:
            margin = self._char_size(structure) * 0.2
        ax.set_xlim(min(xs) - margin, max(xs) + margin)
        ax.set_ylim(min(ys) - margin, max(ys) + margin)

    def plot_stress_contour(self, structure: Structure, result: FEMResult,
                            field: str = "σ_VM", n_subdiv: int = 20,
                            show_isolines: bool = True,
                            show_gp_values: bool = True):
        """Mapa de contornos continuo de un campo de esfuerzos.

        Evalúa el campo en una grilla densa dentro de cada elemento usando
        las shape functions del Q4, lo dibuja como tricontourf y añade
        isolíneas sobre el colormap para resaltar niveles.

        Args:
            field: nombre del campo (clave en STRESS_FIELDS)
            n_subdiv: resolución por dimensión del elemento (20 -> 400 puntos)
            show_isolines: si dibujar líneas de contorno blancas sobre el colormap
            show_gp_values: si anotar los valores numéricos en los Gauss points
        """
        if field not in STRESS_FIELDS:
            field = "σ_VM"
        getter = STRESS_FIELDS[field]
        ax = self._new_axes(f"Mapa de contornos — {field}")

        xs_all, ys_all, vals_all, tris = [], [], [], []
        for el, el_res in zip(structure.elements, result.elements):
            dofs = el.global_dofs()
            u_e = result.displacements[dofs]
            n = n_subdiv
            xi_lin = np.linspace(-1, 1, n)
            eta_lin = np.linspace(-1, 1, n)
            XI, ETA = np.meshgrid(xi_lin, eta_lin)
            xi_pts = XI.flatten()
            eta_pts = ETA.flatten()
            base = len(xs_all)
            for xi, eta in zip(xi_pts, eta_pts):
                N = shape_functions(float(xi), float(eta))
                x_pt = sum(N[i] * el.nodes[i].x for i in range(4))
                y_pt = sum(N[i] * el.nodes[i].y for i in range(4))
                _, sig = el.strains_stresses_at(float(xi), float(eta), u_e)
                xs_all.append(x_pt)
                ys_all.append(y_pt)
                vals_all.append(getter(sig))
            for j in range(n - 1):
                for i in range(n - 1):
                    idx00 = base + j * n + i
                    idx10 = base + j * n + i + 1
                    idx01 = base + (j + 1) * n + i
                    idx11 = base + (j + 1) * n + i + 1
                    tris.append([idx00, idx10, idx11])
                    tris.append([idx00, idx11, idx01])

        xs_arr = np.array(xs_all)
        ys_arr = np.array(ys_all)
        vals_arr = np.array(vals_all)
        triang = Triangulation(xs_arr, ys_arr, triangles=np.array(tris))

        # Contornos rellenos con paleta científica
        n_levels = 21
        cf = ax.tricontourf(triang, vals_arr, levels=n_levels, cmap="RdYlBu_r")
        cbar = self.figure.colorbar(cf, ax=ax, shrink=0.85, pad=0.02)
        cbar.set_label(f"{field} (Pa)", fontsize=10, labelpad=8)
        cbar.ax.tick_params(labelsize=8)

        # Isolíneas blancas sobre el colormap
        if show_isolines:
            ax.tricontour(triang, vals_arr, levels=10, colors="white",
                          linewidths=0.6, alpha=0.6)

        # Bordes de elementos
        for el in structure.elements:
            xy = [(n.x, n.y) for n in el.nodes]
            poly = Polygon(xy, closed=True, facecolor="none",
                           edgecolor="#222", linewidth=1.4)
            ax.add_patch(poly)

        # Nodos
        for n in structure.nodes:
            ax.plot(n.x, n.y, "o", color="black", markersize=5,
                    markeredgecolor="white", markeredgewidth=0.5)

        # Valores en GPs (opcional)
        if show_gp_values:
            for el, el_res in zip(structure.elements, result.elements):
                for (xi, eta, _), sigma in zip(GAUSS_2X2, el_res.stresses_at_gauss):
                    N = shape_functions(xi, eta)
                    x_gp = sum(N[i] * el.nodes[i].x for i in range(4))
                    y_gp = sum(N[i] * el.nodes[i].y for i in range(4))
                    val = getter(sigma)
                    ax.annotate(
                        f"{val:.3g}", (x_gp, y_gp),
                        textcoords="offset points", xytext=(4, 4),
                        fontsize=7, color="black",
                        bbox=dict(boxstyle="round,pad=0.15",
                                  facecolor="white", edgecolor="none",
                                  alpha=0.7),
                    )

        self._fit_view(ax, structure)
        self.canvas.draw_idle()

    def plot_principal_stresses(self, structure: Structure, result: FEMResult,
                                arrow_scale: float | None = None):
        """Dibuja flechas de esfuerzos principales sigma1 y sigma2 en cada GP.

        Convención:
          - Rojo: tracción (sigma > 0)
          - Azul: compresión (sigma < 0)
          - Longitud proporcional a |sigma|
          - Orientadas según el ángulo principal theta_p
          - sigma1 y sigma2 son perpendiculares entre sí
        """
        ax = self._new_axes("Esfuerzos principales sigma1 y sigma2")

        # Calcular máximo absoluto para normalizar longitudes
        all_principals = []
        for el_res in result.elements:
            for sig in el_res.stresses_at_gauss:
                sx, sy, txy = sig[0], sig[1], sig[2]
                R = np.sqrt(((sx - sy) / 2) ** 2 + txy ** 2)
                s_avg = (sx + sy) / 2
                all_principals.extend([abs(s_avg + R), abs(s_avg - R)])
        max_sigma = max(all_principals) if all_principals else 1.0

        # Tamaño visual: ~10% del tamaño del elemento
        size = self._char_size(structure)
        if arrow_scale is None:
            arrow_scale = size * 0.10 / max_sigma if max_sigma > 0 else 1.0

        # Dibujar contornos de los elementos como referencia
        for el in structure.elements:
            self._draw_element(ax, el, facecolor="#fafafa",
                               edgecolor="#888", alpha=0.5, lw=1)
            for n in el.nodes:
                ax.plot(n.x, n.y, "ks", markersize=4)

        # En cada GP de cada elemento dibujar las dos flechas
        for el, el_res in zip(structure.elements, result.elements):
            for (xi, eta, _), sigma in zip(GAUSS_2X2, el_res.stresses_at_gauss):
                sx, sy, txy = sigma[0], sigma[1], sigma[2]
                # Esfuerzos principales y ángulo
                s_avg = (sx + sy) / 2
                R = np.sqrt(((sx - sy) / 2) ** 2 + txy ** 2)
                sigma_1 = s_avg + R
                sigma_2 = s_avg - R
                theta_p = 0.5 * np.arctan2(2 * txy, sx - sy)
                # Posición física del GP
                N = shape_functions(xi, eta)
                x_gp = sum(N[i] * el.nodes[i].x for i in range(4))
                y_gp = sum(N[i] * el.nodes[i].y for i in range(4))

                # Flecha de sigma1: dirección theta_p
                self._draw_principal_arrow(
                    ax, x_gp, y_gp, sigma_1, theta_p, arrow_scale
                )
                # Flecha de sigma2: dirección perpendicular (theta_p + 90°)
                self._draw_principal_arrow(
                    ax, x_gp, y_gp, sigma_2, theta_p + np.pi / 2, arrow_scale
                )

        # Leyenda manual
        from matplotlib.lines import Line2D
        legend_elems = [
            Line2D([0], [0], color="#d62728", lw=3, label="Tracción (σ > 0)"),
            Line2D([0], [0], color="#1f77b4", lw=3, label="Compresión (σ < 0)"),
        ]
        ax.legend(handles=legend_elems, loc="upper right", framealpha=0.9,
                  fontsize=8)

        self._fit_view(ax, structure, margin=size * 0.25)
        self.canvas.draw_idle()

    def _draw_principal_arrow(self, ax, x_gp: float, y_gp: float,
                              sigma: float, theta: float,
                              arrow_scale: float) -> None:
        """Dibuja una flecha doble (en ambas direcciones) para representar sigma.

        - sigma > 0 (tracción): flechas hacia afuera, color rojo
        - sigma < 0 (compresión): flechas hacia adentro, color azul
        """
        magnitude = abs(sigma) * arrow_scale
        if magnitude < 1e-12:
            return
        dx = np.cos(theta) * magnitude
        dy = np.sin(theta) * magnitude
        if sigma >= 0:
            # Tracción: flechas apuntando hacia afuera desde el GP
            color = "#d62728"
            ax.annotate("", xy=(x_gp + dx, y_gp + dy), xytext=(x_gp, y_gp),
                        arrowprops=dict(arrowstyle="->", color=color, lw=2),
                        zorder=5)
            ax.annotate("", xy=(x_gp - dx, y_gp - dy), xytext=(x_gp, y_gp),
                        arrowprops=dict(arrowstyle="->", color=color, lw=2),
                        zorder=5)
        else:
            # Compresión: flechas apuntando hacia el GP desde afuera
            color = "#1f77b4"
            ax.annotate("", xy=(x_gp, y_gp), xytext=(x_gp + dx, y_gp + dy),
                        arrowprops=dict(arrowstyle="->", color=color, lw=2),
                        zorder=5)
            ax.annotate("", xy=(x_gp, y_gp), xytext=(x_gp - dx, y_gp - dy),
                        arrowprops=dict(arrowstyle="->", color=color, lw=2),
                        zorder=5)
        # Marcador en el GP
        ax.plot(x_gp, y_gp, "o", color="black", markersize=3, zorder=4)

    def plot_stress_corners(self, structure: Structure, result: FEMResult):
        ax = self._new_axes("Esfuerzo σx en las esquinas (nodos)")
        for el, el_res in zip(structure.elements, result.elements):
            self._draw_element(ax, el, facecolor="#eeeeee",
                               edgecolor="#888888", alpha=0.4)
            for node, sigma in zip(el.nodes, el_res.stresses_at_corners):
                sx = sigma[0]
                color = "#1f77b4" if sx >= 0 else "#d62728"
                ax.plot(node.x, node.y, "s", color=color, markersize=12)
                ax.annotate(f"{sx:.2f}", (node.x, node.y),
                            textcoords="offset points", xytext=(8, 8),
                            fontsize=9, color=color, fontweight="bold")
        self._fit_view(ax, structure)
        self.canvas.draw_idle()

    def plot_shape_functions_diagram(self, structure: Structure | None = None):
        """Diagrama 3D de las 4 funciones de forma N1..N4 como superficies.

        Cada Ni es Ni(ξ, η) = ¼(1 + ξ·ξi)(1 + η·ηi) y vale 1 en su nodo y 0
        en los otros 3 nodos. Se muestran las 4 superficies en una grilla 2×2
        — esto deja ver por qué N1..N4 son una "partición de la unidad" sobre
        el elemento patrón.
        """
        self.figure.clear()
        # Grilla 2x2 de subplots 3D
        coords = [(+1, +1, "N1 (++)"), (-1, +1, "N2 (-+)"),
                  (-1, -1, "N3 (--)"), (+1, -1, "N4 (+-)")]
        # Malla de evaluacion en [-1,1]^2
        n_grid = 24
        xi = np.linspace(-1, 1, n_grid)
        eta = np.linspace(-1, 1, n_grid)
        Xi, Eta = np.meshgrid(xi, eta)
        for i, (xi_i, eta_i, name) in enumerate(coords):
            ax = self.figure.add_subplot(2, 2, i + 1, projection="3d")
            N = 0.25 * (1 + Xi * xi_i) * (1 + Eta * eta_i)
            ax.plot_surface(Xi, Eta, N, cmap="viridis", alpha=0.85,
                            edgecolor="none", antialiased=True)
            # Wireframe encima para que se vea estructura
            ax.plot_wireframe(Xi, Eta, N, color="black", linewidth=0.25, alpha=0.4)
            # Marcar el nodo de la funcion (donde vale 1) con un punto rojo
            ax.scatter([xi_i], [eta_i], [1.0], color="red", s=60, zorder=10,
                       depthshade=False)
            # Marcar los otros 3 nodos (donde vale 0)
            for xj, ej, _ in coords:
                if (xj, ej) != (xi_i, eta_i):
                    ax.scatter([xj], [ej], [0.0], color="gray", s=30,
                               depthshade=False, alpha=0.6)
            ax.set_title(name, fontsize=10)
            ax.set_xlabel("ξ", fontsize=8); ax.set_ylabel("η", fontsize=8)
            ax.set_zlabel("Ni", fontsize=8)
            ax.set_zlim(0, 1)
            ax.view_init(elev=24, azim=-55)
            ax.tick_params(labelsize=7)
        self.figure.suptitle("Funciones de forma N1..N4 — superficies bilineales sobre el elemento patrón",
                             fontsize=11)
        self.canvas.draw_idle()
        return

        # codigo 2D anterior (deshabilitado, dejado por si se quiere volver)
        ax = self._new_axes("Elemento Q4 en coordenadas naturales (ξ, η)")
        # Cuadrado patrón
        poly = Polygon([(1, 1), (-1, 1), (-1, -1), (1, -1)],
                       closed=True, facecolor="#cfe2f3",
                       edgecolor="#1f77b4", linewidth=2, alpha=0.6)
        ax.add_patch(poly)
        nodes = [
            (1, 1, "N₁", "(+1, +1)"),
            (-1, 1, "N₂", "(−1, +1)"),
            (-1, -1, "N₃", "(−1, −1)"),
            (1, -1, "N₄", "(+1, −1)"),
        ]
        for xi, eta, name, coord in nodes:
            ax.plot(xi, eta, "o", color="black", markersize=10, zorder=5)
            dx, dy = 0.12 * np.sign(xi) if xi != 0 else 0.12, 0.12 * np.sign(eta) if eta != 0 else 0.12
            ax.annotate(
                f"{name} {coord}", (xi, eta),
                xytext=(xi + dx, eta + dy),
                fontsize=11, fontweight="bold", color="#1F4E78",
                ha="center", va="center",
            )
        # Ejes y origen
        ax.annotate("", xy=(1.55, 0), xytext=(-1.55, 0),
                    arrowprops=dict(arrowstyle="->", color="#888", lw=1.2))
        ax.annotate("ξ", (1.58, -0.08), fontsize=14, color="#444")
        ax.annotate("", xy=(0, 1.55), xytext=(0, -1.55),
                    arrowprops=dict(arrowstyle="->", color="#888", lw=1.2))
        ax.annotate("η", (0.08, 1.58), fontsize=14, color="#444")
        ax.plot(0, 0, "x", color="red", markersize=8, zorder=5)
        ax.annotate("origen", (0.05, -0.1), color="red", fontsize=9)
        # Flecha curva CCW indicando orden
        from matplotlib.patches import FancyArrowPatch
        arrow = FancyArrowPatch(
            (0.7, 0.85), (-0.7, 0.85),
            connectionstyle="arc3,rad=0.3",
            arrowstyle="->", color="#ff9800", lw=2, mutation_scale=15, zorder=4,
        )
        ax.add_patch(arrow)
        ax.annotate("Orden CCW", (0, 1.05), color="#ff9800",
                    fontsize=9, ha="center", fontweight="bold")
        ax.set_xlim(-1.85, 1.85); ax.set_ylim(-1.85, 1.85)
        ax.set_xlabel("ξ"); ax.set_ylabel("η")
        ax.grid(True, alpha=0.3)
        ax.set_title("Elemento Q4 patrón — convención ++, -+, --, +-")
        self.canvas.draw_idle()

    def plot_gauss_points_diagram(self, structure: Structure | None = None):
        """Diagrama 3D de los 4 puntos de Gauss en el elemento patrón.

        Vista en perspectiva del cuadrado natural [-1,1]^2 (z=0) con los
        4 GPs flotando levemente arriba para que se distingan, los nodos
        N1..N4 marcados como esquinas y las lineas de cuadratura ξ=±1/√3,
        η=±1/√3 dibujadas sobre el plano.
        """
        self.figure.clear()
        ax = self.figure.add_subplot(111, projection="3d")

        # Cuadrado patron (z=0) como superficie
        square_x = [[-1, 1], [-1, 1]]
        square_y = [[-1, -1], [1, 1]]
        square_z = [[0, 0], [0, 0]]
        ax.plot_surface(np.array(square_x), np.array(square_y), np.array(square_z),
                        color="#cfe2f3", alpha=0.4, edgecolor="#1f77b4",
                        linewidth=1)

        # Aristas del cuadrado patron
        corners = [(1, 1), (-1, 1), (-1, -1), (1, -1), (1, 1)]
        cx = [c[0] for c in corners]
        cy = [c[1] for c in corners]
        cz = [0.0] * len(corners)
        ax.plot(cx, cy, cz, color="#1f77b4", linewidth=2)

        # Nodos N1..N4 como esferas grandes en las esquinas
        nodes = [(+1, +1, "N1 (++)"), (-1, +1, "N2 (-+)"),
                 (-1, -1, "N3 (--)"), (+1, -1, "N4 (+-)")]
        for xi, eta, name in nodes:
            ax.scatter([xi], [eta], [0], color="black", s=60, depthshade=False)
            ax.text(xi * 1.15, eta * 1.15, 0.05, name, fontsize=9,
                    color="#1F4E78", fontweight="bold")

        # 4 puntos de Gauss flotando ligeramente sobre z=0 para que se vean
        g = 1.0 / np.sqrt(3.0)
        z_gp = 0.18  # altura artificial para que destaquen sobre el plano
        gps = [(+g, +g, "GP1"), (-g, +g, "GP2"),
               (-g, -g, "GP3"), (+g, -g, "GP4")]
        for xi, eta, name in gps:
            # Linea vertical desde z=0 hasta el GP para anclar la posicion real
            ax.plot([xi, xi], [eta, eta], [0, z_gp], color="#d62728",
                    linewidth=1, linestyle=":")
            ax.scatter([xi], [eta], [z_gp], color="#d62728", s=120, marker="*",
                       depthshade=False, zorder=10)
            ax.text(xi, eta, z_gp + 0.08,
                    f"{name}\n({xi:+.3f}, {eta:+.3f})",
                    fontsize=8, color="#d62728", ha="center", fontweight="bold")

        # Lineas de cuadratura sobre el plano
        for c in (g, -g):
            ax.plot([c, c], [-1, 1], [0, 0], color="#ffbbbb",
                    linewidth=1, linestyle="--")
            ax.plot([-1, 1], [c, c], [0, 0], color="#ffbbbb",
                    linewidth=1, linestyle="--")

        # Origen
        ax.scatter([0], [0], [0], marker="x", color="red", s=40)

        ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.3); ax.set_zlim(0, 0.6)
        ax.set_xlabel("ξ", fontsize=9); ax.set_ylabel("η", fontsize=9)
        ax.set_zlabel("", fontsize=1)  # ocultar Z (es solo visual)
        ax.set_box_aspect((1, 1, 0.5))
        ax.view_init(elev=28, azim=-50)
        ax.set_title("4 puntos de Gauss en (±1/√3, ±1/√3) ≈ (±0.5774, ±0.5774), wi=1",
                     fontsize=10)
        self.canvas.draw_idle()
        return

        # codigo 2D anterior (deshabilitado, dejado por si se quiere volver)
        ax = self._new_axes("Puntos de Gauss 2×2")
        # Cuadrado patrón en gris claro
        poly = Polygon([(1, 1), (-1, 1), (-1, -1), (1, -1)],
                       closed=True, facecolor="#eeeeee",
                       edgecolor="#888", linewidth=1.5, alpha=0.7)
        ax.add_patch(poly)
        # Nodos como referencia
        for xi, eta, name in [(1, 1, "N₁"), (-1, 1, "N₂"),
                               (-1, -1, "N₃"), (1, -1, "N₄")]:
            ax.plot(xi, eta, "ko", markersize=5, alpha=0.4)
            ax.annotate(name, (xi, eta),
                        textcoords="offset points",
                        xytext=(12 * np.sign(xi), 12 * np.sign(eta)),
                        fontsize=9, color="#666", alpha=0.7)
        # 4 puntos de Gauss en (±1/√3, ±1/√3)
        g = 1 / np.sqrt(3)
        gps = [(+g, +g, "GP₁"), (-g, +g, "GP₂"),
               (-g, -g, "GP₃"), (+g, -g, "GP₄")]
        for xi, eta, name in gps:
            ax.plot(xi, eta, "*", color="#d62728", markersize=22, zorder=6)
            ax.annotate(
                f"{name}\n({xi:+.4f}, {eta:+.4f})",
                (xi, eta),
                textcoords="offset points", xytext=(15, 10),
                fontsize=9, fontweight="bold", color="#d62728",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="#d62728", alpha=0.9),
            )
        # Líneas de cuadratura
        ax.axvline(g, color="#ffcccc", lw=1, ls=":", zorder=1)
        ax.axvline(-g, color="#ffcccc", lw=1, ls=":", zorder=1)
        ax.axhline(g, color="#ffcccc", lw=1, ls=":", zorder=1)
        ax.axhline(-g, color="#ffcccc", lw=1, ls=":", zorder=1)
        ax.set_xlim(-1.4, 1.4); ax.set_ylim(-1.4, 1.4)
        ax.set_xlabel("ξ"); ax.set_ylabel("η")
        ax.grid(True, alpha=0.3)
        ax.set_title("4 puntos de Gauss en (±1/√3, ±1/√3) ≈ (±0.5774, ±0.5774), wᵢ = 1")
        self.canvas.draw_idle()

    def render_plot_key(self, key: str, structure: Structure,
                        result: FEMResult | None, scale: float | None = None,
                        contour_field: str = "σ_VM",
                        vis_mode: str = "Contornos"):
        if key == "mesh":
            self.plot_mesh(structure)
        elif key == "bc":
            self.plot_boundary_conditions(structure)
        elif key == "shape_functions_diagram":
            self.plot_shape_functions_diagram(structure)
        elif key == "gauss_points_diagram":
            self.plot_gauss_points_diagram(structure)
        elif key == "deformed" and result is not None:
            self.plot_deformed(structure, result, scale=scale)
        elif key == "stress" and result is not None:
            # vis_mode controla qué visualización se muestra
            if vis_mode == "Esfuerzos principales":
                self.plot_principal_stresses(structure, result)
            else:
                # "Contornos" (default)
                self.plot_stress_contour(structure, result, field=contour_field)
        elif key == "stress_corners" and result is not None:
            self.plot_stress_corners(structure, result)
        else:
            self.clear()
