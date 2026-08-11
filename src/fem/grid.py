"""Sistema de coordenadas (grillas) del modelo — Corrección 2, paso 2.

Para qué sirve: antes de dibujar, el usuario define las líneas de
referencia de su estructura. Si configura X = 1, 2 ; Y = 3, 4 ; Z = 0, 5,
el aplicativo dibuja esas líneas y el usuario dibuja sobre ellas, con
ajuste automático (snap) a sus intersecciones.

Es exactamente el flujo de un programa de análisis estructural: primero
la retícula de ejes, después los elementos apoyados en esa retícula.

El módulo es Python puro (sin Qt): la vista 3D solo lo consulta.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class GridSystem:
    """Líneas de referencia en X, Y y Z.

    Cada lista contiene las coordenadas donde hay una línea de grilla.
    Las intersecciones de esas líneas son los puntos de ajuste (snap)
    a los que se pegan los nodos al dibujar.
    """
    x: list[float] = field(default_factory=lambda: [0.0, 5.0])
    y: list[float] = field(default_factory=lambda: [0.0, 5.0])
    z: list[float] = field(default_factory=lambda: [0.0])
    visible: bool = True
    snap: bool = True

    # ------------------------------------------------------------ edición
    def set_axis(self, eje: str, valores: list[float]) -> None:
        """Redefine las líneas de un eje, ordenadas y sin duplicados."""
        eje = eje.lower()
        if eje not in ("x", "y", "z"):
            raise ValueError("El eje debe ser 'x', 'y' o 'z'.")
        limpio = sorted({round(float(v), 9) for v in valores})
        if not limpio:
            raise ValueError(f"El eje {eje.upper()} necesita al menos una línea.")
        setattr(self, eje, limpio)

    @staticmethod
    def parse_axis(texto: str) -> list[float]:
        """Convierte '0, 2.5, 5' o '0 2.5 5' en [0.0, 2.5, 5.0].

        Para qué sirve: la UI recibe las líneas como texto; esta función
        centraliza el parseo y da un error claro si algo no es número.
        """
        crudo = texto.replace(";", ",").replace("\t", " ")
        piezas = [p for p in crudo.replace(",", " ").split() if p]
        valores: list[float] = []
        for p in piezas:
            try:
                valores.append(float(p))
            except ValueError:
                raise ValueError(f"{p!r} no es un número válido.") from None
        if not valores:
            raise ValueError("No se indicó ninguna coordenada.")
        return sorted(set(valores))

    @classmethod
    def uniform(cls, nx: int, ny: int, nz: int,
                dx: float, dy: float, dz: float) -> "GridSystem":
        """Grilla regular: nx tramos de ancho dx, etc.

        Atajo típico para edificaciones: 3 vanos de 5 m en X, 2 de 4 m
        en Y y 4 pisos de 3 m en Z.
        """
        return cls(
            x=[i * dx for i in range(nx + 1)],
            y=[j * dy for j in range(ny + 1)],
            z=[k * dz for k in range(nz + 1)],
        )

    # ---------------------------------------------------------- consultas
    @property
    def bounds(self) -> tuple[float, float, float, float, float, float]:
        """Caja que envuelve la grilla: (xmin, xmax, ymin, ymax, zmin, zmax)."""
        return (min(self.x), max(self.x), min(self.y), max(self.y),
                min(self.z), max(self.z))

    @property
    def size(self) -> float:
        """Dimensión característica — sirve para escalar la cámara y el snap."""
        x0, x1, y0, y1, z0, z1 = self.bounds
        d = max(x1 - x0, y1 - y0, z1 - z0)
        return float(d) if d > 1e-9 else 1.0

    def intersections(self) -> np.ndarray:
        """Todos los puntos de intersección de la grilla, como array (N, 3).

        Son los candidatos de ajuste al dibujar.
        """
        pts = [(px, py, pz) for pz in self.z for py in self.y for px in self.x]
        return np.array(pts, dtype=float) if pts else np.zeros((0, 3))

    def lines(self) -> list[tuple[tuple[float, float, float],
                                  tuple[float, float, float]]]:
        """Segmentos (inicio, fin) de todas las líneas de la grilla.

        Para qué sirve: la vista 3D los dibuja tal cual. Por cada plano Z
        se traza la retícula X-Y, y en cada intersección X-Y se traza la
        línea vertical que une el Z menor con el mayor.
        """
        x0, x1, y0, y1, z0, z1 = self.bounds
        segs: list[tuple[tuple[float, float, float],
                         tuple[float, float, float]]] = []
        for pz in self.z:
            for px in self.x:
                segs.append(((px, y0, pz), (px, y1, pz)))
            for py in self.y:
                segs.append(((x0, py, pz), (x1, py, pz)))
        if len(self.z) > 1:
            for px in self.x:
                for py in self.y:
                    segs.append(((px, py, z0), (px, py, z1)))
        return segs

    def points_on_plane(self, kind: str, coord: float,
                        tol: float = 1e-9) -> np.ndarray:
        """Intersecciones que caen sobre un plano de trabajo dado.

        Para qué sirve: al dibujar en el plano "XY con Z = 0", solo deben
        poder capturar el clic las intersecciones de ESE plano; una que
        esté en Z = 3 no debe robarse el punto aunque en pantalla quede
        cerca.
        """
        pts = self.intersections()
        if pts.size == 0:
            return pts
        eje = {"yz": 0, "xz": 1, "xy": 2}[kind]
        return pts[np.abs(pts[:, eje] - coord) < tol]

    def snap_point(self, x: float, y: float, z: float,
                   tol: float | None = None) -> tuple[float, float, float]:
        """Ajusta un punto a la intersección de grilla más cercana.

        Si `snap` está desactivado, o si no hay ninguna intersección dentro
        de la tolerancia, devuelve el punto sin cambios. La tolerancia por
        defecto es el 5 % del tamaño de la grilla.
        """
        if not self.snap:
            return (x, y, z)
        pts = self.intersections()
        if pts.size == 0:
            return (x, y, z)
        if tol is None:
            tol = 0.05 * self.size
        d = np.linalg.norm(pts - np.array([x, y, z]), axis=1)
        i = int(np.argmin(d))
        if d[i] <= tol:
            p = pts[i]
            return (float(p[0]), float(p[1]), float(p[2]))
        return (x, y, z)

    def describe(self) -> str:
        """Resumen legible para la barra de estado."""
        def fmt(vals: list[float]) -> str:
            return ", ".join(f"{v:g}" for v in vals)
        return (f"X: {fmt(self.x)}  |  Y: {fmt(self.y)}  |  Z: {fmt(self.z)}"
                f"  ({len(self.intersections())} intersecciones)")
