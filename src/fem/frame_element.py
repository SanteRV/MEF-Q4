"""Elemento FRAME tridimensional de 12 GDL (cap. 01.02.01).

Implementa el capítulo de elementos frame del documento teórico en su mismo
orden:

    01.02.01.02  Rotación de ejes locales alrededor del eje axial   (ec. 2.2.1)
    01.02.01.03  Matriz de transformación T = diag(r, r, r, r)      (ec. 2.3.1)
                 u_i = T·v_i   y   Q_i = T·F_i                      (ec. 2.3.2)
    01.02.01.04  Rigidez del SCL al SCG:  K = Tᵀ·k·T                (ec. 2.4.2)
    01.02.01.06  Matriz de rigidez k en el SCL, obtenida por el método de
                 los desplazamientos unitarios (ec. 2.6.1 a 2.6.5)
    01.02.01.07  Cargas de fijación (fuerzas de empotramiento perfecto)
    01.02.01.010 Recuperación de resultados: u = T·v, Q = k·u (ec. 2.10.1)
                 y los diagramas S(x) = S_nodos(x) + Σ S_carga

GRADOS DE LIBERTAD del SCL, en el orden del documento (figura 17):

    u1..u6   nodo inicial: ux, uy, uz, θx (torsión), θy, θz
    u7..u12  nodo final:   ux, uy, uz, θx, θy, θz

CONVENCIÓN DE EJES LOCALES. El documento define r por sus cosenos
directores (01.02.01.03) y da la rotación ψ alrededor del eje axial
(ec. 2.2.1), pero no fija de qué orientación se parte. Aquí se adopta el
criterio habitual y se deja explícito: el eje local x sigue el eje del
elemento; el eje local z se toma lo más vertical posible (referencia global
+Z), y el eje local y completa el triedro directo. Sobre esa base se aplica
la rotación ψ de la ec. 2.2.1. Para un elemento vertical la referencia se
cambia a +Y, porque el producto vectorial se degenera.
"""
from __future__ import annotations
from dataclasses import dataclass, field

import numpy as np

from .node_frame import NodeFrame


# Tolerancia para detectar un elemento paralelo a la referencia vertical.
_TOL_VERTICAL = 1e-8


@dataclass
class FrameLoad:
    """Una carga aplicada SOBRE el elemento, en coordenadas locales.

    Los tipos son los que enumera el documento en 01.02.01.010:

        "puntual"        carga transversal W a la distancia a, en el eje
                         local indicado por `eje` ("y" o "z")
        "momento"        momento puntual M a la distancia a, alrededor del
                         eje local perpendicular al plano de flexión
        "distribuida"    carga uniforme w en el eje local `eje`
        "axial_puntual"  carga puntual W a la distancia a, en el eje local x
        "axial_dist"     carga uniforme w en el eje local x
        "torsor"         momento torsor M_T a la distancia a
    """
    tipo: str
    valor: float                 # W, M, w o M_T según el tipo
    a: float = 0.0               # distancia desde el nodo inicial (m)
    eje: str = "y"               # "y" o "z" (ejes locales)


@dataclass
class FrameElement:
    """Elemento frame 3D: 2 nodos, 6 GDL por nodo (12 GDL)."""

    id: int
    nodes: list[NodeFrame]       # [nodo inicial, nodo final]
    E: float                     # módulo de elasticidad (Pa)
    G: float                     # módulo de corte (Pa)
    A: float                     # área de la sección (m²)
    Iy: float                    # inercia respecto al eje local y (m⁴)
    Iz: float                    # inercia respecto al eje local z (m⁴)
    J: float                     # inercia polar / constante torsional (m⁴)
    psi: float = 0.0             # rotación de ejes locales, en radianes (ec. 2.2.1)
    loads: list[FrameLoad] = field(default_factory=list)

    def __post_init__(self) -> None:
        if len(self.nodes) != 2:
            raise ValueError("FrameElement requiere exactamente 2 nodos.")
        if self.length <= 0.0:
            raise ValueError("Los dos nodos del elemento coinciden (L = 0).")

    # ---------- geometría ----------
    @property
    def length(self) -> float:
        """Longitud L del elemento."""
        a = np.array(self.nodes[0].coords)
        b = np.array(self.nodes[1].coords)
        return float(np.linalg.norm(b - a))

    def global_dofs(self) -> list[int]:
        """Los 12 GDL globales: 6 del nodo inicial + 6 del nodo final."""
        dofs: list[int] = []
        for n in self.nodes:
            dofs.extend(n.dofs)
        return dofs

    # ---------- 01.02.01.02 y .03 — ejes locales y transformación ----------
    def rotation_matrix(self) -> np.ndarray:
        """Matriz r (3×3) de cosenos directores del SCG al SCL (ec. 2.3.1).

        Sus filas son los ejes locales expresados en el sistema global:
        fila 0 = eje axial x, fila 1 = eje y, fila 2 = eje z. Es decir,
        r[i][j] = cos θ entre el eje local i y el eje global j, que es
        exactamente la matriz que el documento escribe con cos θxX, cos θxY...
        """
        p1 = np.array(self.nodes[0].coords, dtype=float)
        p2 = np.array(self.nodes[1].coords, dtype=float)
        x_l = (p2 - p1) / self.length

        # Referencia para orientar el triedro: +Z salvo elemento vertical
        ref = np.array([0.0, 0.0, 1.0])
        if abs(float(np.dot(x_l, ref))) > 1.0 - _TOL_VERTICAL:
            ref = np.array([0.0, 1.0, 0.0])
        y_l = np.cross(ref, x_l)
        y_l /= np.linalg.norm(y_l)
        z_l = np.cross(x_l, y_l)

        # Rotación ψ alrededor del eje axial (ec. 2.2.1): los ejes x no
        # cambia y los dos perpendiculares giran entre sí.
        if self.psi != 0.0:
            c, s = np.cos(self.psi), np.sin(self.psi)
            y_rot = c * y_l + s * z_l
            z_rot = -s * y_l + c * z_l
            y_l, z_l = y_rot, z_rot

        return np.vstack([x_l, y_l, z_l])

    def transformation_matrix(self) -> np.ndarray:
        """Matriz T (12×12) del SCG al SCL: bloques r en la diagonal (ec. 2.3.1).

            T = | r 0 0 0 |
                | 0 r 0 0 |        u_i = T·v_i,  Q_i = T·F_i   (ec. 2.3.2)
                | 0 0 r 0 |
                | 0 0 0 r |
        """
        r = self.rotation_matrix()
        T = np.zeros((12, 12))
        for b in range(4):
            T[3 * b:3 * b + 3, 3 * b:3 * b + 3] = r
        return T

    # ---------- 01.02.01.06 — matriz de rigidez en el SCL ----------
    def stiffness_local(self) -> np.ndarray:
        """Matriz k (12×12) del elemento en el sistema local.

        Es la matriz que el documento arma columna a columna con el método
        de los desplazamientos unitarios: rigidez axial EA/L (u1), flexión
        en el plano x-y con Iz (u2 y u6), flexión en el plano x-z con Iy
        (u3 y u5) y torsión GJ/L (u4); los GDL del nodo final repiten los
        mismos valores cambiando signo y posición.
        """
        L = self.length
        E, G, A = self.E, self.G, self.A
        Iy, Iz, J = self.Iy, self.Iz, self.J

        EA_L = E * A / L
        GJ_L = G * J / L
        az1 = 12.0 * E * Iz / L ** 3
        az2 = 6.0 * E * Iz / L ** 2
        az3 = 4.0 * E * Iz / L
        az4 = 2.0 * E * Iz / L
        ay1 = 12.0 * E * Iy / L ** 3
        ay2 = 6.0 * E * Iy / L ** 2
        ay3 = 4.0 * E * Iy / L
        ay4 = 2.0 * E * Iy / L

        k = np.zeros((12, 12))
        # Axial (u1, u7)
        k[0, 0] = k[6, 6] = EA_L
        k[0, 6] = k[6, 0] = -EA_L
        # Torsión (u4, u10)
        k[3, 3] = k[9, 9] = GJ_L
        k[3, 9] = k[9, 3] = -GJ_L
        # Flexión en el plano x-y: GDL 2 (uy), 6 (θz), 8 (uy), 12 (θz)
        k[1, 1] = k[7, 7] = az1
        k[1, 7] = k[7, 1] = -az1
        k[1, 5] = k[5, 1] = az2
        k[1, 11] = k[11, 1] = az2
        k[5, 7] = k[7, 5] = -az2
        k[7, 11] = k[11, 7] = -az2
        k[5, 5] = k[11, 11] = az3
        k[5, 11] = k[11, 5] = az4
        # Flexión en el plano x-z: GDL 3 (uz), 5 (θy), 9 (uz), 11 (θy)
        k[2, 2] = k[8, 8] = ay1
        k[2, 8] = k[8, 2] = -ay1
        k[2, 4] = k[4, 2] = -ay2
        k[2, 10] = k[10, 2] = -ay2
        k[4, 8] = k[8, 4] = ay2
        k[8, 10] = k[10, 8] = ay2
        k[4, 4] = k[10, 10] = ay3
        k[4, 10] = k[10, 4] = ay4
        return k

    # ---------- 01.02.01.04 — rigidez en el SCG ----------
    def stiffness_matrix(self) -> np.ndarray:
        """K (12×12) del elemento en el sistema global: K = Tᵀ·k·T (ec. 2.4.2)."""
        T = self.transformation_matrix()
        return T.T @ self.stiffness_local() @ T

    # ---------- 01.02.01.07 — cargas de fijación ----------
    def fixed_end_forces(self) -> np.ndarray:
        """Vector Q_f (12,) de fuerzas de empotramiento perfecto, en el SCL.

        Son las fuerzas que los apoyos aplican SOBRE el elemento cuando
        este está completamente restringido en ambos extremos, que es la
        definición del documento en 01.02.01.07. Con ellas:

            carga nodal equivalente = -Tᵀ·Q_f
            fuerzas de extremo reales  Q = k·u + Q_f      (ec. 2.10.1)
        """
        L = self.length
        Qf = np.zeros(12)
        for c in self.loads:
            Qf += self._fixed_end_of(c, L)
        return Qf

    @staticmethod
    def _fixed_end_of(c: FrameLoad, L: float) -> np.ndarray:
        """Fuerzas de empotramiento perfecto de UNA carga (ver figura 33)."""
        Qf = np.zeros(12)
        a = float(np.clip(c.a, 0.0, L))
        b = L - a

        if c.tipo == "distribuida":
            w = c.valor
            V = -w * L / 2.0
            M = w * L * L / 12.0
            if c.eje == "y":
                # Flexión en x-y: cortante en uy, momento alrededor de z
                Qf[1] = Qf[7] = V
                Qf[5], Qf[11] = -M, +M
            else:
                # Flexión en x-z: el momento alrededor de y tiene signo
                # opuesto porque θy positivo gira de +z hacia +x
                Qf[2] = Qf[8] = V
                Qf[4], Qf[10] = +M, -M

        elif c.tipo == "puntual":
            W = c.valor
            Vi = -W * b * b * (3.0 * a + b) / L ** 3
            Vj = -W * a * a * (a + 3.0 * b) / L ** 3
            Mi = W * a * b * b / L ** 2
            Mj = -W * a * a * b / L ** 2
            if c.eje == "y":
                Qf[1], Qf[7] = Vi, Vj
                Qf[5], Qf[11] = -Mi, -Mj
            else:
                Qf[2], Qf[8] = Vi, Vj
                Qf[4], Qf[10] = +Mi, +Mj

        elif c.tipo == "momento":
            M0 = c.valor
            Mi = -M0 * b * (3.0 * a - L) / L ** 2
            Mj = -M0 * a * (3.0 * b - L) / L ** 2
            R = 6.0 * M0 * a * b / L ** 3
            if c.eje == "y":
                # Momento alrededor de z, flexión en el plano x-y
                Qf[5], Qf[11] = Mi, Mj
                Qf[1], Qf[7] = R, -R
            else:
                Qf[4], Qf[10] = -Mi, -Mj
                Qf[2], Qf[8] = R, -R

        elif c.tipo == "axial_puntual":
            W = c.valor
            Qf[0] = -W * b / L
            Qf[6] = -W * a / L

        elif c.tipo == "axial_dist":
            w = c.valor
            Qf[0] = Qf[6] = -w * L / 2.0

        elif c.tipo == "torsor":
            MT = c.valor
            Qf[3] = -MT * b / L
            Qf[9] = -MT * a / L

        else:
            raise ValueError(f"Tipo de carga desconocido: {c.tipo!r}")

        return Qf

    def equivalent_nodal_loads(self) -> np.ndarray:
        """Carga nodal equivalente en el SCG: -Tᵀ·Q_f.

        Es lo que entra al vector global de fuerzas para que la solución
        del sistema reproduzca el efecto de las cargas sobre el elemento.
        """
        T = self.transformation_matrix()
        return -T.T @ self.fixed_end_forces()

    # ---------- 01.02.01.010 — recuperación de resultados ----------
    def local_displacements(self, v_global_12: np.ndarray) -> np.ndarray:
        """u = T·v (ec. 2.3.2): desplazamientos del elemento en el SCL."""
        return self.transformation_matrix() @ np.asarray(v_global_12, dtype=float)

    def end_forces(self, v_global_12: np.ndarray) -> np.ndarray:
        """Q = k·u + Q_f (ec. 2.10.1): fuerzas de extremo en el SCL.

        El término Q_f devuelve el efecto de las cargas aplicadas sobre el
        propio elemento, que la solución nodal por sí sola no contiene.
        """
        u = self.local_displacements(v_global_12)
        return self.stiffness_local() @ u + self.fixed_end_forces()

    def internal_forces_at(self, x: float,
                           v_global_12: np.ndarray) -> dict[str, float]:
        """Fuerzas internas a la distancia x del nodo inicial.

        Aplica la superposición que plantea el documento al final del
        capítulo:  S(x) = S_nodos(x) + Σ S_carga, con el equilibrio del
        tramo [0, x] del elemento.

        Devuelve P (axial), Vy, Vz (cortantes), T (torsor), My y Mz
        (flectores), todo en el sistema local.
        """
        Q = self.end_forces(v_global_12)
        x = float(np.clip(x, 0.0, self.length))

        # Aporte de las fuerzas del nodo inicial sobre el tramo [0, x]
        P = -Q[0]
        Vy = -Q[1]
        Vz = -Q[2]
        Tx = -Q[3]
        My = -Q[4] + Q[2] * x
        Mz = -Q[5] - Q[1] * x

        # Aporte de las cargas aplicadas dentro del tramo [0, x]
        for c in self.loads:
            a = float(np.clip(c.a, 0.0, self.length))
            if c.tipo == "distribuida":
                w = c.valor
                if c.eje == "y":
                    Vy -= w * x
                    Mz -= w * x * x / 2.0
                else:
                    Vz -= w * x
                    My += w * x * x / 2.0
            elif c.tipo == "puntual" and x >= a:
                W = c.valor
                if c.eje == "y":
                    Vy -= W
                    Mz -= W * (x - a)
                else:
                    Vz -= W
                    My += W * (x - a)
            elif c.tipo == "momento" and x >= a:
                if c.eje == "y":
                    Mz -= c.valor
                else:
                    My += c.valor
            elif c.tipo == "axial_puntual" and x >= a:
                P -= c.valor
            elif c.tipo == "axial_dist":
                P -= c.valor * x
            elif c.tipo == "torsor" and x >= a:
                Tx -= c.valor

        return {"P": P, "Vy": Vy, "Vz": Vz, "T": Tx, "My": My, "Mz": Mz}

    def diagram(self, componente: str, v_global_12: np.ndarray,
                n_puntos: int = 51) -> tuple[np.ndarray, np.ndarray]:
        """Diagrama de una componente a lo largo del elemento.

        Devuelve (x, valores). Las estaciones incluyen siempre los puntos
        de aplicación de las cargas puntuales, donde el diagrama salta.
        """
        L = self.length
        xs = set(np.linspace(0.0, L, max(n_puntos, 2)).tolist())
        for c in self.loads:
            if c.tipo in ("puntual", "momento", "axial_puntual", "torsor"):
                a = float(np.clip(c.a, 0.0, L))
                xs.add(max(a - 1e-9, 0.0))
                xs.add(min(a + 1e-9, L))
        orden = np.array(sorted(xs))
        vals = np.array([self.internal_forces_at(x, v_global_12)[componente]
                         for x in orden])
        return orden, vals
