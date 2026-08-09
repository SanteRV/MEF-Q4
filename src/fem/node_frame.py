"""Nodos para el modelo FRAME tridimensional (cap. 01.02).

Para qué sirve: en el elemento frame 3D cada nodo tiene 6 grados de
libertad (01.02.01.01): tres desplazamientos y tres giros.

    ux, uy, uz    desplazamientos en X, Y, Z del sistema global (SCG)
    rx, ry, rz    giros alrededor de X, Y, Z del SCG

El documento lo dice explícitamente en 01.02.01.08: "un empotramiento
restringe los seis grados de libertad del nodo, mientras que un apoyo
articulado restringe únicamente las traslaciones".

Familia paralela a node.py (2 GDL), node_plate.py (3) y node_shell.py (5).
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class NodeFrame:
    """Un nodo del modelo frame: posición 3D, apoyos y cargas nodales.

    El nodo i ocupa las posiciones 6i a 6i+5 del sistema global.
    """
    id: int
    x: float
    y: float
    z: float = 0.0

    # Restricciones por GDL: True = el desplazamiento es CONOCIDO
    restraint_ux: bool = False
    restraint_uy: bool = False
    restraint_uz: bool = False
    restraint_rx: bool = False
    restraint_ry: bool = False
    restraint_rz: bool = False

    # Valores impuestos en los GDL restringidos (vector v_c, ec. 2.9.1)
    prescribed_ux: float = 0.0
    prescribed_uy: float = 0.0
    prescribed_uz: float = 0.0
    prescribed_rx: float = 0.0
    prescribed_ry: float = 0.0
    prescribed_rz: float = 0.0

    # Cargas nodales en el SCG
    load_fx: float = 0.0
    load_fy: float = 0.0
    load_fz: float = 0.0
    load_mx: float = 0.0
    load_my: float = 0.0
    load_mz: float = 0.0

    @property
    def coords(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)

    @property
    def dofs(self) -> tuple[int, int, int, int, int, int]:
        """Índices globales de los 6 GDL (ux, uy, uz, rx, ry, rz)."""
        base = 6 * self.id
        return tuple(base + k for k in range(6))   # type: ignore[return-value]

    @property
    def restraints(self) -> tuple[bool, ...]:
        return (self.restraint_ux, self.restraint_uy, self.restraint_uz,
                self.restraint_rx, self.restraint_ry, self.restraint_rz)

    @property
    def loads(self) -> tuple[float, ...]:
        return (self.load_fx, self.load_fy, self.load_fz,
                self.load_mx, self.load_my, self.load_mz)

    @property
    def prescribed(self) -> tuple[float, ...]:
        return (self.prescribed_ux, self.prescribed_uy, self.prescribed_uz,
                self.prescribed_rx, self.prescribed_ry, self.prescribed_rz)

    # ---------- helpers de apoyo ----------
    def empotrar(self) -> None:
        """Empotramiento: restringe los 6 GDL (01.02.01.08)."""
        self.restraint_ux = self.restraint_uy = self.restraint_uz = True
        self.restraint_rx = self.restraint_ry = self.restraint_rz = True

    def articular(self) -> None:
        """Apoyo articulado: restringe solo las traslaciones (01.02.01.08)."""
        self.restraint_ux = self.restraint_uy = self.restraint_uz = True
