"""Secciones del modelo unificado (Corrección 2, paso 1).

Para qué sirve: el usuario define PRIMERO los materiales y las secciones
que va a emplear, y después dibuja asignando esas secciones. Un mismo
modelo puede tener varias secciones de distinto tipo (frame, plane,
plate, shell), igual que en un programa comercial.

Una sección agrupa: el material (E, ν, densidad) y las propiedades
geométricas propias del tipo de elemento:

    FrameSection  -> A, Iy, Iz, J (y G derivado del material)
    AreaSection   -> t (espesor), y para plane la hipótesis de trabajo

Las secciones son objetos con nombre, de modo que el modelo guarda la
REFERENCIA por nombre y no copias sueltas de E, ν, t por elemento.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .materials import Material, get_material


# Tipos de elemento que el aplicativo puede dibujar (Corrección 2, paso 1)
TipoElemento = Literal["frame", "plane", "plate", "shell"]


@dataclass
class Section:
    """Base común: nombre y material de la sección."""
    name: str
    material: Material

    @property
    def E(self) -> float:
        """Módulo de elasticidad del material asignado (Pa)."""
        return self.material.E

    @property
    def nu(self) -> float:
        """Coeficiente de Poisson del material asignado."""
        return self.material.nu

    @property
    def G(self) -> float:
        """Módulo de corte G = E / (2(1+ν)) — lo usa el elemento frame."""
        return self.E / (2.0 * (1.0 + self.nu))


@dataclass
class FrameSection(Section):
    """Sección de barra (viga/columna): propiedades de la sección transversal.

    Para qué sirve: alimenta al FrameElement con A, Iy, Iz y J. Puede
    definirse a mano o generarse a partir de una geometría rectangular
    con `rectangular()`.
    """
    A: float                        # área de la sección (m²)
    Iy: float                       # inercia respecto al eje local y (m⁴)
    Iz: float                       # inercia respecto al eje local z (m⁴)
    J: float                        # constante torsional (m⁴)
    tipo: TipoElemento = "frame"
    # Dimensiones nominales (solo informativas, para la UI y el reporte)
    b: float | None = None          # base (m), si es rectangular
    h: float | None = None          # peralte (m), si es rectangular

    @classmethod
    def rectangular(cls, name: str, material: Material,
                    b: float, h: float) -> "FrameSection":
        """Sección rectangular b×h: calcula A, Iy, Iz y J automáticamente.

        J se aproxima con la fórmula clásica de torsión de Saint-Venant
        para secciones rectangulares macizas:
            J = beta · a · c³   con a = lado mayor, c = lado menor.
        """
        A = b * h
        Iz = b * h ** 3 / 12.0          # flexión alrededor del eje local z
        Iy = h * b ** 3 / 12.0          # flexión alrededor del eje local y
        a, c = max(b, h), min(b, h)
        r = c / a
        beta = (1.0 / 3.0) * (1.0 - 0.63 * r + 0.052 * r ** 5)
        J = beta * a * c ** 3
        return cls(name=name, material=material, A=A, Iy=Iy, Iz=Iz, J=J,
                   b=b, h=h)


@dataclass
class AreaSection(Section):
    """Sección de elemento de área: plane, plate o shell.

    Para qué sirve: los tres comparten el espesor t; `tipo` decide qué
    formulación se usa al ensamblar (membrana, flexión o ambas).
    `plane_stress` solo aplica al tipo "plane".
    """
    t: float                        # espesor (m)
    tipo: TipoElemento = "shell"
    plane_stress: bool = True       # solo para tipo "plane"

    def __post_init__(self) -> None:
        if self.tipo not in ("plane", "plate", "shell"):
            raise ValueError(
                "AreaSection.tipo debe ser 'plane', 'plate' o 'shell'."
            )


@dataclass
class SectionLibrary:
    """Biblioteca de secciones del modelo (Corrección 2, paso 1).

    Para qué sirve: es el catálogo que el usuario define antes de dibujar.
    Los elementos guardan el NOMBRE de la sección; la biblioteca resuelve
    el nombre a sus propiedades al momento de ensamblar.
    """
    sections: dict[str, Section] = field(default_factory=dict)

    def add(self, section: Section) -> Section:
        """Registra una sección (sobrescribe si el nombre ya existe)."""
        self.sections[section.name] = section
        return section

    def get(self, name: str) -> Section:
        """Devuelve la sección por nombre, con error claro si no existe."""
        if name not in self.sections:
            disponibles = ", ".join(sorted(self.sections)) or "(ninguna)"
            raise KeyError(
                f"La sección {name!r} no está definida. "
                f"Secciones disponibles: {disponibles}."
            )
        return self.sections[name]

    def of_type(self, tipo: TipoElemento) -> list[Section]:
        """Secciones de un tipo dado — para poblar los combos de la UI."""
        return [s for s in self.sections.values()
                if getattr(s, "tipo", None) == tipo]

    def __contains__(self, name: object) -> bool:
        return name in self.sections

    def __len__(self) -> int:
        return len(self.sections)


def default_library() -> SectionLibrary:
    """Biblioteca inicial con secciones típicas, para que el usuario
    tenga algo con qué empezar a dibujar sin configurar nada."""
    lib = SectionLibrary()
    concreto = get_material("Hormigón armado")
    acero = get_material("Acero estructural")
    lib.add(FrameSection.rectangular("VIGA 30x50", concreto, 0.30, 0.50))
    lib.add(FrameSection.rectangular("COLUMNA 40x40", concreto, 0.40, 0.40))
    lib.add(AreaSection("LOSA e=0.20", concreto, t=0.20, tipo="shell"))
    lib.add(AreaSection("PLACA e=0.15", concreto, t=0.15, tipo="plate"))
    lib.add(AreaSection("MURO e=0.25", concreto, t=0.25, tipo="plane"))
    lib.add(FrameSection.rectangular("PERFIL ACERO 20x20", acero, 0.20, 0.20))
    return lib
