"""Catálogo de materiales típicos en ingeniería civil/mecánica.

Provee una base de materiales con propiedades elásticas (E, ν) y densidad
para facilitar la selección desde la UI. Todos los valores están en
unidades del Sistema Internacional (Pa, kg/m³).

Uso:
    from src.fem.materials import MATERIALS, get_material, list_materials
    mat = get_material("Acero estructural")
    print(mat.E, mat.nu)
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class Material:
    name: str
    E: float          # Módulo de elasticidad (Pa)
    nu: float         # Coeficiente de Poisson
    density: float    # Densidad (kg/m³) — opcional, para futuras extensiones
    description: str  # Descripción y notas
    color: str        # Color hex para visualización (opcional)


# Catálogo de materiales típicos
MATERIALS: dict[str, Material] = {
    "Acero estructural": Material(
        name="Acero estructural",
        E=2.1e11, nu=0.30, density=7850,
        description="Acero al carbono típico (A36, A572). Material isótropo y "
                    "homogéneo. Usado en estructuras metálicas, perfiles, vigas.",
        color="#7F8C8D",
    ),
    "Hormigón armado": Material(
        name="Hormigón armado",
        E=2.5e10, nu=0.20, density=2400,
        description="Hormigón f'c=21 MPa con armadura típica. ν=0.20 es valor "
                    "promedio (varía 0.15-0.25). Anisotropía local ignorada.",
        color="#95A5A6",
    ),
    "Hormigón simple": Material(
        name="Hormigón simple",
        E=2.0e10, nu=0.20, density=2300,
        description="Hormigón sin refuerzo, f'c≈20 MPa. Comportamiento frágil "
                    "en tracción; el modelo lineal elástico aplica solo para "
                    "estados de servicio en compresión predominante.",
        color="#A6ACAF",
    ),
    "Aluminio 6061-T6": Material(
        name="Aluminio 6061-T6",
        E=6.9e10, nu=0.33, density=2700,
        description="Aleación de aluminio tratada térmicamente. Buena "
                    "resistencia/peso. Usada en perfilería ligera, aeronáutica "
                    "y estructuras secundarias.",
        color="#BDC3C7",
    ),
    "Cobre": Material(
        name="Cobre",
        E=1.1e11, nu=0.34, density=8960,
        description="Cobre puro recocido. Alta conductividad eléctrica y "
                    "térmica. Uso estructural limitado, común en instalaciones "
                    "y conexiones eléctricas.",
        color="#B87333",
    ),
    "Titanio (Ti-6Al-4V)": Material(
        name="Titanio (Ti-6Al-4V)",
        E=1.14e11, nu=0.34, density=4430,
        description="Aleación de titanio grado 5. Alta resistencia específica "
                    "y excelente comportamiento a fatiga y corrosión. Usada "
                    "en aeroespacial y prótesis biomédicas.",
        color="#85929E",
    ),
    "Madera (pino)": Material(
        name="Madera (pino)",
        E=1.0e10, nu=0.20, density=500,
        description="Pino estructural en dirección paralela a la fibra. "
                    "La anisotropía real (E_perp << E_par) se ignora; modelo "
                    "isótropo válido solo para análisis preliminares.",
        color="#D4A574",
    ),
    "Vidrio": Material(
        name="Vidrio",
        E=7.0e10, nu=0.22, density=2500,
        description="Vidrio sodocálcico común. Comportamiento frágil con "
                    "alta resistencia a compresión. Usado en cerramientos, "
                    "fachadas y elementos arquitectónicos no estructurales.",
        color="#AED6F1",
    ),
    "Fibra de carbono (UD)": Material(
        name="Fibra de carbono (UD)",
        E=1.5e11, nu=0.30, density=1600,
        description="Compuesto unidireccional de fibra de carbono/epoxi, "
                    "propiedades en dirección de las fibras. Modelo isótropo "
                    "simplificado; el material real es fuertemente "
                    "anisótropo (E_transversal ~ 1/10 del longitudinal).",
        color="#2C3E50",
    ),
    "Caucho (NBR)": Material(
        name="Caucho (NBR)",
        E=5.0e6, nu=0.49, density=1200,
        description="Caucho nitrílico (NBR), elastómero típico. ν≈0.49 indica "
                    "casi incompresibilidad: los elementos Q4 estándar pueden "
                    "sufrir locking volumétrico — usar con precaución.",
        color="#34495E",
    ),
}


def get_material(name: str) -> Material:
    """Devuelve un material del catálogo por nombre.

    Lanza KeyError si el nombre no existe.
    """
    if name not in MATERIALS:
        raise KeyError(
            f"Material '{name}' no encontrado en el catálogo. "
            f"Disponibles: {list(MATERIALS.keys())}"
        )
    return MATERIALS[name]


def list_materials() -> list[str]:
    """Lista los nombres de todos los materiales disponibles."""
    return list(MATERIALS.keys())
