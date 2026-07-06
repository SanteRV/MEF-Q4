"""Test rápido del núcleo MEF (Q4) sin abrir la UI.

Ejecutar:
    .\venv\Scripts\python.exe test_core.py
"""
from pathlib import Path
import json
from src.fem.node import Node
from src.fem.q4_element import Q4Element
from src.fem.structure import Structure
from src.fem.steps import build_procedure
from src.export.excel_export import export_to_excel
from src.export.pdf_export import export_to_pdf


def build_from_example(path: Path) -> Structure:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    s = Structure()
    nodes = {}
    for nd in data["nodes"]:
        n = Node(
            id=nd["id"], x=nd["x"], y=nd["y"],
            restraint_x=nd.get("restraint_x", False),
            restraint_y=nd.get("restraint_y", False),
            load_x=nd.get("load_x", 0.0),
            load_y=nd.get("load_y", 0.0),
        )
        s.add_node(n)
        nodes[n.id] = n
    for el in data["elements"]:
        s.add_element(Q4Element(
            id=el["id"],
            nodes=[nodes[i] for i in el["nodes"]],
            E=el["E"], nu=el["nu"], t=el["t"],
            plane_stress=el.get("plane_stress", True),
        ))
    return s


if __name__ == "__main__":
    example = Path("examples/q4_placa.json")
    structure = build_from_example(example)
    proc = build_procedure(structure)

    print(f"Pasos generados: {len(proc.steps)}")
    for i, st in enumerate(proc.steps, 1):
        print(f"  {i}. {st.title}")

    print("\nDesplazamientos nodales:")
    for n in structure.nodes:
        ux = proc.result.displacements[n.dofs[0]]
        uy = proc.result.displacements[n.dofs[1]]
        print(f"  Nodo {n.id + 1}: ux = {ux: .6e} m, uy = {uy: .6e} m")

    print("\nEsfuerzos en puntos de Gauss (elemento 1):")
    for i, sig in enumerate(proc.result.elements[0].stresses_at_gauss, 1):
        print(f"  GP{i}: sx={sig[0]: .3e}, sy={sig[1]: .3e}, txy={sig[2]: .3e}")

    Path("out").mkdir(exist_ok=True)
    export_to_excel(proc, Path("out/test_q4.xlsx"))
    export_to_pdf(proc, structure, Path("out/test_q4.pdf"))
    print("\nArchivos exportados a out/test_q4.xlsx y out/test_q4.pdf")
