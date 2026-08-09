"""Persistencia de proyectos: serialización Structure ⇄ JSON."""
from __future__ import annotations
import json
from pathlib import Path

from .node import Node
from .q4_element import Q4Element
from .structure import Structure


PROJECT_VERSION = "1.0"


def structure_to_dict(structure: Structure, description: str = "") -> dict:
    """Convierte una Structure a dict JSON-serializable."""
    return {
        "version": PROJECT_VERSION,
        "description": description,
        "nodes": [
            {
                "id": n.id,
                "x": float(n.x),
                "y": float(n.y),
                "restraint_x": bool(n.restraint_x),
                "restraint_y": bool(n.restraint_y),
                "prescribed_x": float(n.prescribed_x),
                "prescribed_y": float(n.prescribed_y),
                "load_x": float(n.load_x),
                "load_y": float(n.load_y),
            }
            for n in structure.nodes
        ],
        "elements": [
            {
                "id": el.id,
                "nodes": [n.id for n in el.nodes],
                "E": float(el.E),
                "nu": float(el.nu),
                "t": float(el.t),
                "plane_stress": bool(el.plane_stress),
            }
            for el in structure.elements
        ],
    }


def dict_to_structure(data: dict) -> Structure:
    """Reconstruye una Structure desde un dict."""
    s = Structure()
    nodes_by_id: dict[int, Node] = {}
    for nd in data["nodes"]:
        n = Node(
            id=int(nd["id"]),
            x=float(nd["x"]),
            y=float(nd["y"]),
            restraint_x=bool(nd.get("restraint_x", False)),
            restraint_y=bool(nd.get("restraint_y", False)),
            prescribed_x=float(nd.get("prescribed_x", 0.0)),
            prescribed_y=float(nd.get("prescribed_y", 0.0)),
            load_x=float(nd.get("load_x", 0.0)),
            load_y=float(nd.get("load_y", 0.0)),
        )
        s.add_node(n)
        nodes_by_id[n.id] = n
    for el in data["elements"]:
        s.add_element(Q4Element(
            id=int(el["id"]),
            nodes=[nodes_by_id[i] for i in el["nodes"]],
            E=float(el["E"]),
            nu=float(el["nu"]),
            t=float(el["t"]),
            plane_stress=bool(el.get("plane_stress", True)),
        ))
    return s


def save_project(structure: Structure, path: Path, description: str = "") -> None:
    """Guarda la estructura como JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = structure_to_dict(structure, description)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_project(path: Path) -> tuple[Structure, str]:
    """Carga estructura desde JSON. Devuelve (structure, description)."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return dict_to_structure(data), data.get("description", "")
