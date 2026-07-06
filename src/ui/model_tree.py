"""Árbol jerárquico del modelo MEF (panel lateral estilo SAP / Abaqus).

Muestra nodos, elementos, apoyos y cargas agrupados por categoría.
Click selecciona, doble-click edita, Delete pide eliminar.

Uso:
    tree = ModelTreeWidget()
    tree.set_structure(structure)
    tree.item_selected.connect(handler)        # (category: str, obj_id: int)
    tree.item_double_clicked.connect(editor)
    tree.item_delete_requested.connect(deleter)
"""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QKeyEvent
from PySide6.QtWidgets import QTreeWidget, QTreeWidgetItem, QSizePolicy

from .icons import icon
from .theme import Colors


# Claves identificadoras de cada categoría raíz (estables entre refresh()).
# Se guardan en UserRole+1 del item raíz para preservar el estado expand/collapse.
_CAT_NODES     = "cat_nodes"
_CAT_ELEMENTS  = "cat_elements"
_CAT_SUPPORTS  = "cat_supports"
_CAT_LOADS     = "cat_loads"


# QSS local al widget: navy seleccionado, hover gris claro, padding generoso.
_TREE_QSS = f"""
QTreeWidget {{
    background-color: {Colors.BG};
    border: 1px solid {Colors.BORDER};
    border-radius: 4px;
    outline: none;
    font-size: 9.5pt;
    show-decoration-selected: 1;
}}

QTreeWidget::item {{
    padding: 6px 4px;
    color: {Colors.TEXT};
    border: none;
}}

QTreeWidget::item:hover {{
    background-color: {Colors.BG_SUBTLE};
}}

QTreeWidget::item:selected {{
    background-color: {Colors.PRIMARY};
    color: white;
}}

QTreeWidget::item:selected:!active {{
    background-color: {Colors.PRIMARY};
    color: white;
}}

QTreeWidget::branch {{
    background-color: {Colors.BG};
}}

QTreeWidget::branch:selected {{
    background-color: {Colors.PRIMARY};
}}
"""


def _fmt_coord(v: float) -> str:
    """Formato compacto para coordenadas en el árbol (no para round-trip)."""
    return f"{v:.4g}"


class ModelTreeWidget(QTreeWidget):
    """Árbol jerárquico del modelo MEF.

    Estructura:
        Estructura (root)
          Nodos (N)
            N1 (0.5, 0.5)
            ...
          Elementos (M)
            E1 [Q4: N1, N2, N3, N4]
            ...
          Apoyos (K)
            N2  Fix X, Fix Y
            ...
          Cargas (L)
            N1  F=(100, 100) N
            ...

    Señales:
        item_selected(category, obj_id):
            category in {"node", "element", "support", "load", "category"}
            obj_id es el id del Node / Q4Element (o -1 para "category").
        item_double_clicked(category, obj_id): mismo formato.
        item_delete_requested(category, obj_id): tecla Delete sobre un nodo
            o elemento (no se emite para categorías).
    """

    item_selected = Signal(str, int)
    item_double_clicked = Signal(str, int)
    item_delete_requested = Signal(str, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._structure = None

        # Configuración básica: una columna, header oculto.
        self.setColumnCount(1)
        self.setHeaderHidden(True)
        self.setRootIsDecorated(True)
        self.setIndentation(16)
        self.setUniformRowHeights(True)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self.setStyleSheet(_TREE_QSS)

        # Señales internas a las externas tipadas.
        self.itemClicked.connect(self._on_item_clicked)
        self.itemDoubleClicked.connect(self._on_item_double_clicked)

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def set_structure(self, structure) -> None:
        """Reemplaza el contenido. Llamar tras cualquier cambio en la Structure."""
        self._structure = structure
        self.refresh()

    def refresh(self) -> None:
        """Re-puebla el árbol con la Structure actual.

        Preserva el estado expand/collapse de las categorías (todas expandidas
        por defecto la primera vez).
        """
        # Capturar estado de expansión por clave estable de categoría.
        expanded = self._capture_expanded_state()

        self.clear()
        if self._structure is None:
            return

        nodes = list(self._structure.nodes)
        elements = list(self._structure.elements)
        supports = [n for n in nodes if n.restraint_x or n.restraint_y]
        loads = [n for n in nodes if (n.load_x != 0.0) or (n.load_y != 0.0)]

        # Categorías raíz.
        nodes_root = self._make_category_item(
            _CAT_NODES, f"Nodos ({len(nodes)})", "fa5s.circle",
        )
        elements_root = self._make_category_item(
            _CAT_ELEMENTS, f"Elementos ({len(elements)})", "fa5s.vector-square",
        )
        supports_root = self._make_category_item(
            _CAT_SUPPORTS, f"Apoyos ({len(supports)})", "fa5s.anchor",
        )
        loads_root = self._make_category_item(
            _CAT_LOADS, f"Cargas ({len(loads)})", "fa5s.long-arrow-alt-down",
        )

        for root in (nodes_root, elements_root, supports_root, loads_root):
            self.addTopLevelItem(root)

        # Hijos: nodos.
        for n in nodes:
            item = QTreeWidgetItem(nodes_root)
            item.setText(
                0, f"N{n.id + 1}   ({_fmt_coord(n.x)}, {_fmt_coord(n.y)})"
            )
            item.setIcon(0, icon("fa5s.circle"))
            item.setData(0, Qt.UserRole, ("node", n.id))

        # Hijos: elementos.
        for el in elements:
            node_ids = ", ".join(f"N{nd.id + 1}" for nd in el.nodes)
            item = QTreeWidgetItem(elements_root)
            item.setText(0, f"E{el.id + 1}   [Q4: {node_ids}]")
            item.setIcon(0, icon("fa5s.vector-square"))
            item.setData(0, Qt.UserRole, ("element", el.id))

        # Hijos: apoyos.
        for n in supports:
            parts = []
            if n.restraint_x:
                parts.append("Restr. X")
            if n.restraint_y:
                parts.append("Restr. Y")
            item = QTreeWidgetItem(supports_root)
            item.setText(0, f"N{n.id + 1}   {', '.join(parts)}")
            item.setIcon(0, icon("fa5s.anchor", color=Colors.SUCCESS))
            # Categoría "support" mapea al id del nodo asociado.
            item.setData(0, Qt.UserRole, ("support", n.id))

        # Hijos: cargas.
        for n in loads:
            item = QTreeWidgetItem(loads_root)
            item.setText(
                0,
                f"N{n.id + 1}   F=({_fmt_coord(n.load_x)}, "
                f"{_fmt_coord(n.load_y)}) N",
            )
            item.setIcon(0, icon("fa5s.long-arrow-alt-down", color=Colors.DANGER))
            item.setData(0, Qt.UserRole, ("load", n.id))

        # Restaurar expansión: si no hay estado previo (primer poblado),
        # expandir todas las categorías por defecto.
        if expanded is None:
            self.expandAll()
        else:
            for i in range(self.topLevelItemCount()):
                root = self.topLevelItem(i)
                key = root.data(0, Qt.UserRole + 1)
                root.setExpanded(expanded.get(key, True))

    def select_node(self, node_id: int) -> None:
        """Selecciona programáticamente el item del nodo con ese id."""
        self._select_by_payload("node", node_id)

    def select_element(self, el_id: int) -> None:
        """Selecciona programáticamente el item del elemento con ese id."""
        self._select_by_payload("element", el_id)

    # ------------------------------------------------------------------
    # Internos
    # ------------------------------------------------------------------

    def _make_category_item(
        self, key: str, label: str, icon_name: str
    ) -> QTreeWidgetItem:
        """Construye un item raíz de categoría (negrita, icono, clave estable)."""
        item = QTreeWidgetItem([label])
        item.setIcon(0, icon(icon_name))
        # Negrita en el label de la categoría.
        font = item.font(0)
        font.setBold(True)
        item.setFont(0, font)
        # Payload para itemClicked / itemDoubleClicked.
        item.setData(0, Qt.UserRole, ("category", -1))
        # Clave estable para preservar expand/collapse entre refresh().
        item.setData(0, Qt.UserRole + 1, key)
        return item

    def _capture_expanded_state(self) -> dict[str, bool] | None:
        """Lee el estado expand/collapse actual por clave estable.

        Devuelve None si el árbol está vacío (primer poblado).
        """
        if self.topLevelItemCount() == 0:
            return None
        state: dict[str, bool] = {}
        for i in range(self.topLevelItemCount()):
            root = self.topLevelItem(i)
            key = root.data(0, Qt.UserRole + 1)
            if key is not None:
                state[key] = root.isExpanded()
        return state

    def _select_by_payload(self, category: str, obj_id: int) -> None:
        """Busca el item con (category, obj_id) en UserRole y lo selecciona."""
        for i in range(self.topLevelItemCount()):
            root = self.topLevelItem(i)
            for j in range(root.childCount()):
                child = root.child(j)
                payload = child.data(0, Qt.UserRole)
                if payload == (category, obj_id):
                    self.setCurrentItem(child)
                    self.scrollToItem(child)
                    return

    def _payload_of(self, item: QTreeWidgetItem) -> tuple[str, int] | None:
        if item is None:
            return None
        payload = item.data(0, Qt.UserRole)
        if isinstance(payload, tuple) and len(payload) == 2:
            return payload
        return None

    def _on_item_clicked(self, item: QTreeWidgetItem, _column: int) -> None:
        payload = self._payload_of(item)
        if payload is None:
            return
        category, obj_id = payload
        self.item_selected.emit(category, int(obj_id))

    def _on_item_double_clicked(self, item: QTreeWidgetItem, _column: int) -> None:
        payload = self._payload_of(item)
        if payload is None:
            return
        category, obj_id = payload
        self.item_double_clicked.emit(category, int(obj_id))

    def keyPressEvent(self, event: QKeyEvent) -> None:
        # Delete sobre un nodo o elemento solicita borrado; categorías se ignoran.
        if event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            payload = self._payload_of(self.currentItem())
            if payload is not None:
                category, obj_id = payload
                if category in ("node", "element", "support", "load"):
                    self.item_delete_requested.emit(category, int(obj_id))
                    event.accept()
                    return
        super().keyPressEvent(event)
