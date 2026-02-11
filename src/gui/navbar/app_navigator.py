from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QStackedWidget, QWidget

from .navigation_controller import NavNode, View


class AppNavigator(QObject):
    changed = Signal(NavNode)

    def __init__(self, stack: QStackedWidget):
        super().__init__()
        self.stack = stack
        self.root = NavNode(None, "root", None)
        self.nodes: dict[View, NavNode] = {}
        self.current: NavNode | None = None

    def register(
        self,
        *,
        view: View,
        title: str,
        widget: QWidget,
        parent: View | None = None,
    ):
        parent_node = self.nodes.get(parent, self.root)

        node = NavNode(view, title, widget, parent_node)
        parent_node.children.append(node)
        self.nodes[view] = node

        if widget not in (None, self.stack):
            self.stack.addWidget(widget)
            if hasattr(widget, "navigate"):
                widget.navigate.connect(lambda v=view: self.navigate(v))

        return node

    def navigate(self, view: View):
        node = self.nodes[view]
        self.current = node

        if node.widget:
            self.stack.setCurrentWidget(node.widget)

        self.changed.emit(node)

    def ancestors(self, node: NavNode):
        path = []
        while node and node.view:
            path.append(node)
            node = node.parent
        return list(reversed(path))

    def siblings(self, node: NavNode):
        return [c for c in node.parent.children if c != node]

    def get_widget(self, view: View) -> QWidget:
        node = self.nodes[view]
        return node.widget
