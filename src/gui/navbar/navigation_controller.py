from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QWidget


class View(Enum):
    HOME = auto()
    COMPARE_MATCHES = auto()
    MANAGE_MATCH = auto()
    MOMENTUM = auto()
    HEAT_MAP = auto()
    COMPARE_MOMENTUM = auto()


@dataclass
class NavNode:
    view: View
    title: str
    widget: QWidget | None = None
    parent: "NavNode | None" = None
    children: list["NavNode"] = field(default_factory=list)


class NavigationController(QObject):
    changed = Signal(NavNode)

    def __init__(self, root: NavNode):
        super().__init__()
        self.root = root
        self.current = root.children[0]

    def navigate(self, node: NavNode):
        self.current = node
        self.changed.emit(node)

    def ancestors(self, node: NavNode):
        path = []
        while node and node.view:
            path.append(node)
            node = node.parent
        return list(reversed(path))
