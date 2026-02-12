from dataclasses import dataclass
from typing import Callable, Iterable

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import QHBoxLayout, QPushButton, QWidget


@dataclass
class TaskAction:
    id: str
    label: str
    shortcut: Qt.Key | None
    callback: Callable
    enabled: bool = True


class ActionsPanelWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._shortcuts: list[QShortcut] = []

    def set_actions(self, actions: Iterable[TaskAction]):
        self.clear()

        actions = list(actions)

        for action in actions:
            btn = QPushButton(action.label, self)
            btn.setEnabled(action.enabled)
            btn.clicked.connect(action.callback)
            self._layout.addWidget(btn)

            if action.shortcut:
                sc = QShortcut(QKeySequence(action.shortcut), self.window())
                sc.setContext(Qt.WindowShortcut)
                sc.activated.connect(action.callback)
                sc.setEnabled(action.enabled)
                self._shortcuts.append(sc)

        self.setVisible(bool(actions))

    def clear(self):
        # Clear buttons
        while self._layout.count():
            item = self._layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

        # Clear shortcuts
        for sc in self._shortcuts:
            sc.setParent(None)
        self._shortcuts.clear()
