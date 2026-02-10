import sys
from dataclasses import dataclass, field
from typing import List

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QApplication, QHBoxLayout, QLabel, QMainWindow, QWidget

from src.gui.navbar.global_navbar import GlobalNavbar

# ---------------- mock NavNode ----------------


@dataclass
class NavNode:
    title: str
    widget: QWidget | None = None
    children: List["NavNode"] = field(default_factory=list)


class MockController:
    def __init__(self, root: NavNode):
        self.root = root

    def ancestors(self, node: NavNode):
        path = []

        def walk(cur, acc):
            if cur == node:
                path.extend(acc + [cur])
                return True
            for c in cur.children:
                if walk(c, acc + [cur]):
                    return True
            return False

        walk(self.root, [])
        return path


# ---------------- global navbar under test ----------------
# paste your GlobalNavbar here verbatim


# ---------------- test harness ----------------


class TestWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GlobalNavbar smoke test")

        # pages
        self.page = QLabel("Select a node")
        self.page.setStyleSheet("font-size: 18px;")

        # nav tree
        home = NavNode("Home", QLabel("Home page"))
        matches = NavNode("Matches", QLabel("Matches page"))
        manage = NavNode("Manage Match", QLabel("Manage page"))
        momentum = NavNode("Momentum", QLabel("Momentum page"))

        home.children = [matches]
        matches.children = [manage]
        manage.children = [momentum]
        root = NavNode("root", None, [home])

        self.controller = MockController(root)

        # navbar
        self.navbar = GlobalNavbar()
        self.navbar.navigate.connect(self.on_navigate)

        # layout
        central = QWidget()
        layout = QHBoxLayout(central)
        layout.addWidget(self.navbar)
        layout.addWidget(self.page, 1)
        self.setCentralWidget(central)

        # start state
        self.on_navigate(momentum)

    def on_navigate(self, node: NavNode):
        self.navbar.set_node(node, self.controller)
        if node.widget:
            self.page.setText(node.widget.text())


# ---------------- run ----------------

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = TestWindow()
    w.resize(800, 500)
    w.show()
    sys.exit(app.exec())
