from PySide6.QtCore import Signal
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QFrame, QPushButton, QSizePolicy, QVBoxLayout, QWidget

from src.gui.navbar.navigation_controller import NavNode


class GlobalNavbar(QWidget):
    navigate = Signal(NavNode)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setFixedWidth(240)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(8, 8, 8, 8)
        self.main_layout.setSpacing(6)
        # self.setStyleSheet("border: 2px solid blue;")

        self.global_container = QVBoxLayout()
        self.local_container = QVBoxLayout()
        self.local_container.setContentsMargins(0, 12, 0, 0)

        self.main_layout.addLayout(self.global_container)
        self._local_wrapper = QWidget(self)
        self._local_wrapper.setLayout(self.local_container)
        # self._local_wrapper.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._local_wrapper.setMinimumHeight(1)
        # self._local_wrapper.setStyleSheet("border: 2px solid red;")

        self.main_layout.addWidget(self._local_wrapper)
        self.main_layout.addStretch(1)

        self._current_local_navbar = None

    # ---------------- navigation ----------------

    def set_node(self, node: NavNode, controller):
        self._clear_global()
        self._detach_local_navbar()

        ancestors = controller.ancestors(node)
        for n in ancestors:
            is_active = n == node
            self._add_global_button(n, active=is_active)

        # ---- local navbar (page-owned) ----
        local = getattr(node.widget, "local_navbar", None)
        if local:
            local.setParent(self._local_wrapper)
            local.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
            self.local_container.addWidget(local)
            self._current_local_navbar = local

    # ---------------- helpers ----------------

    def _add_global_button(self, node: NavNode, active=False):
        btn = QPushButton(node.title, self)
        btn.setEnabled(not active)

        if active:
            pal = btn.palette()
            pal.setColor(QPalette.Button, QColor("#2a2a2a"))
            pal.setColor(QPalette.ButtonText, QColor("white"))
            btn.setPalette(pal)
            btn.setStyleSheet("font-weight: bold;")

        btn.clicked.connect(lambda _, n=node: self.navigate.emit(n))
        self.global_container.addWidget(btn)

    def _separator(self):
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        line.setContentsMargins(0, 6, 0, 6)
        return line

    # ---------------- lifecycle safety ----------------

    def _clear_global(self):
        """Destroy only global navigation widgets."""
        while self.global_container.count():
            item = self.global_container.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

    def _detach_local_navbar(self):
        """Detach local navbar without destroying it."""
        if self._current_local_navbar:
            self.local_container.removeWidget(self._current_local_navbar)
            self._current_local_navbar.setParent(None)
            self._current_local_navbar = None
