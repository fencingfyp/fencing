import sys
from typing import List

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QWidget

from src.gui.navbar.app_navigator import AppNavigator, View
from src.model.FileManager import FileManager

from .select_match_dialog import SelectMatchDialog


def navigation(nav: AppNavigator, match_ctx: FileManager):
    nav.register(
        view=View.HOME,
        title="Home",
        widget=HomeWidget(),
    )


class HomeWidget(QWidget):
    """
    Home page widget.

    Provides:
    - Open single match
    - Compare multiple matches
    - Extension point for future actions
    """

    # Signals for the application layer
    single_selected = Signal(str)  # emits one video path
    multiple_selected = Signal(list)  # emits list[str]

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Home")

        # --- widgets ---
        self.title_label = QLabel("Select an action:")

        self.open_single_button = QPushButton("Open Match")
        self.compare_button = QPushButton("Compare Matches")

        # --- layout ---
        layout = QVBoxLayout(self)
        layout.addWidget(self.title_label)
        layout.addWidget(self.open_single_button)
        layout.addWidget(self.compare_button)
        layout.addStretch()

        # --- signals ---
        self.open_single_button.clicked.connect(self._open_single_match)
        self.compare_button.clicked.connect(self._compare_matches)

    # ------------------------------------------------------------------
    # Single match flow
    # ------------------------------------------------------------------

    def _open_single_match(self):
        dialog = SelectMatchDialog(self, multi_select=False)
        dialog.submitted.connect(self._on_single_selected)
        dialog.open()

    def _on_single_selected(self, file_paths: list[str]):
        FileManager.create_sidecar(file_paths[0])
        self.single_selected.emit(file_paths[0])

    # ------------------------------------------------------------------
    # Multi match comparison flow
    # ------------------------------------------------------------------

    def _compare_matches(self):
        dialog = SelectMatchDialog(self, multi_select=True)
        dialog.submitted.connect(self._on_multiple_selected)
        dialog.open()

    def _on_multiple_selected(self, file_paths: List[str]):
        for path in file_paths:
            FileManager.create_sidecar(path)

        self.multiple_selected.emit(file_paths)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = HomeWidget()
    window.show()
    sys.exit(app.exec())
