import sys
from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QApplication,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.model.FileManager import FileManager

from .select_match_dialog import SelectMatchDialog


class SelectMatchWidget(QWidget):
    selected = Signal(str)  # emits full video file path

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Select Match")

        # --- widgets ---
        self.label = QLabel("Select a video file to open:")
        # self.input_line = QLineEdit()
        # self.input_line.setReadOnly(True)
        # self.input_line.setPlaceholderText("No file selected")

        self.select_button = QPushButton("Select Match")

        # --- layout ---
        layout = QVBoxLayout(self)
        layout.addWidget(self.select_button)
        # layout.addStretch()

        # --- signals ---
        self.select_button.clicked.connect(self._select_file)

    def _select_file(self):
        dialog = SelectMatchDialog(self)
        dialog.submitted.connect(self._on_file_selected)
        dialog.open()

    def _on_file_selected(self, file_path: str):
        """
        Emits the selected video file path.
        The main app / context can handle sidecar folder logic.
        """

        # Example integration scaffolding with FileManager:
        # 1. Ensure sidecar exists
        FileManager.create_sidecar(file_path)

        # 2. Construct a FileManager instance (raises if metadata missing)
        # fm = FileManager(file_path)

        # 3. Emit the video path for the app / match context
        self.selected.emit(file_path)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SelectMatchWidget()
    window.show()
    sys.exit(app.exec())
