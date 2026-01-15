import re
import sys

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QApplication,
    QDialogButtonBox,
    QFileDialog,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QWidget,
)

from src.gui.util.io import load_ui_dynamic


def is_video_file(file_path: str) -> bool:
    video_extensions = {".mp4"}
    return any(file_path.lower().endswith(ext) for ext in video_extensions)


class AddMatchDialog(QWidget):  # This is a wrapper for the actual dialog UI
    submitted = Signal(str, str)  # name, file_path

    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = load_ui_dynamic("src/gui/add_match_dialog.ui", self)

        self.name_label = self.ui.findChild(QLabel, "nameLabel")
        self.name_box = self.ui.findChild(QPlainTextEdit, "nameBox")
        self.file_path_label = self.ui.findChild(QLabel, "filePathLabel")
        self.file_path_box = self.ui.findChild(QLineEdit, "filePathBox")
        self.browse_file_button = self.ui.findChild(QPushButton, "browseFileButton")
        self.control_box = self.ui.findChild(QDialogButtonBox, "controlBox")
        self.error_label = self.ui.findChild(QLabel, "errorLabel")

        self.browse_file_button.clicked.connect(self.pick_file)
        self.file_path_box.setReadOnly(True)
        self.file_path_box.setPlaceholderText("No file selected")
        self.control_box.accepted.connect(self.on_ok)
        self.error_label.setStyleSheet("color: red;")

    def pick_file(self):
        file_dialog = QFileDialog(self)
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        if file_dialog.exec():
            selected_files = file_dialog.selectedFiles()
            if selected_files:
                self.file_path_box.setText(selected_files[0])

    def on_ok(self):
        # print("Submitted:", self.name_box.toPlainText(), self.file_path_box.text())
        # print(self.name_box.toPlainText().strip())
        name = self.name_box.toPlainText().strip()
        file_path = self.file_path_box.text().strip()
        if not len(name):
            self.error_label.setText("Please enter a match name.")
            return

        regex = "^[a-zA-Z0-9._-]+$"
        if not re.match(regex, name):
            self.error_label.setText(
                "Match name can only contain letters, numbers, dots, underscores, and hyphens."
            )
            return

        if not len(file_path):
            self.error_label.setText("Please select a valid file path.")
            return
        if not is_video_file(file_path):
            self.error_label.setText("Please select a valid video file (.mp4).")
            return
        self.submitted.emit(name, file_path)
        self.ui.accept()

    def open(self):
        self.ui.open()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    dialog = AddMatchDialog()
    dialog.open()
    sys.exit(app.exec())
