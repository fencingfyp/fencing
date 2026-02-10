import sys
from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)


def is_video_file(file_path: str) -> bool:
    return file_path.lower().endswith(".mp4")


class SelectMatchDialog(QDialog):
    submitted = Signal(str)  # emits the selected video file path

    SIDECAR_SUFFIX = ".data"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Match")

        # --- widgets ---
        self.file_path_label = QLabel("File Path")
        self.file_path_box = QLineEdit()
        self.file_path_box.setReadOnly(True)
        self.file_path_box.setPlaceholderText("No file selected")

        self.browse_file_button = QPushButton("Browse")
        self.error_label = QLabel()
        self.error_label.setWordWrap(True)
        self.error_label.setStyleSheet("color: red;")

        self.control_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )

        # --- layout ---
        file_row = QHBoxLayout()
        file_row.addWidget(self.file_path_box)
        file_row.addWidget(self.browse_file_button)

        layout = QVBoxLayout(self)
        layout.addWidget(self.file_path_label)
        layout.addLayout(file_row)
        layout.addWidget(self.error_label)
        layout.addStretch()
        layout.addWidget(self.control_box)

        # --- signals ---
        self.browse_file_button.clicked.connect(self.pick_file)
        self.control_box.accepted.connect(self.on_ok)
        self.control_box.rejected.connect(self.reject)

    def pick_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select video file",
            "",
            "Video Files (*.mp4)",
        )
        if file_path:
            self.file_path_box.setText(file_path)

    def on_ok(self):
        file_path = self.file_path_box.text().strip()

        # --- basic validation ---
        if not file_path:
            self.error_label.setText("Please select a file.")
            return

        if not is_video_file(file_path):
            self.error_label.setText("Please select a valid video file (.mp4).")
            return

        # --- check for sidecar folder ---
        video_file = Path(file_path)
        sidecar_folder = video_file.parent / f"{video_file.stem}{self.SIDECAR_SUFFIX}"

        if not sidecar_folder.exists() or not sidecar_folder.is_dir():
            result = QMessageBox.question(
                self,
                "Missing Sidecar Data",
                f"The required sidecar folder '{sidecar_folder.name}' does not exist.\n\n"
                "This may be a new project or the data has been lost/renamed.\n\n"
                "Do you want to proceed anyway?",
                QMessageBox.Yes | QMessageBox.Cancel,
                QMessageBox.Cancel,
            )
            if result != QMessageBox.Yes:
                return  # cancel submission

        # --- emit the selected video path ---
        self.submitted.emit(file_path)
        self.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    dialog = SelectMatchDialog()
    dialog.open()
    sys.exit(app.exec())
