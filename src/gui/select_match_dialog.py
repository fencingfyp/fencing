import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, List, Optional

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

from src.gui.util.file_row import FileRow
from src.model.FileManager import FileManager


class ValidationState(Enum):
    ACCEPT = 1
    WARNING = 2
    REJECT = 3


@dataclass
class ValidationResult:
    state: ValidationState
    message: str | None = None


def is_video_file(file_path: str) -> bool:
    return file_path.lower().endswith(".mp4") or file_path.lower().endswith(".mts")


class SelectMatchDialog(QDialog):
    submitted = Signal(list)  # emits List[str]

    SIDECAR_SUFFIX = ".data"

    def __init__(
        self,
        parent=None,
        multi_select: bool = False,
        validator: Optional[Callable] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Select Match")

        self.multi_select = multi_select
        self.validator = validator or self._default_validator
        self.file_rows: List[FileRow] = []

        # --- widgets ---
        self.info_label = QLabel(
            "Select video file(s):" if multi_select else "Select a video file:"
        )

        self.rows_layout = QVBoxLayout()
        self.add_file_button = QPushButton("+ Add File")
        if not multi_select:
            self.add_file_button.hide()

        self.error_label = QLabel()
        self.error_label.setStyleSheet("color: red;")
        self.error_label.setWordWrap(True)

        self.control_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )

        # --- layout ---
        layout = QVBoxLayout(self)
        layout.addWidget(self.info_label)
        layout.addLayout(self.rows_layout)
        layout.addWidget(self.add_file_button)
        layout.addWidget(self.error_label)
        layout.addStretch()
        layout.addWidget(self.control_box)

        # --- signals ---
        self.add_file_button.clicked.connect(self.add_row)
        self.control_box.accepted.connect(self.on_ok)
        self.control_box.rejected.connect(self.reject)

        # --- initial state ---
        self.add_row()

        if not self.multi_select:
            self.add_file_button.setDisabled(True)

    # ============================================================
    # Row management
    # ============================================================

    def add_row(self):
        row = FileRow(self)
        row.removed.connect(self.remove_row)

        self.file_rows.append(row)
        self.rows_layout.addWidget(row)

    def remove_row(self, row: FileRow):
        if not self.multi_select:
            return  # prevent removing the only row in single mode

        self.file_rows.remove(row)
        row.setParent(None)
        row.deleteLater()

    def get_selected_paths(self) -> List[str]:
        return [r.get_path() for r in self.file_rows if r.get_path()]

    # ============================================================
    # Validation + submission
    # ============================================================

    def on_ok(self):
        file_paths = self.get_selected_paths()

        if not file_paths:
            self.error_label.setText("Please select at least one file.")
            return

        result = self.validator(file_paths)

        if result.state == ValidationState.REJECT:
            self.error_label.setText(result.message or "Invalid selection.")
            return

        if result.state == ValidationState.WARNING:
            confirm = QMessageBox.question(
                self,
                "Warning",
                result.message or "Proceed?",
                QMessageBox.Yes | QMessageBox.Cancel,
                QMessageBox.Cancel,
            )
            if confirm != QMessageBox.Yes:
                return

        # ACCEPT or confirmed WARNING
        self.submitted.emit(file_paths)
        self.accept()

    # ------------------------------------------------------------------
    # Default validator
    # ------------------------------------------------------------------

    def _default_validator(self, file_paths: list[str]) -> ValidationResult:
        missing_folders = []
        for file_path in file_paths:
            if not is_video_file(file_path):
                return ValidationResult(
                    ValidationState.REJECT,
                    f"Invalid video file: {file_path}",
                )

            valid_folder_exists = FileManager.has_valid_sidecar(file_path)
            sidecar_folder_name = file_path[:-4] + self.SIDECAR_SUFFIX

            if not valid_folder_exists:
                missing_folders.append(sidecar_folder_name)

        if missing_folders:
            message = (
                "The following sidecar folders are missing:\n"
                + "\n".join(missing_folders)
                + "\n\nProceed anyway?"
            )
            return ValidationResult(ValidationState.WARNING, message)
        return ValidationResult(ValidationState.ACCEPT)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    dialog = SelectMatchDialog(multi_select=True)
    dialog.open()
    sys.exit(app.exec())
