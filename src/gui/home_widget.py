import sys
from typing import List

from PySide6.QtCore import QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QWidget

from src.gui.navbar.app_navigator import AppNavigator, View
from src.model.FileManager import FileManager, FileRole

from .select_match_dialog import SelectMatchDialog, ValidationResult, ValidationState

FEEDBACK_FORM_URL = "https://forms.gle/HUzQw4kAScirmgUf8"
BUG_REPORT_FORM_URL = "https://forms.gle/QEiH8tLXRMjn2Npz5"


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
    - Submit feedback
    - Report a bug
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
        self.feedback_button = QPushButton("Submit Feedback")
        self.bug_report_button = QPushButton("Report a Bug")

        # --- layout ---
        layout = QVBoxLayout(self)
        layout.addWidget(self.title_label)
        layout.addWidget(self.open_single_button)
        layout.addWidget(self.compare_button)
        layout.addStretch()
        layout.addWidget(self.feedback_button)
        layout.addWidget(self.bug_report_button)

        # --- signals ---
        self.open_single_button.clicked.connect(self._open_single_match)
        self.compare_button.clicked.connect(self._compare_matches)
        self.feedback_button.clicked.connect(self._open_feedback_form)
        self.bug_report_button.clicked.connect(self._open_bug_report_form)

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
        dialog = SelectMatchDialog(
            self, multi_select=True, validator=self._validate_multiple_selection
        )
        dialog.submitted.connect(self._on_multiple_selected)
        dialog.open()

    def _on_multiple_selected(self, file_paths: List[str]):
        for path in file_paths:
            FileManager.create_sidecar(path)
        self.multiple_selected.emit(file_paths)

    def _validate_multiple_selection(self, file_paths: List[str]) -> ValidationResult:
        if len(file_paths) < 2:
            return ValidationResult(
                ValidationState.REJECT,
                "Please select at least 2 video files for comparison.",
            )
        for file in file_paths:
            file_manager = FileManager(file)
            has_required_files = file_manager.file_exists(
                FileRole.RAW_MOMENTUM_DATA
            ) and file_manager.file_exists(FileRole.PERIODS)
            if not has_required_files:
                return ValidationResult(
                    ValidationState.REJECT,
                    f"File '{file}' is missing required data. Please ensure momentum data and periods have been generated for all selected matches.",
                )
        return ValidationResult(ValidationState.ACCEPT)

    # ------------------------------------------------------------------
    # Feedback flows
    # ------------------------------------------------------------------
    def _open_feedback_form(self):
        QDesktopServices.openUrl(QUrl(FEEDBACK_FORM_URL))

    def _open_bug_report_form(self):
        QDesktopServices.openUrl(QUrl(BUG_REPORT_FORM_URL))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = HomeWidget()
    window.show()
    sys.exit(app.exec())
