import os
import shutil
import sys

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QApplication,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QWidget,
)

from src.gui.util.io import load_ui_dynamic
from src.util.file_names import ORIGINAL_VIDEO_NAME

from .add_match_dialog import AddMatchDialog

MATCH_LIST_FOLDER = "matches_data/"


class SelectMatchWidget(QWidget):
    selected = Signal(str)  # match_name

    def __init__(self):
        super().__init__()

        # Load .ui
        self.ui_file = "src/gui/select_match_widget.ui"
        self.ui = load_ui_dynamic(self.ui_file, self)

        self.list_widget = self.ui.findChild(QListWidget, "listWidget")
        self.add_button = self.ui.findChild(QPushButton, "pushButton")
        self.input_line = self.ui.findChild(QLineEdit, "lineEdit")

        self.add_button.clicked.connect(self._add_item)
        self.list_widget.itemDoubleClicked.connect(self._on_item_clicked)
        self.update()

    def _add_item(self):
        dialog = AddMatchDialog(self)
        dialog.submitted.connect(self._on_match_submitted)
        dialog.open()

    def _on_item_clicked(self, item: QListWidgetItem):
        match_name = item.text()
        print("Selected match:", match_name)
        self.selected.emit(match_name)

    def _add_to_match_list(self, name: str):
        item = QListWidgetItem(name)
        item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.list_widget.addItem(item)

    def _on_match_submitted(self, name: str, file_path: str):
        self._create_new_match(name, file_path)
        self.update()

    def _create_new_match(self, name: str, file_path: str):
        match_folder = os.path.join(MATCH_LIST_FOLDER, name)
        os.makedirs(match_folder, exist_ok=True)
        dest_path = os.path.join(match_folder, ORIGINAL_VIDEO_NAME)
        shutil.copy(file_path, dest_path)

    def update(self):
        super().update()
        folder_names = self._get_eligible_folders()
        self.list_widget.clear()
        for folder_name in folder_names:
            self._add_to_match_list(folder_name)

    @staticmethod
    def _get_eligible_folders():
        folder_names = []
        if os.path.exists(MATCH_LIST_FOLDER):
            folder_names = [
                name
                for name in os.listdir(MATCH_LIST_FOLDER)
                if SelectMatchWidget._is_eligible_folder(name)
            ]
        # Sort by last modified time, newest first
        folder_names.sort(
            key=lambda name: os.path.getmtime(os.path.join(MATCH_LIST_FOLDER, name)),
            reverse=True,
        )
        return folder_names

    @staticmethod
    def _is_eligible_folder(name: str) -> bool:
        full_path = os.path.join(MATCH_LIST_FOLDER, name)
        if (
            not os.path.exists(full_path)
            or not os.path.isdir(full_path)
            or ORIGINAL_VIDEO_NAME not in os.listdir(full_path)
        ):
            return False
        return True


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SelectMatchWidget()
    window.show()
    sys.exit(app.exec())
