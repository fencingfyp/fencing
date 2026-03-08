from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QVBoxLayout,
)

from src.gui.util.tag import Tag

SUBCATEGORIES = {
    "Action": ["Attack", "Counterattack", "Parry", "Riposte", "General"],
}


class AddTagDialog(QDialog):

    def __init__(self, time_sec: int, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add Tag")

        minutes = int(time_sec / 1000 // 60)
        seconds = int(time_sec / 1000 % 60)

        self._description = QLineEdit()

        self._category = QComboBox()
        self._category.addItems(SUBCATEGORIES.keys())
        self._category.currentTextChanged.connect(self._on_category_changed)

        self._subcategory = QComboBox()
        self._on_category_changed(self._category.currentText())

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        form = QFormLayout()
        form.addRow(QLabel(f"Timestamp: {minutes:02}:{seconds:02}"))
        form.addRow("Description:", self._description)
        form.addRow("Category:", self._category)
        form.addRow("Subcategory:", self._subcategory)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)

    def _on_category_changed(self, category: str):
        self._subcategory.clear()
        self._subcategory.addItems(SUBCATEGORIES.get(category, []))

    def result_tag(self, frame_idx: int, time_msec: int) -> Tag:
        return Tag(
            frame_idx=frame_idx,
            time_msec=time_msec,
            description=self._description.text().strip(),
            category=self._category.currentText(),
            subcategory=self._subcategory.currentText(),
        )
