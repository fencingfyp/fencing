from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QRadioButton,
    QVBoxLayout,
)

from src.gui.util.tag import Tag

SUBCATEGORIES = {
    "Action": ["Attack", "Counterattack", "Parry", "Riposte", "General"],
    "Other": ["Injury", "Timeout", "Other"],
}


class AddTagDialog(QDialog):
    def __init__(self, time_sec: int, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add Tag")
        minutes = int(time_sec / 1000 // 60)
        seconds = int(time_sec / 1000 % 60)
        self._description = QLineEdit()

        # Category radio buttons
        self._category_group = QButtonGroup(self)
        category_layout = QHBoxLayout()
        for i, category in enumerate(SUBCATEGORIES.keys()):
            rb = QRadioButton(category)
            self._category_group.addButton(rb, i)
            category_layout.addWidget(rb)
            if i == 0:
                rb.setChecked(True)

        # Subcategory radio buttons
        self._subcategory_group = QButtonGroup(self)
        self._subcategory_layout = QHBoxLayout()

        self._category_group.idClicked.connect(self._on_category_changed)
        self._on_category_changed(0)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        form = QFormLayout()
        form.addRow(QLabel(f"Timestamp: {minutes:02}:{seconds:02}"))
        form.addRow("Category:", category_layout)
        form.addRow("Subcategory:", self._subcategory_layout)
        form.addRow("Description:", self._description)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)

    def _on_category_changed(self, id: int):
        # Clear existing subcategory buttons
        for btn in self._subcategory_group.buttons():
            self._subcategory_group.removeButton(btn)
            self._subcategory_layout.removeWidget(btn)
            btn.deleteLater()

        category = self._category_group.button(id).text()
        for i, subcategory in enumerate(SUBCATEGORIES.get(category, [])):
            rb = QRadioButton(subcategory)
            self._subcategory_group.addButton(rb, i)
            self._subcategory_layout.addWidget(rb)
            if i == 0:
                rb.setChecked(True)

    def result_tag(self, frame_idx: int, time_msec: int) -> Tag:
        category_btn = self._category_group.checkedButton()
        subcategory_btn = self._subcategory_group.checkedButton()
        return Tag(
            frame_idx=frame_idx,
            time_msec=time_msec,
            description=self._description.text().strip(),
            category=category_btn.text() if category_btn else "",
            subcategory=subcategory_btn.text() if subcategory_btn else "",
        )
