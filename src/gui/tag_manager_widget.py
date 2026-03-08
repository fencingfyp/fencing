from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
)

from src.gui.util.tag import Tag
from src.gui.util.tag_store import TagStore

SUBCATEGORIES = {
    "Action": ["Attack", "Counterattack", "Parry", "Riposte", "General"],
}


class AddTagDialog(QDialog):

    def __init__(self, frame_idx: int, time_msec: float, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add Tag")

        minutes = int(time_msec / 1000 // 60)
        seconds = int(time_msec / 1000 % 60)

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

    def build_tag(self, frame_idx: int, time_msec: float) -> Tag:
        return Tag(
            frame_idx=frame_idx,
            time_msec=time_msec,
            description=self._description.text().strip(),
            category=self._category.currentText(),
            subcategory=self._subcategory.currentText(),
        )


class TagManagerWidget(QFrame):

    time_selected = Signal(int)  # frame_idx, for seeking on click

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.StyledPanel)
        self.setFixedWidth(220)

        self._store = None  # initialized in set_db_path

        title = QLabel("Tags")
        title.setAlignment(Qt.AlignCenter)

        self._list = QListWidget()
        self._list.itemClicked.connect(self._on_item_clicked)

        self._delete_button = QPushButton("Delete")
        self._delete_button.clicked.connect(self._on_delete)

        button_row = QHBoxLayout()
        button_row.addStretch()
        button_row.addWidget(self._delete_button)

        layout = QVBoxLayout(self)
        layout.addWidget(title)
        layout.addWidget(self._list)
        layout.addLayout(button_row)

    # ------------------------------------------------------------------ public

    def add_tag(self, tag: Tag):
        """Persist and display a new tag."""
        saved = self._store.add(tag)
        self._insert_item(saved)

    def set_db_path(self, db_path: str):
        """Change the underlying tag store and refresh."""
        if self._store:
            self._store.close()
        self._store = TagStore(db_path)
        self._load_all()

    # ------------------------------------------------------------------ internal

    def _load_all(self):
        self._list.clear()
        for tag in self._store.all():
            self._insert_item(tag)

    def _insert_item(self, tag: Tag):
        minutes = int(tag.time_msec / 1000 // 60)
        seconds = int(tag.time_msec / 1000 % 60)
        display = f"{minutes:02}:{seconds:02} [{tag.subcategory}]"
        if tag.description:
            display += f" {tag.description}"
        item = QListWidgetItem(display)
        item.setData(Qt.UserRole, tag.id)
        self._list.addItem(item)

    def _on_item_clicked(self, item: QListWidgetItem):
        tag = self._store.get(item.data(Qt.UserRole))
        if tag:
            self.time_selected.emit(tag.frame_idx)

    def _on_delete(self):
        item = self._list.currentItem()
        if item is None:
            return
        tag_id = item.data(Qt.UserRole)
        self._store.remove(tag_id)
        self._list.takeItem(self._list.row(item))

    def closeEvent(self, event):
        if self._store:
            self._store.close()
        super().closeEvent(event)
