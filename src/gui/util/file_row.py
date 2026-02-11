from PySide6.QtCore import Signal
from PySide6.QtWidgets import QFileDialog, QHBoxLayout, QLineEdit, QPushButton, QWidget


class FileRow(QWidget):
    removed = Signal(QWidget)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.path_box = QLineEdit()
        self.path_box.setReadOnly(True)

        self.browse_btn = QPushButton("Browse")
        self.remove_btn = QPushButton("Remove")

        layout = QHBoxLayout(self)
        layout.addWidget(self.path_box)
        layout.addWidget(self.browse_btn)
        layout.addWidget(self.remove_btn)

        self.browse_btn.clicked.connect(self.pick_file)
        self.remove_btn.clicked.connect(lambda: self.removed.emit(self))

    def pick_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select video file",
            "",
            "Video Files (*.mp4)",
        )
        if file_path:
            self.path_box.setText(file_path)

    def get_path(self) -> str:
        return self.path_box.text().strip()
