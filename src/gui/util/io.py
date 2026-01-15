import sys

from PySide6.QtCore import QFile, QIODevice
from PySide6.QtUiTools import QUiLoader
from PySide6.QtWidgets import QWidget


def load_ui_dynamic(ui_file: str, parent: QWidget = None) -> QWidget:
    """Load a .ui file dynamically and return the root widget."""
    file = QFile(ui_file)
    if not file.open(QIODevice.ReadOnly):
        raise IOError(f"Cannot open {ui_file}: {file.errorString()}")

    loader = QUiLoader()
    ui = loader.load(file, parent)
    file.close()
    return ui
