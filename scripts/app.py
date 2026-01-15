import sys

from PySide6.QtWidgets import QApplication

from gui.select_match_widget import SelectMatchWidget

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SelectMatchWidget("src/gui/main_window.ui")
    window.ui.show()
    sys.exit(app.exec())
