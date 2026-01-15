import sys

from PySide6.QtCore import Slot
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import QApplication, QMainWindow, QStackedWidget

from .manage_match_widget import ManageMatchWidget
from .select_match_widget import SelectMatchWidget


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Fencing Analysis Tool")
        geometry = self.screen().availableGeometry()
        self.setGeometry(
            0, 0, geometry.width() * 0.8, geometry.height() * 0.9
        )  # Magic numbers for now

        # Status Bar
        self.status = self.statusBar()

        # Menu
        self.menu = self.menuBar()
        self.file_menu = self.menu.addMenu("File")

        # Exit QAction
        exit_action = QAction("Exit", self)
        exit_action.setShortcut(QKeySequence.Quit)
        exit_action.triggered.connect(self.close)
        self.file_menu.addAction(exit_action)

        # Status Bar
        self.status = self.statusBar()

        self.stack = QStackedWidget()

        self.select_match_widget = SelectMatchWidget()
        self.manage_match_widget = ManageMatchWidget()
        self.stack.addWidget(self.select_match_widget)
        self.stack.addWidget(self.manage_match_widget)
        self.stack.currentChanged.connect(lambda idx: self.stack.widget(idx).update())
        self.select_match_widget.selected.connect(self.on_match_selected)
        self.manage_match_widget.navigate_to_select_match.connect(
            lambda: self.stack.setCurrentWidget(self.select_match_widget)
        )

        self.setCentralWidget(self.stack)

    @Slot(str)
    def on_match_selected(self, match_name: str):
        self.status.showMessage(f"Selected match: {match_name}")
        self.manage_match_widget.set_match(match_name)
        self.stack.setCurrentWidget(self.manage_match_widget)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
