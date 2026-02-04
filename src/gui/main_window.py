import sys

from PySide6.QtCore import Slot
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import QApplication, QMainWindow, QStackedWidget

from src.gui.momentum_graph.momentum_graph_main_widget import MomentumGraphMainWidget

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

        self.stack = QStackedWidget(self)
        self.stack.currentChanged.connect(
            lambda idx: self.stack.widget(idx).update() and self.stack.clearFocus()
        )

        self.select_match_widget = SelectMatchWidget()
        self.stack.addWidget(self.select_match_widget)
        self.select_match_widget.selected.connect(self.on_match_selected)

        self.manage_match_widget = ManageMatchWidget()
        self.stack.addWidget(self.manage_match_widget)
        self.manage_match_widget.navigate_to_select_match.connect(
            lambda: self.stack.setCurrentWidget(self.select_match_widget)
        )

        self.momentum_graph_widget = MomentumGraphMainWidget()
        self.stack.addWidget(self.momentum_graph_widget)
        self.momentum_graph_widget.exit_requested.connect(
            self.on_navigate_to_manage_match
        )
        self.manage_match_widget.navigate_to_momentum_graph.connect(
            lambda: self.stack.setCurrentWidget(self.momentum_graph_widget)
        )

        self.setCentralWidget(self.stack)

    @Slot()
    def on_navigate_to_manage_match(self):
        self.stack.setCurrentWidget(self.manage_match_widget)
        self.manage_match_widget.initialise()

    @Slot(str)
    def on_match_selected(self, match_name: str):
        self.status.showMessage(f"Selected match: {match_name}")
        self.manage_match_widget.set_match(match_name)
        self.momentum_graph_widget.set_match(
            match_name,
        )
        self.on_navigate_to_manage_match()


if __name__ == "__main__":
    import cProfile

    def main():
        app = QApplication(sys.argv)
        window = MainWindow()
        window.show()
        sys.exit(app.exec())

    # Run the profiler and save stats to a file
    cProfile.run("main()", "profile.stats")
    # Load stats
    import pstats

    # Load stats
    stats = pstats.Stats("profile.stats")
    stats.strip_dirs()  # remove extraneous path info
    stats.sort_stats("tottime")  # sort by total time

    # Print only top 10 functions
    stats.print_stats(10)
