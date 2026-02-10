import sys

from PySide6.QtCore import Slot
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QMainWindow,
    QStackedWidget,
    QWidget,
)

from src.gui.heat_map.heat_map_main_widget import HeatMapMainWidget
from src.gui.momentum_graph.momentum_graph_main_widget import MomentumGraphMainWidget
from src.gui.navbar.global_navbar import GlobalNavbar
from src.gui.navbar.navigation_controller import NavigationController, NavNode, View
from src.pyside.MatchContext import MatchContext

from .manage_match_widget import ManageMatchWidget
from .select_match_widget import SelectMatchWidget


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Fencing Analysis Tool")
        geometry = self.screen().availableGeometry()
        self.setGeometry(0, 0, geometry.width() * 0.8, geometry.height() * 0.9)

        # ---- context ----
        self.match_ctx = MatchContext()

        # ---- menu ----
        file_menu = self.menuBar().addMenu("File")
        exit_action = QAction("Exit", self)
        exit_action.setShortcut(QKeySequence.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # ---- widgets ----
        self.select_match_widget = SelectMatchWidget()
        self.manage_match_widget = ManageMatchWidget(self.match_ctx)
        self.momentum_graph_widget = MomentumGraphMainWidget(self.match_ctx)
        self.heat_map_widget = HeatMapMainWidget(self.match_ctx)

        # ---- stack ----
        self.stack = QStackedWidget()
        for w in (
            self.select_match_widget,
            self.manage_match_widget,
            self.momentum_graph_widget,
            self.heat_map_widget,
        ):
            self.stack.addWidget(w)

        # ---- navigation tree ----
        self.nav_root = NavNode(None, "root")

        self.select_match_node = NavNode(
            View.SELECT_MATCH, "Select Match", self.select_match_widget, self.nav_root
        )

        self.manage_match_node = NavNode(
            View.MANAGE_MATCH,
            "Manage Match",
            self.manage_match_widget,
            self.select_match_node,
        )

        self.momentum_node = NavNode(
            View.MOMENTUM,
            "Momentum Graph",
            self.momentum_graph_widget,
            self.manage_match_node,
        )

        self.heat_map_node = NavNode(
            View.HEAT_MAP,
            "Heat Map",
            self.heat_map_widget,
            self.manage_match_node,
        )

        self.nav_root.children = [self.select_match_node]
        self.select_match_node.children = [self.manage_match_node]
        self.manage_match_node.children = [
            self.momentum_node,
            self.heat_map_node,
        ]

        # ---- navigation controller + navbar ----
        self.nav = NavigationController(self.nav_root)
        self.navbar = GlobalNavbar()

        self.nav.changed.connect(self.on_nav_changed)
        self.navbar.navigate.connect(self.nav.navigate)

        # ---- layout ----
        central = QWidget(self)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(self.navbar)
        layout.addWidget(self.stack, 1)

        self.setCentralWidget(central)

        # ---- signals from pages ----
        self.select_match_widget.selected.connect(self.on_video_file_path_selected)

        self.manage_match_widget.navigate_to_momentum_graph.connect(
            lambda: self.nav.navigate(self.momentum_node)
        )
        self.manage_match_widget.navigate_to_heat_map.connect(
            lambda: self.nav.navigate(self.heat_map_node)
        )

        self.momentum_graph_widget.exit_requested.connect(
            lambda: self.nav.navigate(self.manage_match_node)
        )
        self.heat_map_widget.exit_requested.connect(
            lambda: self.nav.navigate(self.manage_match_node)
        )

        # ---- start ----
        self.nav.navigate(self.select_match_node)

    # ---- navigation reaction ----
    def on_nav_changed(self, node: NavNode):
        if node.widget:
            self.stack.setCurrentWidget(node.widget)
        self.navbar.set_node(node, self.nav)
        self.stack.clearFocus()

    # ---- domain logic ----
    @Slot(str)
    def on_video_file_path_selected(self, file_path: str):
        self.statusBar().showMessage(f"Selected match: {file_path}")
        self.match_ctx.set_file(file_path)
        self.nav.navigate(self.manage_match_node)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


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
