import sys

from PySide6.QtCore import Slot
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QMainWindow,
    QStackedWidget,
    QWidget,
)

from src.gui.heat_map.heat_map_main_widget import navigation as heatmap_nav_register
from src.gui.home_widget import navigation as select_nav_register
from src.gui.manage_match_widget import navigation as manage_nav_register
from src.gui.momentum_graph.momentum_graph_main_widget import (
    navigation as momentum_nav_register,
)
from src.gui.multi_momentum_graph_widget import (
    navigation as multi_momentum_nav_register,
)
from src.gui.navbar.app_navigator import AppNavigator
from src.gui.navbar.global_navbar import GlobalNavbar
from src.gui.navbar.navigation_controller import View
from src.pyside.MatchContext import MatchContext


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # ---- window setup ----
        self.setWindowTitle("Fencing Analysis Tool")
        geometry = self.screen().availableGeometry()
        self.setGeometry(0, 0, geometry.width() * 0.8, geometry.height() * 0.9)
        self.statusBar()

        # ---- domain context ----
        self.match_ctx = MatchContext()

        # ---- navigation ----
        self.stack = QStackedWidget()
        self.nav = AppNavigator(self.stack)  # router/hashmap

        # ---- feature registration ----
        select_nav_register(self.nav, self.match_ctx)
        manage_nav_register(self.nav, self.match_ctx)
        momentum_nav_register(self.nav, self.match_ctx)
        heatmap_nav_register(self.nav, self.match_ctx)
        multi_momentum_nav_register(self.nav, self.match_ctx)

        # ---- navbar ----
        self.navbar = GlobalNavbar()
        self.nav.changed.connect(self.on_nav_changed)
        self.navbar.navigate.connect(lambda navnode: self.nav.navigate(navnode.view))

        # ---- layout ----
        self._setup_layout()

        # ---- domain-level wiring ----
        home_widget = self.nav.get_widget(View.HOME)
        home_widget.single_selected.connect(self.on_video_file_path_selected)
        home_widget.multiple_selected.connect(self.on_compare_video_file_paths_selected)

        # ---- start ----
        self.nav.navigate(View.HOME)

    # -------------------------------------------------
    # Navigation reaction
    # -------------------------------------------------
    def on_nav_changed(self, node):
        self.navbar.set_node(node, self.nav)
        self.stack.clearFocus()

    def _setup_layout(self):
        # ---- layout ----
        central = QWidget(self)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(self.navbar)
        separator = QFrame()
        separator.setFrameShape(QFrame.VLine)
        separator.setFrameShadow(QFrame.Sunken)
        layout.addWidget(separator)

        layout.addWidget(self.stack, 1)

        self.setCentralWidget(central)

    # ---- domain logic ----
    @Slot(str)
    def on_video_file_path_selected(self, file_path: str):
        self.statusBar().showMessage(f"Selected match: {file_path}")
        self.match_ctx.set_file(file_path)
        self.nav.navigate(View.MANAGE_MATCH)

    @Slot(list)
    def on_compare_video_file_paths_selected(self, file_paths: list[str]):
        self.statusBar().showMessage(f"Selected matches: {file_paths}")

        def create_match_context(file_path: str) -> MatchContext:
            ctx = MatchContext()
            ctx.set_file(file_path)
            return ctx

        match_contexts = [create_match_context(fp) for fp in file_paths]

        multi_momentum_widget = self.nav.get_widget(View.COMPARE_MOMENTUM)
        multi_momentum_widget.set_matches(match_contexts)

        self.nav.navigate(View.COMPARE_MOMENTUM)


if __name__ == "__main__":
    import cProfile
    import os

    import cv2

    def main():
        app = QApplication(sys.argv)
        window = MainWindow()
        window.show()
        cv2.setNumThreads(os.cpu_count())  # Set OpenCV to use all available threads
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
