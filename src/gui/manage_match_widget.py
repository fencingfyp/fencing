import sys

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.gui.MatchContext import MatchContext
from src.gui.navbar.app_navigator import AppNavigator
from src.gui.navbar.navigation_controller import View
from src.model.FileManager import FileRole

from .tag_manager_widget import TagManagerWidget
from .taggable_video_player_widget import TaggableVideoPlayerWidget


def navigation(nav: AppNavigator, match_ctx: MatchContext):
    widget = ManageMatchWidget(match_ctx)
    nav.register(
        view=View.MANAGE_MATCH,
        title="Manage Match",
        widget=widget,
        parent=View.HOME,
    )


class ManageMatchWidget(QWidget):
    navigate = Signal(View)

    def __init__(self, ctx: MatchContext, parent=None):
        super().__init__(parent)
        self.ctx = ctx
        self.ctx.match_changed.connect(self.handle_match_changed)
        self._build_ui()
        self.actionMapButton.setEnabled(False)  # Not implemented yet

    def _build_ui(self):
        # Widgets
        self.matchName = QLabel("matchName")
        self.matchName.setFont(QFont("", 25))

        self.videoPlayerWidget = TaggableVideoPlayerWidget()

        self.momentumGraphButton = QPushButton("Momentum Graph")
        self.heatMapButton = QPushButton("Heat Map")
        self.actionMapButton = QPushButton("Action Map")

        # Tag panel
        self._tag_manager = TagManagerWidget()
        self._tag_manager.setFixedWidth(220)
        self.videoPlayerWidget.tag_requested.connect(self._tag_manager.add_tag)
        self._tag_manager.time_selected.connect(
            self.videoPlayerWidget.set_frame_position
        )

        # Button row
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.momentumGraphButton)
        button_layout.addWidget(self.heatMapButton)
        button_layout.addWidget(self.actionMapButton)
        button_layout.addStretch()

        # Left column
        left_layout = QVBoxLayout()
        left_layout.addWidget(self.matchName)
        left_layout.addWidget(self.videoPlayerWidget)
        left_layout.addLayout(button_layout)

        # Root layout
        root_layout = QHBoxLayout(self)
        root_layout.addLayout(left_layout)
        root_layout.addWidget(self._tag_manager)

        # Connections
        self.momentumGraphButton.clicked.connect(self.on_momentumGraphButton_clicked)
        self.heatMapButton.clicked.connect(self.on_heatMapButton_clicked)

    # -------- context reactions --------

    @Slot()
    def handle_match_changed(self):
        if not self.ctx.match_name:
            return
        self.matchName.setText(self.ctx.match_name)
        video_path = self.ctx.file_manager.get_original_video()
        print(f"Loading video preview from {video_path}")
        self.videoPlayerWidget.set_video_source(video_path)
        self._tag_manager.set_db_path(self.ctx.file_manager.get_path(FileRole.TAG_DB))

    @Slot()
    def on_momentumGraphButton_clicked(self):
        self.navigate.emit(View.MOMENTUM)

    @Slot()
    def on_heatMapButton_clicked(self):
        self.navigate.emit(View.HEAT_MAP)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    match_context = MatchContext()

    widget = ManageMatchWidget(match_context)
    match_context.set_file("matches_data/epee_3.mp4")
    widget.show()
    sys.exit(app.exec())
    sys.exit(app.exec())
