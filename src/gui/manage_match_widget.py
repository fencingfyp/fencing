import sys

from PySide6.QtCore import Signal, Slot
from PySide6.QtWidgets import QApplication, QWidget

from src.gui.navbar.app_navigator import AppNavigator
from src.gui.navbar.navigation_controller import View
from src.pyside.MatchContext import MatchContext

from .ui_manage_match_widget import Ui_ManageMatchWidget


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

        self.ui = Ui_ManageMatchWidget()
        self.ui.setupUi(self)

        self.ui.actionMapButton.setEnabled(False)  # Not implemented yet

    # -------- context reactions --------
    @Slot()
    def handle_match_changed(self):  # on_* triggers autowiring
        if not self.ctx.match_name:
            return

        self.ui.matchName.setText(self.ctx.match_name)

        video_path = self.ctx.file_manager.get_original_video()
        print(f"Loading video preview from {video_path}")
        self.ui.videoPlayerWidget.set_video_source(video_path)

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
    match_context.set_file("matches_data/epee_3/epee_3.mp4")
    widget.show()
    sys.exit(app.exec())
