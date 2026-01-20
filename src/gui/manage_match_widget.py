import os
import sys

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QApplication, QWidget

from src.util.file_names import ORIGINAL_VIDEO_NAME

from .select_match_widget import MATCH_LIST_FOLDER
from .ui_manage_match_widget import Ui_ManageMatchWidget


class ManageMatchWidget(QWidget):
    navigate_to_select_match = Signal()
    navigate_to_momentum_graph = Signal()

    def __init__(self):
        super().__init__()

        self.ui = Ui_ManageMatchWidget()
        self.ui.setupUi(self)
        self.ui.backButton.clicked.connect(self.on_back_button_clicked)
        self.ui.momentumGraphButton.clicked.connect(
            self.on_momentum_graph_button_clicked
        )
        self.ui.heatMapButton.setEnabled(False)  # Not implemented yet
        self.ui.actionMapButton.setEnabled(False)  # Not implemented yet

    def set_match(self, match_name: str):
        self.ui.matchName.setText(match_name)
        self.ui.videoPlayerWidget.set_video_source(
            os.path.join(MATCH_LIST_FOLDER, match_name, ORIGINAL_VIDEO_NAME),
        )

    def on_back_button_clicked(self):
        self.ui.videoPlayerWidget.cleanup()
        self.navigate_to_select_match.emit()

    def on_momentum_graph_button_clicked(self):
        self.ui.videoPlayerWidget.cleanup()
        self.navigate_to_momentum_graph.emit()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    widget = ManageMatchWidget()
    widget.set_match("epee_1")
    widget.show()
    sys.exit(app.exec())
