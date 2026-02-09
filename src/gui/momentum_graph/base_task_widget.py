from PySide6.QtCore import Signal, Slot
from PySide6.QtWidgets import (
    QApplication,
    QLabel,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from src.gui.util.task_graph import MomentumGraphTasksToIds
from src.pyside.PysideUi import PysideUi


class BaseTaskWidget(QWidget):
    run_started = Signal(object)
    run_completed = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._content_stack = QStackedWidget(self)
        # self._content_stack.setStyleSheet("border: 2px solid red;")

        # ==================================================
        # Default / legacy UI widgets
        # ==================================================
        self.videoLabel = QLabel(self)
        self.videoLabel.setMinimumSize(1, 1)
        self.videoLabel.setStyleSheet("background: black;")

        self.uiTextLabel = QLabel(self)
        self.uiTextLabel.setWordWrap(True)

        self.runButton = QPushButton("Run", self)
        self.runButton.hide()

        self._default_view = QWidget(self)
        default_layout = QVBoxLayout(self._default_view)
        default_layout.setContentsMargins(0, 0, 0, 0)
        default_layout.setSpacing(8)

        default_layout.addWidget(self.videoLabel, stretch=1)
        default_layout.addWidget(self.uiTextLabel)
        default_layout.addWidget(self.runButton)

        self._content_stack.addWidget(self._default_view)

        # ==================================================
        # Root layout
        # ==================================================
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
        layout.addWidget(self._content_stack)
        self.setLayout(layout)

        # ==================================================
        # State
        # ==================================================
        self.video_label_original_size = self.videoLabel.size()
        self.cap = None
        self.working_dir = None
        self.controller = None
        self.is_running = False

        # ==================================================
        # Interactive UI adapter (unchanged)
        # ==================================================
        self.ui = PysideUi(
            video_label=self.videoLabel,
            text_label=self.uiTextLabel,
            parent=self,
        )

        # ==================================================
        # Signals
        # ==================================================
        self.runButton.clicked.connect(self.on_runButton_clicked)

    # --------------------------------------------------
    # Content management (NEW, extensible API)
    # --------------------------------------------------

    def show_default_ui(self):
        """Show the legacy video/text/run UI."""
        self._content_stack.setCurrentWidget(self._default_view)

    def register_widget(self, widget: QWidget):
        """Register a self-contained widget for later use."""
        if self._content_stack.indexOf(widget) == -1:
            self._content_stack.addWidget(widget)
            widget.hide()

    def show_widget(self, widget: QWidget):
        """Replace content area with a self-contained widget."""
        self._content_stack.setCurrentWidget(widget)
        widget.show()

    # --------------------------------------------------
    # Lifecycle
    # --------------------------------------------------

    def cancel(self):
        if self.controller and hasattr(self.controller, "cancel"):
            self.controller.cancel()
            if hasattr(self.controller, "deleteLater"):
                self.controller.deleteLater()
            self.controller = None

        self.ui.close_additional_windows()
        self.ui.cancel_running_subtasks()

    @Slot(str)
    def set_working_directory(self, working_dir: str):
        self.working_dir = working_dir
        self.setup()

    def setup(self):
        """Override in subclasses."""
        pass

    # --------------------------------------------------
    # Video sizing logic
    # --------------------------------------------------

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.ui and self.ui.video_renderer:
            self.ui.video_renderer._redraw()

    # --------------------------------------------------
    # Run handling
    # --------------------------------------------------

    @Slot()
    def on_runButton_clicked(self):
        pass

    def run_task(self):
        self.is_running = True

    # --------------------------------------------------
    # Qt cleanup
    # --------------------------------------------------

    def closeEvent(self, event):
        self.cancel()
        return super().closeEvent(event)
