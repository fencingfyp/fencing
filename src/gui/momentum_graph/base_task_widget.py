from typing import override

from PySide6.QtCore import Property, QEasingCurve, QPropertyAnimation, Signal, Slot
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from src.gui.util.actions_panel_widget import ActionsPanelWidget
from src.pyside.MatchContext import MatchContext
from src.pyside.PysideUi import PysideUi


class InstructionLabel(QLabel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Overlay
        self._highlight = QWidget(self)
        self._highlight.setStyleSheet(
            """
            background-color: rgba(255, 255, 0, 180);
            border-radius: 4px;
        """
        )
        self._highlight.hide()

        # Opacity effect
        self._effect = QGraphicsOpacityEffect(self._highlight)
        self._highlight.setGraphicsEffect(self._effect)

        # Animation
        self._anim = QPropertyAnimation(self._effect, b"opacity")
        self._anim.setDuration(1200)
        self._anim.setEasingCurve(QEasingCurve.OutCubic)
        self._anim.finished.connect(self._highlight.hide)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._highlight.setGeometry(self.rect())

    def setText(self, text):
        super().setText(text)
        self.animate_highlight()

    def animate_highlight(self):
        self._highlight.show()
        self._effect.setOpacity(1.0)

        self._anim.stop()
        self._anim.setStartValue(1.0)
        self._anim.setEndValue(0.0)
        self._anim.start()


class BaseTaskWidget(QWidget):
    run_started = Signal(object)
    run_completed = Signal(object)

    def __init__(self, match_context: MatchContext, parent=None):
        super().__init__(parent)
        self.match_context = match_context
        self._content_stack = QStackedWidget(self)
        # self._content_stack.setStyleSheet("border: 2px solid red;")

        # ==================================================
        # Default / legacy UI widgets
        # ==================================================
        self.videoLabel = QLabel(self)
        self.videoLabel.setMinimumSize(1, 1)
        self.videoLabel.setStyleSheet("background: black;")

        self.uiTextLabel = InstructionLabel(self)
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
        self._action_panel = ActionsPanelWidget(self)
        layout.addWidget(self._action_panel)
        self.setLayout(layout)

        # ==================================================
        # State
        # ==================================================
        self.video_label_original_size = self.videoLabel.size()
        self.cap = None
        self.controller = None
        self.is_running = False

        # ==================================================
        # Interactive UI adapter
        # ==================================================
        self.ui = PysideUi(
            video_label=self.videoLabel,
            text_label=self.uiTextLabel,
            action_panel=self._action_panel,
            parent=self,
        )

        # ==================================================
        # Signals
        # ==================================================
        self.runButton.clicked.connect(self.on_runButton_clicked)
        self.match_context.match_changed.connect(self.on_match_context_changed)

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

        self.ui.cancel_running_subtasks()

    def cleanup(self):
        self.ui.close_additional_windows()

    @Slot()
    def on_match_context_changed(self):
        self._setup_done = False

    @override
    def showEvent(self, event):
        if not getattr(self, "_setup_done", False):
            self.setup()
            self._setup_done = True
        return super().showEvent(event)

    @override
    def hideEvent(self, event):
        self._setup_done = False
        if self.is_running:
            self.cancel()
        self.cleanup()
        return super().hideEvent(event)

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

    def set_actions(self, actions: list[tuple[str, callable]]):
        """
        Replace the current action buttons.
        actions: list of (button_text, callback)
        """
        self._action_panel.set_actions(actions)

    def clear_actions(self):
        """Remove all action buttons."""
        self._action_panel.clear()

    # --------------------------------------------------
    # Qt cleanup
    # --------------------------------------------------

    def closeEvent(self, event):
        if self.is_running:
            self.cancel()
        self.cleanup()
        return super().closeEvent(event)
