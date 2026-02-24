from typing import override

from PySide6.QtCore import Signal, Slot
from PySide6.QtWidgets import QCheckBox, QLabel, QStackedWidget, QVBoxLayout, QWidget

from src.gui.instruction_label import InstructionLabel
from src.gui.pre_run_panel import PreRunPanel
from src.gui.util.actions_panel_widget import ActionsPanelWidget
from src.pyside.MatchContext import MatchContext
from src.pyside.PysideUi import PysideUi


class BaseTaskWidget(QWidget):
    run_started = Signal(object)
    run_completed = Signal(object)

    def __init__(self, match_context: MatchContext, parent=None):
        super().__init__(parent)
        self.match_context = match_context
        self._content_stack = QStackedWidget(self)

        # ==================================================
        # Default / legacy UI widgets
        # ==================================================
        self.videoLabel = QLabel(self)
        self.videoLabel.setMinimumSize(1, 1)
        self.videoLabel.setStyleSheet("background: black;")

        self.uiTextLabel = InstructionLabel(self)
        self.uiTextLabel.setWordWrap(True)

        # ==================================================
        # Pre-run panel (options + run button)
        # ==================================================
        self._pre_run_panel = PreRunPanel(self)
        # Expose run button at the top level for back-compat
        self.runButton = self._pre_run_panel.run_button

        self._default_view = QWidget(self)
        default_layout = QVBoxLayout(self._default_view)
        default_layout.setContentsMargins(0, 0, 0, 0)
        default_layout.setSpacing(8)
        default_layout.addWidget(self.videoLabel, stretch=1)
        default_layout.addWidget(self.uiTextLabel)
        default_layout.addWidget(self._pre_run_panel)

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
    # Pre-run options API (delegates to PreRunPanel)
    # --------------------------------------------------

    def add_run_option(
        self, key: str, label: str, *, checked: bool = False
    ) -> QCheckBox:
        """
        Register a checkbox option that appears above the Run button.
        The panel and Run button are shown automatically.

        Example (in subclass setup()):
            self.add_run_option("verbose", "Verbose output")
            self.add_run_option("dry_run", "Dry run", checked=True)
        """
        cb = self._pre_run_panel.add_checkbox(key, label, checked=checked)
        self._pre_run_panel.show_run_button()
        return cb

    def remove_run_option(self, key: str):
        """Remove a previously added option by key."""
        self._pre_run_panel.remove_checkbox(key)

    def clear_run_options(self):
        """Remove all options (checkboxes). Hides the panel if run button is also hidden."""
        self._pre_run_panel.clear_checkboxes()

    def get_run_settings(self) -> dict[str, bool]:
        """
        Consume the current state of all pre-run options.
        Call this inside on_runButton_clicked (or run_task) to read selections.

        Returns:
            dict mapping each option key to its checked state, e.g.
            {"verbose": True, "dry_run": False}
        """
        return self._pre_run_panel.get_settings()

    def show_run_button(self):
        """Explicitly show the Run button (and the panel)."""
        self._pre_run_panel.show_run_button()

    def hide_run_button(self):
        """Hide the Run button after it has been pressed (or before run starts)."""
        self._pre_run_panel.hide_run_button()

    # --------------------------------------------------
    # Content management
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
        self.clear_run_options()

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
        """Override in subclasses to configure options and run button."""
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
        """
        Override in subclasses. Call get_run_settings() here to read options,
        then call hide_run_button() if you want to hide the panel while running.

        Example:
            settings = self.get_run_settings()
            self.hide_run_button()
            self.run_task(verbose=settings["verbose"])
        """
        pass

    def run_task(self):
        self.is_running = True
        self._pre_run_panel.hide()

    def set_actions(self, actions: list[tuple[str, callable]]):
        """Replace the current action buttons. actions: list of (button_text, callback)."""
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
        return super().closeEvent(event)
