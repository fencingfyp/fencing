from PySide6.QtCore import QObject, Qt

from src.gui.util.actions_panel_widget import ActionsPanelWidget, TaskAction
from src.model.OpenCvUi import calculate_centrepoint


class FencerSelectionController(QObject):
    """
    Self-contained interactive fencer selector.
    Short-lived: shortcuts and mouse are created on activate, removed on deactivate.
    """

    def __init__(
        self, *, ui, video_label, action_panel: ActionsPanelWidget, on_done: callable
    ):
        super().__init__()
        self.ui = ui
        self.video_label = video_label
        self.action_panel = action_panel
        self.on_done = on_done

        # state
        self.candidates: dict[int, dict] = {}
        self.selected_id: int | None = None
        self.left: bool = True
        self._active = False

        # save original mouse handler
        self._orig_mouse_handler = video_label.mousePressEvent

    # ----------------------- lifecycle -----------------------
    def start(self, candidates: dict[int, dict], left: bool):
        if not candidates:
            self.on_done(None)
            return

        self.candidates = candidates
        self.left = left
        self.selected_id = None

        self.activate()
        self._render()

    def stop(self):
        self.deactivate()
        self.candidates = {}
        self.selected_id = None

    def activate(self):
        if self._active:
            return
        self._active = True

        # mouse
        self.video_label.setMouseTracking(True)
        self.video_label.mousePressEvent = self._on_mouse_click

        self._update_actions()

    def deactivate(self):
        if not self._active:
            return
        self._active = False

        # restore mouse
        self.video_label.mousePressEvent = self._orig_mouse_handler

        self.action_panel.clear()

    def _update_actions(self):
        side = "Left" if self.left else "Right"

        actions = []

        # Confirm only enabled if something selected
        actions.append(
            TaskAction(
                id="confirm_fencer",
                label="Confirm (W)",
                shortcut=Qt.Key.Key_W,
                callback=self._confirm,
                enabled=self.selected_id is not None,
            )
        )

        actions.append(
            TaskAction(
                id="skip_fencer",
                label="Skip (S)",
                shortcut=Qt.Key.Key_S,
                callback=self._skip,
            )
        )

        self.action_panel.set_actions(actions)

    # ----------------------- input handlers -----------------------
    def _on_mouse_click(self, event):
        if event.button() != Qt.LeftButton:
            return

        label_w, label_h = self.video_label.width(), self.video_label.height()
        frame = self.ui.get_current_frame()
        if frame is None:
            return

        fh, fw = frame.shape[:2]
        scale_x, scale_y = fw / label_w, fh / label_h

        x = event.position().x() * scale_x
        y = event.position().y() * scale_y

        # Closest candidate
        closest_candidate = min(
            self.candidates.values(),
            key=lambda c: (calculate_centrepoint(c)[0] - x) ** 2
            + (calculate_centrepoint(c)[1] - y) ** 2,
            default=None,
        )
        if closest_candidate:
            self.selected_id = closest_candidate["id"]

        self._update_actions()
        self._render()

    def _confirm(self):
        self.on_done(self.selected_id)
        self.stop()

    def _skip(self):
        self.on_done(None)
        self.stop()

    # ----------------------- rendering -----------------------
    def _render(self):
        side = "Left" if self.left else "Right"

        if self.selected_id is None:
            prompt = f"Select {side} Fencer: Click on a bounding box"
        else:
            prompt = f"{side} Fencer: ID {self.selected_id} (Press W to confirm)"

        self.ui.write(prompt)
        self.ui.draw_detections(self.candidates, highlight_id=self.selected_id)
