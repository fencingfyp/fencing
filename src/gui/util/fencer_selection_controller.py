from PySide6.QtCore import QObject, Qt

from src.model.OpenCvUi import calculate_centrepoint


class FencerSelectionController(QObject):
    """
    Manages interactive fencer selection.
    UI just calls `start` and receives a callback when done.
    """

    def __init__(
        self,
        *,
        ui,  # PysideUi
        video_label,
        w_shortcut,
        s_shortcut,
        on_done: callable,
    ):
        super().__init__()
        self.ui = ui
        self.video_label = video_label
        self.w_shortcut = w_shortcut
        self.s_shortcut = s_shortcut
        self.on_done = on_done

        # state
        self.candidates: dict[int, dict] = {}
        self.selected_id: int | None = None
        self.left: bool = True

        # save original mouse handler
        self._orig_mouse_handler = video_label.mousePressEvent
        video_label.setMouseTracking(True)
        video_label.mousePressEvent = self._on_mouse_click

    # ---------- public API ----------

    def start(self, candidates: dict[int, dict], left: bool):
        if not candidates:
            self.on_done(None)
            return

        self.candidates = candidates
        self.left = left
        self.selected_id = None

        self._connect_inputs()
        self._render()

    def stop(self):
        self._disconnect_inputs()
        self.candidates = {}
        self.selected_id = None

    # ---------- input wiring ----------

    def _connect_inputs(self):
        self.w_shortcut.activated.connect(self._confirm)
        self.w_shortcut.setEnabled(True)

        self.s_shortcut.activated.connect(self._skip)
        self.s_shortcut.setEnabled(True)

        self.video_label.mousePressEvent = self._on_mouse_click

    def _disconnect_inputs(self):
        self.w_shortcut.activated.disconnect(self._confirm)
        self.w_shortcut.setEnabled(False)

        self.s_shortcut.activated.disconnect(self._skip)
        self.s_shortcut.setEnabled(False)

        self.video_label.mousePressEvent = self._orig_mouse_handler

    # ---------- handlers ----------

    def _on_mouse_click(self, event):
        if event.button() != Qt.LeftButton:  # Left click
            return

        label_w = self.video_label.width()
        label_h = self.video_label.height()
        frame = self.ui.get_current_frame()
        if frame is None:
            return

        fh, fw = frame.shape[:2]
        scale_x = fw / label_w
        scale_y = fh / label_h

        x = event.position().x() * scale_x
        y = event.position().y() * scale_y

        # Find closest candidate centrepoint
        closest_candidate = min(
            self.candidates.values(),
            key=lambda c: (calculate_centrepoint(c)[0] - x) ** 2
            + (calculate_centrepoint(c)[1] - y) ** 2,
            default=None,
        )
        if closest_candidate:
            self.selected_id = closest_candidate["id"]

        self._render()

    def _confirm(self):
        self.on_done(self.selected_id)
        self.stop()

    def _skip(self):
        self.on_done(None)
        self.stop()

    # ---------- rendering ----------

    def _render(self):
        side = "Left" if self.left else "Right"

        if self.selected_id is None:
            prompt = f"Select {side} Fencer: Click on a bounding box"
        else:
            prompt = f"{side} Fencer: ID {self.selected_id} (Press W to confirm)"

        self.ui.write(prompt)
        self.ui.draw_detections(self.candidates, highlight_id=self.selected_id)
