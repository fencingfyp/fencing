from typing import Callable

from .OpenCvUi import calculate_centrepoint
from .Ui import Ui


class InputController:
    def __init__(self, ui: Ui):
        self.ui = ui
        self._active = False

        self._click_handler = None
        self._confirm_handler = None
        self._skip_handler = None

    # --------------------------------------------------
    # Public API used by tasks
    # --------------------------------------------------

    def request_fencer_selection(
        self,
        *,
        candidates: dict[int, dict],
        left: bool,
        callback: Callable[[int | None], None],
    ) -> None:
        if self._active:
            raise RuntimeError("InputController already active")

        self._active = True
        selected_id = None

        side = "Left" if left else "Right"
        self.ui.set_status_text(
            f"Select {side} fencer (click to highlight, W=confirm, S=skip)"
        )

        # ---------------- Mouse ----------------

        def on_click(x: int, y: int):
            nonlocal selected_id
            closest = min(
                candidates.values(),
                key=lambda c: (calculate_centrepoint(c)[0] - x) ** 2
                + (calculate_centrepoint(c)[1] - y) ** 2,
                default=None,
            )
            if closest:
                selected_id = closest["id"]
                self.ui.draw_candidates(candidates)  # stateless redraw
                self.ui.set_status_text(f"{side} fencer: ID {selected_id} (W=confirm)")

        # ---------------- Keys ----------------

        def on_confirm():
            if selected_id is None:
                return
            self._finish(callback, selected_id)

        def on_skip():
            self._finish(callback, None)

        # ---------------- Bind ----------------

        self.ui.on_click(on_click)
        self.ui.on_key("W", on_confirm)
        self.ui.on_key("S", on_skip)

        self._click_handler = on_click
        self._confirm_handler = on_confirm
        self._skip_handler = on_skip

    def _finish(self, callback, result):
        self._unbind_all()
        self._active = False
        callback(result)

    def cancel(self):
        """Force-cancel any active input session."""
        if not self._active:
            return
        self._unbind_all()
        self._active = False

    def _unbind_all(self):
        self.ui.clear_input_handlers()
        self._click_handler = None
        self._confirm_handler = None
        self._skip_handler = None
