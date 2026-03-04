from PySide6.QtWidgets import QCheckBox, QPushButton, QVBoxLayout, QWidget


class PreRunPanel(QWidget):
    """
    A panel that houses pre-run options (checkboxes, etc.) and the Run button.
    Hidden by default; auto-shows when options or the run button are registered.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._checkboxes: dict[str, QCheckBox] = {}  # key -> QCheckBox

        self._options_layout = QVBoxLayout()
        self._options_layout.setContentsMargins(0, 0, 0, 0)
        self._options_layout.setSpacing(4)

        self.run_button = QPushButton("Run")
        self.run_button.hide()

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(6)
        root.addLayout(self._options_layout)
        root.addWidget(self.run_button)

        self.hide()

    # --------------------------------------------------
    # Checkbox API
    # --------------------------------------------------

    def add_checkbox(self, key: str, label: str, *, checked: bool = False) -> QCheckBox:
        """
        Add a checkbox option. Returns the QCheckBox so callers can connect
        signals or tweak properties further if needed.
        Automatically shows the panel and the Run button.
        """
        if key in self._checkboxes:
            return self._checkboxes[key]

        cb = QCheckBox(label)
        cb.setChecked(checked)
        self._checkboxes[key] = cb
        self._options_layout.addWidget(cb)

        self._auto_show()
        return cb

    def remove_checkbox(self, key: str):
        """Remove a checkbox by key."""
        cb = self._checkboxes.pop(key, None)
        if cb:
            self._options_layout.removeWidget(cb)
            cb.deleteLater()
        self._auto_hide_if_empty()

    def clear_checkboxes(self):
        """Remove all checkboxes."""
        for cb in self._checkboxes.values():
            self._options_layout.removeWidget(cb)
            cb.deleteLater()
        self._checkboxes.clear()
        self._auto_hide_if_empty()

    # --------------------------------------------------
    # Settings consumption
    # --------------------------------------------------

    def get_settings(self) -> dict[str, bool]:
        """
        Returns a snapshot of all checkbox states as {key: bool}.
        Call this when the Run button is pressed to read options before running.
        """
        return {key: cb.isChecked() for key, cb in self._checkboxes.items()}

    def checkbox_value(self, key: str) -> bool:
        """Convenience accessor for a single checkbox value."""
        cb = self._checkboxes.get(key)
        return cb.isChecked() if cb else False

    # --------------------------------------------------
    # Run button visibility helpers
    # --------------------------------------------------

    def show_run_button(self):
        self.run_button.show()
        self._auto_show()

    def hide_run_button(self):
        self.run_button.hide()
        self._auto_hide_if_empty()

    # --------------------------------------------------
    # Internal visibility management
    # --------------------------------------------------

    def _auto_show(self):
        self.show()

    def _auto_hide_if_empty(self):
        """Hide the panel when there's nothing to show."""
        has_checkboxes = bool(self._checkboxes)
        has_run_button = self.run_button.isVisible()
        if not has_checkboxes and not has_run_button:
            self.hide()
