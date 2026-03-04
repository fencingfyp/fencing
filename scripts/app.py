import os
import sys

import cv2
from PySide6.QtWidgets import QApplication

from src.gui.main_window import MainWindow

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    cv2.setNumThreads(os.cpu_count())
    sys.exit(app.exec())
