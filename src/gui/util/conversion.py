import cv2
import numpy as np
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QLabel


def qlabel_to_np(label: QLabel) -> np.ndarray | None:
    pixmap = label.pixmap()
    if pixmap is None:
        return None

    qimg = pixmap.toImage().convertToFormat(QImage.Format_RGB888)

    w = qimg.width()
    h = qimg.height()
    bytes_per_line = qimg.bytesPerLine()

    ptr = qimg.bits()  # memoryview
    arr = np.frombuffer(ptr, np.uint8)

    arr = arr.reshape((h, bytes_per_line))[:, : w * 3]
    arr = arr.reshape((h, w, 3))
    arr = arr.copy()

    arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    return arr


def np_to_pixmap(image: np.ndarray) -> QPixmap:
    frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, ch = frame.shape
    bits_per_line = ch * w
    qimg = QImage(frame.data, w, h, bits_per_line, QImage.Format_RGB888)
    pixmap = QPixmap.fromImage(qimg)
    return pixmap
