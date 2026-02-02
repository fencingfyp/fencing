# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'base_task_widget.ui'
##
## Created by: Qt User Interface Compiler version 6.10.1
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QLabel, QPushButton, QSizePolicy,
    QWidget)

class Ui_BaseTaskWidget(object):
    def setupUi(self, BaseTaskWidget):
        if not BaseTaskWidget.objectName():
            BaseTaskWidget.setObjectName(u"BaseTaskWidget")
        BaseTaskWidget.resize(897, 726)
        self.runButton = QPushButton(BaseTaskWidget)
        self.runButton.setObjectName(u"runButton")
        self.runButton.setGeometry(QRect(0, 660, 100, 32))
        self.uiTextLabel = QLabel(BaseTaskWidget)
        self.uiTextLabel.setObjectName(u"uiTextLabel")
        self.uiTextLabel.setGeometry(QRect(10, 610, 871, 41))
        self.videoLabel = QLabel(BaseTaskWidget)
        self.videoLabel.setObjectName(u"videoLabel")
        self.videoLabel.setGeometry(QRect(10, 10, 861, 601))

        self.retranslateUi(BaseTaskWidget)

        QMetaObject.connectSlotsByName(BaseTaskWidget)
    # setupUi

    def retranslateUi(self, BaseTaskWidget):
        BaseTaskWidget.setWindowTitle(QCoreApplication.translate("BaseTaskWidget", u"Form", None))
        self.runButton.setText(QCoreApplication.translate("BaseTaskWidget", u"Run", None))
        self.uiTextLabel.setText(QCoreApplication.translate("BaseTaskWidget", u"Press \"Run\" to start the task", None))
        self.videoLabel.setText("")
    # retranslateUi

