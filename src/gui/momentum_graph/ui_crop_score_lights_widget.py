# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'crop_score_lights_widget.ui'
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

class Ui_CropScoreLightsWidget(object):
    def setupUi(self, CropScoreLightsWidget):
        if not CropScoreLightsWidget.objectName():
            CropScoreLightsWidget.setObjectName(u"CropScoreLightsWidget")
        CropScoreLightsWidget.resize(897, 726)
        self.backButton = QPushButton(CropScoreLightsWidget)
        self.backButton.setObjectName(u"backButton")
        self.backButton.setGeometry(QRect(10, 10, 100, 32))
        self.runButton = QPushButton(CropScoreLightsWidget)
        self.runButton.setObjectName(u"runButton")
        self.runButton.setGeometry(QRect(0, 660, 100, 32))
        self.uiTextLabel = QLabel(CropScoreLightsWidget)
        self.uiTextLabel.setObjectName(u"uiTextLabel")
        self.uiTextLabel.setGeometry(QRect(10, 610, 871, 41))
        self.videoLabel = QLabel(CropScoreLightsWidget)
        self.videoLabel.setObjectName(u"videoLabel")
        self.videoLabel.setGeometry(QRect(10, 50, 861, 561))

        self.retranslateUi(CropScoreLightsWidget)

        QMetaObject.connectSlotsByName(CropScoreLightsWidget)
    # setupUi

    def retranslateUi(self, CropScoreLightsWidget):
        CropScoreLightsWidget.setWindowTitle(QCoreApplication.translate("CropScoreLightsWidget", u"Form", None))
        self.backButton.setText(QCoreApplication.translate("CropScoreLightsWidget", u"Back", None))
        self.runButton.setText(QCoreApplication.translate("CropScoreLightsWidget", u"Run", None))
        self.uiTextLabel.setText(QCoreApplication.translate("CropScoreLightsWidget", u"Press \"Run\" to start cropping the score lights", None))
        self.videoLabel.setText("")
    # retranslateUi

