# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'crop_scoreboard_widget.ui'
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

class Ui_CropScoreboardWidget(object):
    def setupUi(self, CropScoreboardWidget):
        if not CropScoreboardWidget.objectName():
            CropScoreboardWidget.setObjectName(u"CropScoreboardWidget")
        CropScoreboardWidget.resize(897, 726)
        self.backButton = QPushButton(CropScoreboardWidget)
        self.backButton.setObjectName(u"backButton")
        self.backButton.setGeometry(QRect(10, 10, 100, 32))
        self.runButton = QPushButton(CropScoreboardWidget)
        self.runButton.setObjectName(u"runButton")
        self.runButton.setGeometry(QRect(0, 660, 100, 32))
        self.uiTextLabel = QLabel(CropScoreboardWidget)
        self.uiTextLabel.setObjectName(u"uiTextLabel")
        self.uiTextLabel.setGeometry(QRect(10, 610, 871, 41))
        self.videoLabel = QLabel(CropScoreboardWidget)
        self.videoLabel.setObjectName(u"videoLabel")
        self.videoLabel.setGeometry(QRect(10, 50, 861, 561))

        self.retranslateUi(CropScoreboardWidget)

        QMetaObject.connectSlotsByName(CropScoreboardWidget)
    # setupUi

    def retranslateUi(self, CropScoreboardWidget):
        CropScoreboardWidget.setWindowTitle(QCoreApplication.translate("CropScoreboardWidget", u"Form", None))
        self.backButton.setText(QCoreApplication.translate("CropScoreboardWidget", u"Back", None))
        self.runButton.setText(QCoreApplication.translate("CropScoreboardWidget", u"Run", None))
        self.uiTextLabel.setText(QCoreApplication.translate("CropScoreboardWidget", u"Press \"Run\" to start cropping the scoreboard", None))
        self.videoLabel.setText("")
    # retranslateUi

