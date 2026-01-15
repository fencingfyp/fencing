# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'manage_match_widget.ui'
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

from src.gui.video_player_widget import VideoPlayerWidget

class Ui_ManageMatchWidget(object):
    def setupUi(self, ManageMatchWidget):
        if not ManageMatchWidget.objectName():
            ManageMatchWidget.setObjectName(u"ManageMatchWidget")
        ManageMatchWidget.resize(870, 749)
        self.matchName = QLabel(ManageMatchWidget)
        self.matchName.setObjectName(u"matchName")
        self.matchName.setGeometry(QRect(130, 10, 511, 31))
        font = QFont()
        font.setPointSize(25)
        self.matchName.setFont(font)
        self.videoPlayerWidget = VideoPlayerWidget(ManageMatchWidget)
        self.videoPlayerWidget.setObjectName(u"videoPlayerWidget")
        self.videoPlayerWidget.setGeometry(QRect(10, 50, 851, 591))
        self.backButton = QPushButton(ManageMatchWidget)
        self.backButton.setObjectName(u"backButton")
        self.backButton.setGeometry(QRect(10, 10, 100, 32))

        self.retranslateUi(ManageMatchWidget)

        QMetaObject.connectSlotsByName(ManageMatchWidget)
    # setupUi

    def retranslateUi(self, ManageMatchWidget):
        ManageMatchWidget.setWindowTitle(QCoreApplication.translate("ManageMatchWidget", u"Form", None))
        self.matchName.setText(QCoreApplication.translate("ManageMatchWidget", u"matchName", None))
        self.backButton.setText(QCoreApplication.translate("ManageMatchWidget", u"Back", None))
    # retranslateUi

