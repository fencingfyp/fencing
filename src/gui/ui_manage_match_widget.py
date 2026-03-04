# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'manage_match_widget.ui'
##
## Created by: Qt User Interface Compiler version 6.10.1
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (
    QCoreApplication,
    QDate,
    QDateTime,
    QLocale,
    QMetaObject,
    QObject,
    QPoint,
    QRect,
    QSize,
    Qt,
    QTime,
    QUrl,
)
from PySide6.QtGui import (
    QBrush,
    QColor,
    QConicalGradient,
    QCursor,
    QFont,
    QFontDatabase,
    QGradient,
    QIcon,
    QImage,
    QKeySequence,
    QLinearGradient,
    QPainter,
    QPalette,
    QPixmap,
    QRadialGradient,
    QTransform,
)
from PySide6.QtWidgets import QApplication, QLabel, QPushButton, QSizePolicy, QWidget

from src.gui.util.video_player_widget import VideoPlayerWidget


class Ui_ManageMatchWidget(object):
    def setupUi(self, ManageMatchWidget):
        if not ManageMatchWidget.objectName():
            ManageMatchWidget.setObjectName("ManageMatchWidget")
        ManageMatchWidget.resize(870, 749)
        self.matchName = QLabel(ManageMatchWidget)
        self.matchName.setObjectName("matchName")
        self.matchName.setGeometry(QRect(10, 10, 511, 31))
        font = QFont()
        font.setPointSize(25)
        self.matchName.setFont(font)
        self.videoPlayerWidget = VideoPlayerWidget(ManageMatchWidget)
        self.videoPlayerWidget.setObjectName("videoPlayerWidget")
        self.videoPlayerWidget.setGeometry(QRect(10, 50, 851, 591))
        self.momentumGraphButton = QPushButton(ManageMatchWidget)
        self.momentumGraphButton.setObjectName("momentumGraphButton")
        self.momentumGraphButton.setGeometry(QRect(50, 660, 141, 51))
        self.heatMapButton = QPushButton(ManageMatchWidget)
        self.heatMapButton.setObjectName("heatMapButton")
        self.heatMapButton.setGeometry(QRect(210, 660, 141, 51))
        self.actionMapButton = QPushButton(ManageMatchWidget)
        self.actionMapButton.setObjectName("actionMapButton")
        self.actionMapButton.setGeometry(QRect(370, 660, 141, 51))

        self.retranslateUi(ManageMatchWidget)

        QMetaObject.connectSlotsByName(ManageMatchWidget)

    # setupUi

    def retranslateUi(self, ManageMatchWidget):
        ManageMatchWidget.setWindowTitle(
            QCoreApplication.translate("ManageMatchWidget", "Form", None)
        )
        self.matchName.setText(
            QCoreApplication.translate("ManageMatchWidget", "matchName", None)
        )
        self.momentumGraphButton.setText(
            QCoreApplication.translate("ManageMatchWidget", "Momentum Graph", None)
        )
        self.heatMapButton.setText(
            QCoreApplication.translate("ManageMatchWidget", "Heat Map", None)
        )
        self.actionMapButton.setText(
            QCoreApplication.translate("ManageMatchWidget", "Action Map", None)
        )
        # retranslateUi

        self.videoPlayerWidget.setObjectName("videoPlayerWidget")
        self.videoPlayerWidget.setGeometry(QRect(10, 50, 851, 591))
        self.momentumGraphButton = QPushButton(ManageMatchWidget)
        self.momentumGraphButton.setObjectName("momentumGraphButton")
        self.momentumGraphButton.setGeometry(QRect(50, 660, 141, 51))
        self.heatMapButton = QPushButton(ManageMatchWidget)
        self.heatMapButton.setObjectName("heatMapButton")
        self.heatMapButton.setGeometry(QRect(210, 660, 141, 51))
        self.actionMapButton = QPushButton(ManageMatchWidget)
        self.actionMapButton.setObjectName("actionMapButton")
        self.actionMapButton.setGeometry(QRect(370, 660, 141, 51))

        self.retranslateUi(ManageMatchWidget)

        QMetaObject.connectSlotsByName(ManageMatchWidget)

    # setupUi

    def retranslateUi(self, ManageMatchWidget):
        ManageMatchWidget.setWindowTitle(
            QCoreApplication.translate("ManageMatchWidget", "Form", None)
        )
        self.matchName.setText(
            QCoreApplication.translate("ManageMatchWidget", "matchName", None)
        )
        self.momentumGraphButton.setText(
            QCoreApplication.translate("ManageMatchWidget", "Momentum Graph", None)
        )
        self.heatMapButton.setText(
            QCoreApplication.translate("ManageMatchWidget", "Heat Map", None)
        )
        self.actionMapButton.setText(
            QCoreApplication.translate("ManageMatchWidget", "Action Map", None)
        )

    # retranslateUi
