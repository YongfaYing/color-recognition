# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'color-recognition.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1569, 1003)
        self.centralWidget = QtWidgets.QWidget(MainWindow)
        self.centralWidget.setObjectName("centralWidget")
        self.gridLayoutWidget = QtWidgets.QWidget(self.centralWidget)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(0, 30, 1061, 926))
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.gridLayoutWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.Original_Data = QtWidgets.QTabWidget(self.gridLayoutWidget)
        self.Original_Data.setEnabled(True)
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(11)
        self.Original_Data.setFont(font)
        self.Original_Data.setObjectName("Original_Data")
        self.tab_9 = QtWidgets.QWidget()
        self.tab_9.setMinimumSize(QtCore.QSize(0, 888))
        self.tab_9.setMaximumSize(QtCore.QSize(1283, 16777215))
        self.tab_9.setObjectName("tab_9")
        self.analysis_report_table = QtWidgets.QTableWidget(self.tab_9)
        self.analysis_report_table.setGeometry(QtCore.QRect(13, 13, 1011, 861))
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(18)
        self.analysis_report_table.setFont(font)
        self.analysis_report_table.setTabletTracking(False)
        self.analysis_report_table.setContextMenuPolicy(QtCore.Qt.DefaultContextMenu)
        self.analysis_report_table.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.analysis_report_table.setTextElideMode(QtCore.Qt.ElideMiddle)
        self.analysis_report_table.setObjectName("analysis_report_table")
        self.analysis_report_table.setColumnCount(4)
        self.analysis_report_table.setRowCount(12)
        item = QtWidgets.QTableWidgetItem()
        font = QtGui.QFont()
        font.setPointSize(1)
        font.setKerning(False)
        item.setFont(font)
        self.analysis_report_table.setVerticalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.analysis_report_table.setVerticalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.analysis_report_table.setVerticalHeaderItem(2, item)
        item = QtWidgets.QTableWidgetItem()
        self.analysis_report_table.setVerticalHeaderItem(3, item)
        item = QtWidgets.QTableWidgetItem()
        self.analysis_report_table.setVerticalHeaderItem(4, item)
        item = QtWidgets.QTableWidgetItem()
        self.analysis_report_table.setVerticalHeaderItem(5, item)
        item = QtWidgets.QTableWidgetItem()
        self.analysis_report_table.setVerticalHeaderItem(6, item)
        item = QtWidgets.QTableWidgetItem()
        self.analysis_report_table.setVerticalHeaderItem(7, item)
        item = QtWidgets.QTableWidgetItem()
        self.analysis_report_table.setVerticalHeaderItem(8, item)
        item = QtWidgets.QTableWidgetItem()
        self.analysis_report_table.setVerticalHeaderItem(9, item)
        item = QtWidgets.QTableWidgetItem()
        self.analysis_report_table.setVerticalHeaderItem(10, item)
        item = QtWidgets.QTableWidgetItem()
        self.analysis_report_table.setVerticalHeaderItem(11, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignCenter)
        font = QtGui.QFont()
        font.setFamily("Microsoft YaHei UI")
        font.setPointSize(18)
        font.setBold(True)
        font.setWeight(75)
        item.setFont(font)
        self.analysis_report_table.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        font = QtGui.QFont()
        font.setPointSize(18)
        font.setBold(True)
        font.setWeight(75)
        item.setFont(font)
        self.analysis_report_table.setHorizontalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        item.setFont(font)
        self.analysis_report_table.setHorizontalHeaderItem(2, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignCenter)
        font = QtGui.QFont()
        font.setPointSize(18)
        font.setBold(True)
        font.setWeight(75)
        item.setFont(font)
        self.analysis_report_table.setHorizontalHeaderItem(3, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignCenter)
        font = QtGui.QFont()
        font.setPointSize(18)
        item.setFont(font)
        self.analysis_report_table.setItem(0, 0, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignCenter)
        self.analysis_report_table.setItem(0, 1, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignCenter)
        font = QtGui.QFont()
        font.setPointSize(18)
        item.setFont(font)
        self.analysis_report_table.setItem(1, 0, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignCenter)
        self.analysis_report_table.setItem(1, 1, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignCenter)
        font = QtGui.QFont()
        font.setPointSize(18)
        item.setFont(font)
        self.analysis_report_table.setItem(2, 0, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignCenter)
        self.analysis_report_table.setItem(2, 1, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignCenter)
        font = QtGui.QFont()
        font.setPointSize(18)
        item.setFont(font)
        self.analysis_report_table.setItem(3, 0, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignCenter)
        self.analysis_report_table.setItem(3, 1, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignCenter)
        font = QtGui.QFont()
        font.setPointSize(18)
        item.setFont(font)
        self.analysis_report_table.setItem(4, 0, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignCenter)
        self.analysis_report_table.setItem(4, 1, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignCenter)
        font = QtGui.QFont()
        font.setPointSize(18)
        item.setFont(font)
        self.analysis_report_table.setItem(5, 0, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignCenter)
        self.analysis_report_table.setItem(5, 1, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignCenter)
        font = QtGui.QFont()
        font.setPointSize(18)
        item.setFont(font)
        self.analysis_report_table.setItem(6, 0, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignCenter)
        self.analysis_report_table.setItem(6, 1, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignCenter)
        font = QtGui.QFont()
        font.setPointSize(18)
        item.setFont(font)
        self.analysis_report_table.setItem(7, 0, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignCenter)
        self.analysis_report_table.setItem(7, 1, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignCenter)
        font = QtGui.QFont()
        font.setPointSize(18)
        item.setFont(font)
        self.analysis_report_table.setItem(8, 0, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignCenter)
        self.analysis_report_table.setItem(8, 1, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignCenter)
        font = QtGui.QFont()
        font.setPointSize(18)
        item.setFont(font)
        self.analysis_report_table.setItem(9, 0, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignCenter)
        self.analysis_report_table.setItem(9, 1, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignCenter)
        font = QtGui.QFont()
        font.setPointSize(18)
        item.setFont(font)
        self.analysis_report_table.setItem(10, 0, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignCenter)
        self.analysis_report_table.setItem(10, 1, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignCenter)
        font = QtGui.QFont()
        font.setPointSize(18)
        item.setFont(font)
        self.analysis_report_table.setItem(11, 0, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignCenter)
        self.analysis_report_table.setItem(11, 1, item)
        self.analysis_report_table.horizontalHeader().setVisible(True)
        self.analysis_report_table.horizontalHeader().setCascadingSectionResizes(False)
        self.analysis_report_table.horizontalHeader().setDefaultSectionSize(240)
        self.analysis_report_table.horizontalHeader().setSortIndicatorShown(False)
        self.analysis_report_table.horizontalHeader().setStretchLastSection(False)
        self.analysis_report_table.verticalHeader().setVisible(True)
        self.analysis_report_table.verticalHeader().setDefaultSectionSize(60)
        self.Original_Data.addTab(self.tab_9, "")
        self.gridLayout.addWidget(self.Original_Data, 1, 0, 1, 1)
        self.File_Path = QtWidgets.QTextBrowser(self.centralWidget)
        self.File_Path.setGeometry(QtCore.QRect(150, 0, 661, 31))
        self.File_Path.setObjectName("File_Path")
        self.start_run = QtWidgets.QPushButton(self.centralWidget)
        self.start_run.setGeometry(QtCore.QRect(810, -1, 131, 32))
        font = QtGui.QFont()
        font.setFamily("等线")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.start_run.setFont(font)
        self.start_run.setObjectName("start_run")
        self.Open_File = QtWidgets.QPushButton(self.centralWidget)
        self.Open_File.setGeometry(QtCore.QRect(0, -1, 151, 32))
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.Open_File.setFont(font)
        self.Open_File.setObjectName("Open_File")
        self.Export = QtWidgets.QPushButton(self.centralWidget)
        self.Export.setGeometry(QtCore.QRect(930, -1, 131, 32))
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(11)
        self.Export.setFont(font)
        self.Export.setObjectName("Export")
        MainWindow.setCentralWidget(self.centralWidget)

        self.retranslateUi(MainWindow)
        self.Original_Data.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        item = self.analysis_report_table.horizontalHeaderItem(0)
        item.setText(_translate("MainWindow", "位置"))
        item = self.analysis_report_table.horizontalHeaderItem(1)
        item.setText(_translate("MainWindow", "细菌种类"))
        item = self.analysis_report_table.horizontalHeaderItem(2)
        item.setText(_translate("MainWindow", "颜色取值"))
        item = self.analysis_report_table.horizontalHeaderItem(3)
        item.setText(_translate("MainWindow", "阴性/阳性"))
        __sortingEnabled = self.analysis_report_table.isSortingEnabled()
        self.analysis_report_table.setSortingEnabled(False)
        item = self.analysis_report_table.item(0, 0)
        item.setText(_translate("MainWindow", "1"))
        item = self.analysis_report_table.item(0, 1)
        item.setText(_translate("MainWindow", "细菌1"))
        item = self.analysis_report_table.item(1, 0)
        item.setText(_translate("MainWindow", "2"))
        item = self.analysis_report_table.item(1, 1)
        item.setText(_translate("MainWindow", "细菌2"))
        item = self.analysis_report_table.item(2, 0)
        item.setText(_translate("MainWindow", "3"))
        item = self.analysis_report_table.item(2, 1)
        item.setText(_translate("MainWindow", "细菌3"))
        item = self.analysis_report_table.item(3, 0)
        item.setText(_translate("MainWindow", "4"))
        item = self.analysis_report_table.item(3, 1)
        item.setText(_translate("MainWindow", "细菌4"))
        item = self.analysis_report_table.item(4, 0)
        item.setText(_translate("MainWindow", "5"))
        item = self.analysis_report_table.item(4, 1)
        item.setText(_translate("MainWindow", "细菌5"))
        item = self.analysis_report_table.item(5, 0)
        item.setText(_translate("MainWindow", "6"))
        item = self.analysis_report_table.item(5, 1)
        item.setText(_translate("MainWindow", "细菌6"))
        item = self.analysis_report_table.item(6, 0)
        item.setText(_translate("MainWindow", "7"))
        item = self.analysis_report_table.item(6, 1)
        item.setText(_translate("MainWindow", "细菌7"))
        item = self.analysis_report_table.item(7, 0)
        item.setText(_translate("MainWindow", "8"))
        item = self.analysis_report_table.item(7, 1)
        item.setText(_translate("MainWindow", "细菌8"))
        item = self.analysis_report_table.item(8, 0)
        item.setText(_translate("MainWindow", "9"))
        item = self.analysis_report_table.item(8, 1)
        item.setText(_translate("MainWindow", "细菌9"))
        item = self.analysis_report_table.item(9, 0)
        item.setText(_translate("MainWindow", "10"))
        item = self.analysis_report_table.item(9, 1)
        item.setText(_translate("MainWindow", "细菌10"))
        item = self.analysis_report_table.item(10, 0)
        item.setText(_translate("MainWindow", "11"))
        item = self.analysis_report_table.item(10, 1)
        item.setText(_translate("MainWindow", "细菌11"))
        item = self.analysis_report_table.item(11, 0)
        item.setText(_translate("MainWindow", "12"))
        item = self.analysis_report_table.item(11, 1)
        item.setText(_translate("MainWindow", "细菌12"))
        self.analysis_report_table.setSortingEnabled(__sortingEnabled)
        self.Original_Data.setTabText(self.Original_Data.indexOf(self.tab_9), _translate("MainWindow", "各孔分析报表"))
        self.File_Path.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'SimSun\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-size:9.07563pt;\"><br /></p></body></html>"))
        self.start_run.setText(_translate("MainWindow", "运行"))
        self.Open_File.setText(_translate("MainWindow", "打开文件"))
        self.Export.setText(_translate("MainWindow", "数据导出"))

