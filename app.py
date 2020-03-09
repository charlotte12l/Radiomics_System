#!/usr/bin/python3
import sys, os
sys.path.append('E:\\Study\\thesis\\xk\\cartilage_atlas')

from PyQt5.QtCore import QCoreApplication
from PyQt5.QtWidgets import QApplication, QMainWindow, QDockWidget
from PyQt5.QtCore import QTimer, Qt, QSettings, QByteArray, QPoint, QSize

import mainWindow
import qdarkstyle

# ref: https://github.com/ColinDuquesnoy/QDarkStyleSheet/ \
#        blob/master/example/example.py
def write_settings(window):
    """Get window settings and write it into a file."""
    settings = QSettings()
    settings.setValue('pos', window.pos())
    settings.setValue('size', window.size())
    settings.setValue('state', window.saveState())

def read_settings(window):
    """Read and set window settings from a file."""
    settings = QSettings()
    pos = settings.value('pos', window.pos(), type='QPoint')
    size = settings.value('size', window.size(), type='QSize')
    state = settings.value('state', window.saveState(), type='QByteArray')
    window.restoreState(state)
    window.resize(size)
    window.move(pos)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())

    QCoreApplication.setApplicationName( \
            "Cartilage Evaluation System")
    QCoreApplication.setOrganizationName( \
            "MIC@SJTU")

    mainWindow = mainWindow.mainWindow()
    #read_settings(mainWindow)
    #mainWindow.showMaximized()
    retVal = app.exec_()
    #write_settings(mainWindow)
    sys.exit(retVal)
