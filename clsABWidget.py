#!/usr/bin/python3

from PyQt5.QtWidgets import QWidget,QDialog, \
    QMessageBox, QFileDialog, QDockWidget,QPushButton, \
    QTableWidget, QLineEdit,QTableWidgetItem, QVBoxLayout,\
    QGridLayout,QLabel

from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
from analysis.LogiRegre_AB import LogiRegre_AB
from analysis.SVC_AB import SVC_AB
import time
import cv2
from imageViewerWidget import imageViewer


class clsABWidget(QDialog):
    def __init__(self,ifSVC =False):
        super(clsABWidget,self).__init__()

        self.__A_addr = None
        self.__B_addr = None
        self.__save_addr = None

        self.SVC = ifSVC

        self.LogiRegre = LogiRegre_AB()
        self.SVC_AB = SVC_AB()

        self.layout = QGridLayout()

        self.A_addrLabel = QLabel("A Type Folder")
        self.A_addrLineEdit = QLineEdit("")
        self.A_btn = QPushButton('...')
        self.B_addrLabel = QLabel("B Type Folder")
        self.B_addrLineEdit = QLineEdit("")
        self.B_btn = QPushButton('...')
        self.save_addrLabel = QLabel("Save Folder")
        self.save_addrLineEdit = QLineEdit("")
        self.save_btn = QPushButton('...')
        self.ok_btn = QPushButton('OK')
        self.quit_btn = QPushButton('Cancel')

        #self.initUI()
        self.setWindowTitle('Classification')

        self.layout.addWidget(self.A_addrLabel, 0, 0)
        self.layout.addWidget(self.A_addrLineEdit, 0, 1)
        self.layout.addWidget(self.A_btn, 0, 2)
        self.layout.addWidget(self.B_addrLabel, 1, 0)
        self.layout.addWidget(self.B_addrLineEdit, 1, 1)
        self.layout.addWidget(self.B_btn, 1, 2)
        self.layout.addWidget(self.save_addrLabel, 2, 0)
        self.layout.addWidget(self.save_addrLineEdit, 2, 1)
        self.layout.addWidget(self.save_btn, 2, 2)


        self.layout.addWidget(self.ok_btn, 4,0)
        self.layout.addWidget(self.quit_btn, 4, 2)

        self.setLayout(self.layout)

        # self.dockRefViewer = QDockWidget("Reference", self)
        # self.dockRefViewer.setObjectName("dockRefViewer")
        # # self.addDockWidget(Qt.LeftDockWidgetArea, self.dockRefViewer)
        # self.dockRefViewer.setWidget(self.refViewer)
        # self.dockRefViewer.setFloating(True)
        # self.dockRefViewer.setVisible(False)

        self.A_btn.clicked.connect(self.ALoad)
        self.B_btn.clicked.connect(self.BLoad)
        self.save_btn.clicked.connect(self.saveLoad)
        self.ok_btn.clicked.connect(self.ok)
        self.quit_btn.clicked.connect(self.reject)

        # self.imageView = QLabel("add a image file")  # 得到一个QLabel的实例，并将它保存在成员imageView里，负责显示消息以及图片
        # self.imageView.setAlignment(Qt.AlignCenter)

        self.imageViewer = imageViewer()

    def set_A_addr(self,addr):
        self.__A_addr = addr

    def set_B_addr(self,addr):
        self.__B_addr = addr

    def set_save_addr(self,addr):
        self.__save_addr = addr

    def ALoad(self):
        dialog = QFileDialog(self)
        dialog.setFileMode(QFileDialog.DirectoryOnly)
        dialog.setViewMode(QFileDialog.List)
        dialog.setOption(QFileDialog.ShowDirsOnly, True)
        if dialog.exec_():
            directory = str(dialog.selectedFiles()[0])
        try:
            self.A_addrLineEdit.setText(directory)
            self.set_A_addr(directory)
        except Exception as err:
            msgBox = QMessageBox(self)
            msgBox.setText(str(type(err)) + str(err))
            msgBox.exec()
            # self.statusBar().showMessage( \
            #         'Ready ({:.2f}s)')
            return
        return

    def BLoad(self):
        dialog = QFileDialog(self)
        dialog.setFileMode(QFileDialog.DirectoryOnly)
        dialog.setViewMode(QFileDialog.List)
        dialog.setOption(QFileDialog.ShowDirsOnly, True)
        if dialog.exec_():
            directory = str(dialog.selectedFiles()[0])
        try:
            self.B_addrLineEdit.setText(directory)
            self.set_B_addr(directory)
        except Exception as err:
            msgBox = QMessageBox(self)
            msgBox.setText(str(type(err)) + str(err))
            msgBox.exec()
            # self.statusBar().showMessage( \
            #         'Ready ({:.2f}s)')
            return

        return

    def saveLoad(self):
        dialog = QFileDialog(self)
        dialog.setFileMode(QFileDialog.DirectoryOnly)
        dialog.setViewMode(QFileDialog.List)
        dialog.setOption(QFileDialog.ShowDirsOnly, True)
        if dialog.exec_():
            directory = str(dialog.selectedFiles()[0])
        try:
            self.save_addrLineEdit.setText(directory)
            self.set_save_addr(directory)
        except Exception as err:
            msgBox = QMessageBox(self)
            msgBox.setText(str(type(err)) + str(err))
            msgBox.exec()
            self.statusBar().showMessage( \
                    'Ready ({:.2f}s)')
            return

        return

    def ok(self):
        start = time.time()
        msgBox = QMessageBox(self)
        msgBox.setText('Statistical Analyzing...')
        # self.statusBar().showMessage('Statistical Analyzing...')
        try:
            if self.SVC:
                flag = self.SVC_AB(self.__A_addr,self.__B_addr,self.__save_addr)
                print('SVC!')
            else:
                flag = self.LogiRegre(self.__A_addr,self.__B_addr,self.__save_addr)
                print('Logi!')
        except Exception as err:
            msgBox = QMessageBox(self)
            msgBox.setText(str(type(err)) + str(err))
            msgBox.exec()
            return
        msgBox = QMessageBox(self)
        if flag is True:
            msgBox.setText('Done! Please check the folder.')
            # print('2')

        img_addr = self.__save_addr +'/'+ 'TYPE.png'


        img = cv2.imread(img_addr)
        # print(img)
        self.imageViewer.set_image(img)
        self.imageViewer.show()

        msgBox = QMessageBox(self)
        msgBox.setText('AUC')
        self.accept()
        return
    #def quit(self):








if __name__ == '__main__':
    import sys, os
    import numpy as np
    import time
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    ex = gradeDispWidget()
    ex.setGeometry(300, 300, 350, 350)
    ex.setWindowTitle('DicomInfoWidget')
    ex.show()
    ex.setGrade(np.array([[0.2,0.8],[0.1,0.9],[0.3,0.7],[0.6,0.4]]))
    sys.exit(app.exec_())

