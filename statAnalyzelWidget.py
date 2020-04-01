#!/usr/bin/python3

from PyQt5.QtWidgets import QWidget,QDialog, \
    QMessageBox, QFileDialog, QPushButton, QTableWidget, QLineEdit,QTableWidgetItem, QVBoxLayout,QGridLayout,QLabel
from PyQt5.QtCore import Qt
from analysis.FeatureSelection import FeatureSel
import time


class statAnalyzeWidget(QDialog):
    def __init__(self):
        super(statAnalyzeWidget,self).__init__()

        self.__A_addr = None
        self.__B_addr = None
        self.__save_addr = None

        self.FeatureSel = FeatureSel()

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
        self.setWindowTitle('Statistical Analysis')

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

        self.A_btn.clicked.connect(self.ALoad)
        self.B_btn.clicked.connect(self.BLoad)
        self.save_btn.clicked.connect(self.saveLoad)
        self.ok_btn.clicked.connect(self.ok)
        self.quit_btn.clicked.connect(self.reject)

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
            flag = self.FeatureSel(self.__A_addr,self.__B_addr,self.__save_addr)
            print('done!')
        except Exception as err:
            msgBox = QMessageBox(self)
            msgBox.setText(str(type(err)) + str(err))
            msgBox.exec()
            self.statusBar().showMessage( \
                    'Ready ({:.2f}s)'.format(time.time() - start))
            return
        msgBox = QMessageBox(self)
        if flag is True:
            msgBox.setText('Done! Please check the folder.')
            # print('2')

        self.accept()
        return
    #def quit(self):








        directory = QFileDialog.getOpenFileName(self,
                                             "Select one file to open",
                                             "./",
                                             "Files (*.nii *.dcm)")
        print(directory)
        directory = str(directory[0])


        #self.setHorizontalHeaderLabels(['Grade', 'Confidence'])

    # def setGrade(self, probs=None):
    #     '''probs: 2D numpy array of size [slices_number, classes_number]
    #     '''
    #     if probs is None:
    #         self.setRowCount(0)
    #         return
    #     classesNumber = 2
    #     assert probs.shape[1] == classesNumber
    #     self.setRowCount(probs.shape[0])
    #     probs = probs.tolist()
    #     for i, prob in enumerate(probs):
    #         confidence = max(prob)
    #         grade = prob.index(confidence)
    #         gradeItem = QTableWidgetItem(str(grade))
    #         self.setItem(i, 0, gradeItem)
    #         if grade > 0:
    #             gradeItem.setBackground(Qt.blue)
    #         self.setItem(i, 1, QTableWidgetItem(format(confidence, '.2f')))
    #     return

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

