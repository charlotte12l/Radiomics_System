#!/usr/bin/python3

from PyQt5.QtWidgets import QWidget, QTableWidget, QTableWidgetItem
from PyQt5.QtCore import Qt

class featureSelWidget(QTableWidget):
    def __init__(self, parent=None):
        super(featureSelWidget, self).__init__(0, 2, parent)
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

