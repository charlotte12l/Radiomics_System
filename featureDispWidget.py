#!/usr/bin/python3

from PyQt5.QtWidgets import QWidget, QTableWidget, QTableWidgetItem
from PyQt5.QtCore import Qt

class featureDispWidget(QTableWidget):
    def __init__(self, parent=None):
        super(featureDispWidget, self).__init__(0, 2, parent)
        self.setHorizontalHeaderLabels(['Feature', 'Value'])

    def setFeature(self, results =None):
        '''probs: 2D numpy array of size [slices_number, classes_number]
        '''
        if results is None:
            self.setRowCount(0)
            return

        list = sorted(results.items())
        name, value = zip(*list)
        #classesNumber = 2
        #assert values.shape[1] == classesNumber
        self.setRowCount(len(name)-20)
        #probs = values.tolist()
        for i in range(len(name)-20):
            #confidence = max(prob)
            #grade = prob.index(confidence)
            nameItem = QTableWidgetItem(str(name[i+20]))
            self.setItem(i, 0, nameItem)
            #if grade > 0:
                #gradeItem.setBackground(Qt.blue)
            #self.setItem(i, 1, QTableWidgetItem(format(value[i], '.8f')))
            self.setItem(i, 1, QTableWidgetItem(str(value[i+20])))
        return

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

