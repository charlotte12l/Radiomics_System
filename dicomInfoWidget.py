#!/usr/bin/python3

from PyQt5.QtWidgets import QWidget, QTableWidget, QTableWidgetItem

dicomTags = { \
        "PatientName" : "0010|0010", \
        "PatientID" : "0010|0020", \
        "AccessionNumber" : "0008|0050", \
        'PatientSex': '0010|0040', \
        'PatientAge': '0010|1010', \
        'PatientBirthData': '0010|0030', \
        'PatientWeight' : '0010|1030', \
        'StudyData' : '0008|0020', \
        'SeriesDate' : '0008|0021', \
        'SeriesTime' : '0008|0031', \
        'Modality' : '0008|0060', \
        'Manufacturer' : '0008|0070'}

class dicomInfoWidget(QTableWidget):
    def __init__(self, parent=None):
        super(dicomInfoWidget, self).__init__(len(dicomTags), 3, parent)
        self.setHorizontalHeaderLabels(['Tag Name', 'Values', 'Tag ID'])
        self.__makeTable(None)

    def setImage(self, image=None):
        self.__makeTable(image)

    def __makeTable(self, image=None):
        for i, tagName in enumerate(dicomTags.keys()):
            tagID = dicomTags[tagName]
            tagVal = 'N/A'
            if (image is not None) and (tagID in image.GetMetaDataKeys()):
                tagVal = image.GetMetaData(tagID)
            self.setItem(i, 0, QTableWidgetItem(tagName))
            self.setItem(i, 1, QTableWidgetItem(tagVal))
            self.setItem(i, 2, QTableWidgetItem(tagID))

if __name__ == '__main__':
    import sys, os
    from PyQt5.QtWidgets import QApplication, QStyleFactory
    app = QApplication(sys.argv)
    #app.setStyle(QStyleFactory.create('Breeze'))
    ex = dicomInfoWidget()
    ex.setGeometry(300, 300, 350, 350)
    ex.setWindowTitle('DicomInfoWidget')
    ex.show()
    sys.exit(app.exec_())

