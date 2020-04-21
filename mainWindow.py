#!/usr/bin/python3

import os,sys
import time

import cv2
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from ReadSagittalPD import ReadImage, ReadROI

from PyQt5.QtWidgets import QGridLayout, \
        QDesktopWidget, QMainWindow, QApplication, qApp, \
        QDockWidget, QWidget, QAction, QTableWidget, QTableWidgetItem, \
        QMenu, QPushButton, QLabel,\
        QVBoxLayout, QHBoxLayout, \
        QFileDialog, QMessageBox, QDialog, QRadioButton,  QButtonGroup
from PyQt5.QtCore import Qt, pyqtSlot, QSize

from volumeViewerWidget import volumeViewerWidget, volumeSliceViewerWidget
from featureDispWidget import featureDispWidget
from statAnalyzelWidget import statAnalyzeWidget
from clsABWidget import clsABWidget
from gradeDispWidget import gradeDispWidget
from curveWidget import curveWidget

from mainLogic import mainLogic

class dicomSelectDialog(QDialog):
    def __init__(self, images, parent=None):
        super(dicomSelectDialog, self).__init__(parent)
        self.__images = images
        self.btnOK = QPushButton('OK')
        self.btnQuit = QPushButton('Cancel')
        #self.table = QTableWidget(len(images), 3)
        hbox = QHBoxLayout()
        hbox.addStretch(1)
        hbox.addWidget(self.btnOK)
        hbox.addWidget(self.btnQuit)
        vbox = QVBoxLayout(self)
        #vbox.addWidget(self.table, stretch=1)
        vbox.addLayout(hbox)
        self.setLayout(vbox)
        #self.__makeTable()
        self.selectedIndex = None
        
        self.btnQuit.clicked.connect(self.reject)
        #self.btnOK.clicked.connect( \
        #        lambda x, self=self: self.done(self.table.currentRow()))
        self.btnOK.clicked.connect(self.btnOKClicked)

    def btnOKClicked(self, flag):
        self.selectedIndex = self.table.currentRow()
        self.accept()

    #def __makeTable(self):
        #labels = ['PatientID', 'SeriesDescription', 'Size']
        # self.table.setHorizontalHeaderLabels(labels)
        # for i, image in enumerate(self.__images):
        #     self.table.setItem(i, 0, \
        #             QTableWidgetItem(str(image.GetMetaData('0010|0020'))))
        #     self.table.setItem(i, 1, \
        #             QTableWidgetItem(str(image.GetMetaData('0008|103e'))))
        #     self.table.setItem(i, 2, \
        #             QTableWidgetItem(str(image.GetSize())))
        # self.table.resizeColumnsToContents()
        # self.table.adjustSize()


class controlPannelWidget(QWidget):
    def __init__(self, parent=None):
        super(controlPannelWidget, self).__init__(parent)
        self.btnLoad = QPushButton('Load Image')
        self.btnROI = QPushButton('Load ROI')
        #self.btnLoad.setToolTip('Load a dicom study from directory.')
        self.btnExt = QPushButton('Feature Extraction')
        self.btnSel = QPushButton('Feature Selection')
        # self.btnCla = QPushButton('Predict Grade')
        # self.btnRef = QPushButton('Get Reference')
        # self.btnSeg = QPushButton('Auto Segment')
        # self.btnSR = QPushButton('Enhance Resolution')

        vbox = QVBoxLayout(self)
        vbox.addWidget(self.btnLoad)
        vbox.addWidget(self.btnROI)
        vbox.addWidget(self.btnExt)
        vbox.addWidget(self.btnSel)
        # vbox.addWidget(self.btnSeg)
        # vbox.addWidget(self.btnCla)
        # vbox.addWidget(self.btnRef)
        self.setLayout(vbox)

class annotationPannelWidget(QWidget):
    def __init__(self, parent=None):
        super(annotationPannelWidget, self).__init__(parent)

        # self.bg1 = QButtonGroup(self)
        # self.bg1.addButton(self.rb11, 11)
        # self.bg1.addButton(self.rb12, 12)
        # self.bg1.addButton(self.rb13, 13)


        self.btnDoAnn = QPushButton('Draw Polygon')
        self.btnAccROI = QPushButton('Accept Drawing')
        self.btnClrSelROI = QPushButton('Clear Selected ROI')
        self.btnClrAllROI = QPushButton('Clear All ROI')
        self.btnSaveSelROI = QPushButton('Save Selected ROI')
        self.btnSaveAllROI = QPushButton('Save All ROI')

        self.rbS = QRadioButton('S', self)
        self.rbA = QRadioButton('A',self)
        self.rbC = QRadioButton('C',self)
        self.rbS.setChecked(True)
        childBox = QHBoxLayout()
        childBox.addWidget(self.rbS)
        childBox.addWidget(self.rbA)
        childBox.addWidget(self.rbC)

        vbox = QVBoxLayout(self)
        # vbox.addWidget(QLabel("Select AX"))
        # vbox.addSpacing(0)
        vbox.addLayout(childBox)
        vbox.addWidget(self.btnDoAnn)
        vbox.addWidget(self.btnAccROI)
        vbox.addWidget(self.btnClrSelROI)
        vbox.addWidget(self.btnClrAllROI)
        vbox.addWidget(self.btnSaveSelROI)
        vbox.addWidget(self.btnSaveAllROI)

        self.setLayout(vbox)

class mainWindow(QMainWindow):
    def __init__(self):
        super(mainWindow, self).__init__()
        self.__image = None # SimpleITK.Image
        self.__grade = None # Numpy array with shape [num_slice, num_class]
        self.child_stat = statAnalyzeWidget()
        self.cls_SVC = clsABWidget(ifSVC=True)
        self.cls_Logi = clsABWidget(ifSVC=False)
        self.statusBar().showMessage('Initializing UI...')
        self.initUI()
        self.statusBar().showMessage('Initializing Core...')
        self.main = mainLogic()
        self.statusBar().showMessage('Initializing Signal...')
        self.initSignals()
        self.statusBar().showMessage('Ready')

    
    def initUI(self):
        self.setWindowTitle('Radiomics System')
        screenGeometry = QApplication.desktop().screenGeometry()
        aspectRatio = 4/3.0
        blockLen = min(screenGeometry.width()/aspectRatio, \
                screenGeometry.height())
        self.resize(blockLen*aspectRatio, blockLen)
        #qr = self.frameGeometry()
        #cp = QDesktopWidget().availableGeometry().center()
        #qr.moveCenter(cp)
        #self.move(qr.topLeft())
        self.initUI_Menubar()
        self.initUI_Toolbar()
        self.initUI_ContextMenu()
        self.initUI_Layout()
        self.show()

    def initUI_Menubar(self):
        LogiRegAct = QAction('Logit Regression', self)
        LogiRegAct.triggered.connect(self.cls_Logi.show)

        SVCAct = QAction('Support Vector Classification', self)
        SVCAct.triggered.connect(self.cls_SVC.show)

        menubar = self.menuBar()
        fileMenu = menubar.addMenu('Classification')
        fileMenu.addAction(LogiRegAct)
        fileMenu.addAction(SVCAct)

        DenoiseAct = QAction('Wavelet Denoise', self)
        DenoiseAct.triggered.connect(self.actDenoise)

        GasSmoothAct = QAction('Guassian Smooth', self)
        GasSmoothAct.triggered.connect(self.actGasSmooth)

        MeanSmoothAct = QAction('Mean Smooth', self)
        MeanSmoothAct.triggered.connect(self.actMeanSmooth)

        MedSmoothAct = QAction('Median Smooth', self)
        MedSmoothAct.triggered.connect(self.actMedSmooth)

        ResampleAct = QAction('Resample', self)
        ResampleAct.triggered.connect(self.actResample)

        menubar = self.menuBar()
        fileMenu = menubar.addMenu('Pre-processing')
        fileMenu.addAction(DenoiseAct)
        fileMenu.addAction(GasSmoothAct)
        fileMenu.addAction(MeanSmoothAct)
        fileMenu.addAction(MedSmoothAct)
        fileMenu.addAction(ResampleAct)

        ThredAct = QAction('Threshold Seg', self)
        ThredAct.triggered.connect(self.actThreshold)

        DPSegAct = QAction('Deep Learning', self)
        DPSegAct.triggered.connect(self.actDPSeg)

        segbar = self.menuBar()
        segMenu = segbar.addMenu('Segmentation')
        segMenu.addAction(ThredAct)
        segMenu.addAction(DPSegAct)

        #loadStudyAct = QAction('Load Dicom Study', self)
        #fileMenu.addAction(loadStudyAct)
        #exitAct = QAction('Quit', self)
        #exitAct.triggered.connect(qApp.quit)
        #fileMenu.addAction(exitAct)

    def initUI_Toolbar(self):
        pass

    def initUI_ContextMenu(self):
        pass

    def initUI_Layout(self):
        self.volumeViewer = volumeViewerWidget(self)
        self.FeatureDisp = featureDispWidget(self)
        #self.dicomInfo = dicomInfoWidget(self)
        self.controlPanel = controlPannelWidget(self)
        self.annotationPanel = annotationPannelWidget(self)
        #self.FeatureSel = featureSelWidget(self)
        #self.gradeDisp = gradeDispWidget(self)
        # self.refViewer = curveWidget(self)
        # self.SRViewer = volumeViewerWidget(self)
        colormap = (plt.cm.bwr(np.array( \
                list([x for x in range(256)]), dtype=np.uint8 \
                ))*255).astype(np.uint8).reshape(256,1,4) #RGBA
        colormap[:,0,3] = list(range(255,0,-2))+list(range(1,256,2))
        # dock and central widget
        # volumeViewer
        self.setCentralWidget(self.volumeViewer)
        # FeatureExtract
        self.dockFeatureDisp = QDockWidget("Extracted Feature", self)
        self.dockFeatureDisp.setObjectName("dockFeatureDisp")
        self.addDockWidget(Qt.RightDockWidgetArea, self.dockFeatureDisp)
        self.dockFeatureDisp.setWidget(self.FeatureDisp)

        # Annoation
        self.dockAnnotation = QDockWidget("Annotation Tools", self)
        self.dockAnnotation.setObjectName("dockAnnotation")
        self.addDockWidget(Qt.RightDockWidgetArea, self.dockAnnotation)
        self.dockAnnotation.setWidget(self.annotationPanel)
        self.tabifyDockWidget(self.dockAnnotation, self.dockFeatureDisp)

        # FeatureSelection
        # self.dockFeatureSel = QDockWidget("Selected Feature", self)
        # self.dockFeatureSel.setObjectName("dockFeatureSel")
        # self.addDockWidget(Qt.RightDockWidgetArea, self.dockFeatureSel)
        # self.dockFeatureSel.setWidget(self.FeatureSel)
        # self.tabifyDockWidget(self.dockFeatureDisp, self.dockFeatureSel)

        # # dicomInfo
        # self.dockDicomInfo = QDockWidget("DICOM Info", self)
        # self.dockDicomInfo.setObjectName("dockDicomInfo")
        # self.addDockWidget(Qt.RightDockWidgetArea, self.dockDicomInfo)
        # self.dockDicomInfo.setWidget(self.dicomInfo)
        # # gradeDisp
        # self.dockGradeDisp = QDockWidget("Grade (Prediction)", self)
        # self.dockGradeDisp.setObjectName("dockGradeDisp")
        # self.addDockWidget(Qt.RightDockWidgetArea, self.dockGradeDisp)
        # self.dockGradeDisp.setWidget(self.gradeDisp)
        # self.tabifyDockWidget(self.dockDicomInfo, self.dockGradeDisp)

        # controlPanel
        self.dockControlPanel = QDockWidget("Control Panel", self)
        self.dockControlPanel.setObjectName("dockControlPanel")
        self.addDockWidget(Qt.RightDockWidgetArea, self.dockControlPanel)
        self.dockControlPanel.setWidget(self.controlPanel)
        self.dockControlPanel.showMinimized()
        self.resizeDocks([self.dockControlPanel, self.dockFeatureDisp], \
                [1,10], Qt.Vertical)
        # reference image and SR image
        # self.dockRefViewer = QDockWidget("Reference", self)
        # self.dockRefViewer.setObjectName("dockRefViewer")
        # self.addDockWidget(Qt.LeftDockWidgetArea, self.dockRefViewer)
        # self.dockRefViewer.setWidget(self.refViewer)
        # self.dockRefViewer.setFloating(True)
        # self.dockRefViewer.setVisible(False)
        # self.dockSRViewer = QDockWidget("Super-Resolution", self)
        # self.dockSRViewer.setObjectName("dockSRViewer")
        # self.addDockWidget(Qt.LeftDockWidgetArea, self.dockSRViewer)
        # self.dockSRViewer.setWidget(self.SRViewer)
        # self.dockSRViewer.setFloating(True)
        # self.dockSRViewer.setVisible(False)

    def initSignals(self):
        self.controlPanel.btnLoad.clicked.connect(self.actLoadStudy)
        self.controlPanel.btnROI.clicked.connect(self.actLoadROI)
        self.controlPanel.btnExt.clicked.connect(self.actFeatureExt)
        self.controlPanel.btnSel.clicked.connect(self.child_stat.show)

        self.annotationPanel.btnDoAnn.clicked.connect(self.actDoAnn)
        self.annotationPanel.btnAccROI.clicked.connect(self.actAccROI)
        self.annotationPanel.btnClrSelROI.clicked.connect(self.actClrSelROI)
        self.annotationPanel.btnClrAllROI.clicked.connect(self.actClrAllROI)
        self.annotationPanel.btnSaveSelROI.clicked.connect(self.actSaveSelROI)
        self.annotationPanel.btnSaveAllROI.clicked.connect(self.actSaveAllROI)

        # self.controlPanel.btnCla.clicked.connect(self.actGetGrade)
        # self.controlPanel.btnSeg.clicked.connect(self.actGetSeg)
        # self.controlPanel.btnRef.clicked.connect(self.actGetReference)
        # self.controlPanel.btnSR.clicked.connect(self.actGetSR)
        #self.volumeViewer.sliderIndex.valueChanged.connect( \
        #        lambda x: self.refViewer.setIndex(x-1))
        self.FeatureDisp.cellPressed.connect( \
                lambda row, col: self.volumeViewer.sliderIndex.setValue(row+1))


    def actDoAnn(self):
        if self.annotationPanel.rbS.isChecked()==True:
            self.volumeViewer.setROI(ax='S')
        if self.annotationPanel.rbA.isChecked() == True:
            self.volumeViewer.setROI(ax='A')
        if self.annotationPanel.rbC.isChecked() == True:
            self.volumeViewer.setROI(ax='C')
        return

    def actAccROI(self):
        self.volumeViewer.accROI()
        return

    def actSaveSelROI(self):
        self.volumeViewer.saveSelROI()
        return

    def actSaveAllROI(self):
        self.volumeViewer.saveAllROI()
        return

    def actClrSelROI(self):
        self.volumeViewer.clrSelROI()
        return

    def actClrAllROI(self):
        self.volumeViewer.clrAllROI()
        return

    def actDenoise(self):
        self.volumeViewer.denoise()
        return

    def actGasSmooth(self):
        self.volumeViewer.setPreprocessMethod(method='Gaussian')
        return

    def actMeanSmooth(self):
        self.volumeViewer.setPreprocessMethod(method='Mean')
        return

    def actMedSmooth(self):
        self.volumeViewer.setPreprocessMethod(method='Median')
        return

    def actResample(self):
        self.volumeViewer.setPreprocessMethod(method='Resample')
        return

    def actThreshold(self):
        self.volumeViewer.setSegMethod(method='Threshold')
        return

    def actDPSeg(self):
        pass

    @pyqtSlot()
    def actLoadStudy(self, directory=None):
        start = time.time()
        self.statusBar().showMessage('Loading Study...')
        directory = QFileDialog.getOpenFileName(self,
                                             "Select one file to open",
                                             "./",
                                             "Files (*.nii *.dcm)")
        print(directory)
        directory = str(directory[0])
        #if directory is None:
            #dialog = QFileDialog(self)
            # dialog.setFileMode(QFileDialog.DirectoryOnly)
            # dialog.setViewMode(QFileDialog.List)
            # dialog.setOption(QFileDialog.ShowDirsOnly, True)
            # if dialog.exec_():
            #     directory = str(dialog.selectedFiles()[0])
            # else:
            #     self.statusBar().showMessage( \
            #             'Ready ({:.2f}s)'.format(time.time() - start))
            #     return
        try:
            #images = ReadSagittalPDs(directory)
            image = ReadImage(directory)
        except Exception as err:
            msgBox = QMessageBox(self)
            msgBox.setText(str(type(err)) + str(err))
            msgBox.exec()
            self.statusBar().showMessage( \
                    'Ready ({:.2f}s)'.format(time.time() - start))
            return
        # selectDialog = dicomSelectDialog(images)
        # if selectDialog.exec_():
        #     image = images[selectDialog.selectedIndex]
        # else:
        #     self.statusBar().showMessage( \
        #             'Ready ({:.2f}s)'.format(time.time() - start))
        #     return
        self.main.setImage(image)
        self.volumeViewer.setImage(image)
        #self.dicomInfo.setImage(image)
        # self.dockDicomInfo.setVisible(True)
        # self.dockRefViewer.setVisible(False)
        # self.dockSRViewer.setVisible(False)
        # self.gradeDisp.setGrade(None)
        self.statusBar().showMessage( \
                'Ready ({:.2f}s)'.format(time.time() - start))
        return

    def actLoadROI(self, directory=None):
        start = time.time()
        self.statusBar().showMessage('Loading ROI...')
        directory = QFileDialog.getOpenFileName(self,
                                             "Select one file to open",
                                             "./",
                                             "Files (*.nii *.dcm)")
        print(directory)
        directory = str(directory[0])

        try:
            ROI = ReadROI(directory)
        except Exception as err:
            msgBox = QMessageBox(self)
            msgBox.setText(str(type(err)) + str(err))
            msgBox.exec()
            self.statusBar().showMessage( \
                    'Ready ({:.2f}s)'.format(time.time() - start))
            return

        self.main.setROI(ROI)
        self.volumeViewer.setLabel(ROI)
        #self.dicomInfo.setROI(ROI)

        self.statusBar().showMessage( \
                'Ready ({:.2f}s)'.format(time.time() - start))
        return

    def actFeatureExt(self):
        start = time.time()
        self.statusBar().showMessage('Extracting Feature...')
        try:
            feature_extracted = self.main.getFeature()
        except Exception as err:
            msgBox = QMessageBox(self)
            msgBox.setText(str(type(err)) + str(err))
            msgBox.exec()
            self.statusBar().showMessage( \
                    'Ready ({:.2f}s)'.format(time.time() - start))
            return
        self.FeatureDisp.setFeature(feature_extracted)
        self.dockFeatureDisp.setVisible(True)
        self.statusBar().showMessage( \
                'Ready ({:.2f}s)'.format(time.time() - start))

        return




    # def actFeatureSel(self):
    #     start = time.time()
    #     self.statusBar().showMessage('Statistical Analyzing...')
    #     directory = QFileDialog.getOpenFileName(self,
    #                                          "Select the feature folder",
    #                                          "./",
    #                                          "Files (*.nii *.dcm)")
    #     print(directory)
    #     directory = str(directory[0])
    #     try:
    #         feature_selected = self.main.featureSel()
    #     except Exception as err:
    #         msgBox = QMessageBox(self)
    #         msgBox.setText(str(type(err)) + str(err))
    #         msgBox.exec()
    #         self.statusBar().showMessage( \
    #                 'Ready ({:.2f}s)'.format(time.time() - start))
    #         return
    #     msgBox = QMessageBox(self)
    #     if feature_selected is True:
    #         msgBox.setText('Done! Please check the folder.')
    #     return


    @pyqtSlot()
    def actGetGrade(self):
        start = time.time()
        self.statusBar().showMessage('Predicting Grade...')
        try:
            grade = self.main.getGrade()
        except Exception as err:
            msgBox = QMessageBox(self)
            msgBox.setText(str(type(err)) + str(err))
            msgBox.exec()
            self.statusBar().showMessage( \
                    'Ready ({:.2f}s)'.format(time.time() - start))
            return
        self.gradeDisp.setGrade(grade)
        self.dockGradeDisp.setVisible(True)
        self.statusBar().showMessage( \
                'Ready ({:.2f}s)'.format(time.time() - start))
        msgBox = QMessageBox(self)
        msgBox.setText('得出诊断意见：'+
                '我们从四千多个膝关节测共振图层（健康人和患者）'+\
                '与医生的诊断结果中，利用深度学习的方法，'+\
                '提取图像的特征及其与诊断结果的关系。'+\
                '利用训练出的模型，'+\
                '计算机可以自动从图像中预测诊断意见。'+'\n'+\
                '提示水肿区域：'+\
                '我们从三百多层带有水肿区域的磁共振图像'+\
                '与医生手工勾勒出的水肿区域中，'+\
                '利用深度学习的方法提取特征。利用训练出的模型，'+\
                '计算机可以自动从图像中预测图像是否有水肿，'+\
                '以及水肿区域的位置。')
        #msgBox.exec()
        return

    @pyqtSlot()
    def actGetSeg(self):
        start = time.time()
        self.statusBar().showMessage('Predicting Segmentation...')
        try:
            seg = self.main.getSeg()
        except Exception as err:
            msgBox = QMessageBox(self)
            msgBox.setText(str(type(err)) + str(err))
            msgBox.exec()
            self.statusBar().showMessage( \
                    'Ready ({:.2f}s)'.format(time.time() - start))
            return
        self.volumeViewer.setLabel(seg)
        self.statusBar().showMessage( \
                'Ready ({:.2f}s)'.format(time.time() - start))
        msgBox = QMessageBox(self)
        msgBox.setText('生成分割结果：'+\
                '我们从100多例患者膝关节磁共振的手工标记结果中，'+\
                '利用深度学习的方法提取特征。利用训练出的模型，'+\
                '计算机可以自动从图像中得到膝关节结构的分割结果。')
        #msgBox.exec()

    @pyqtSlot()
    def actGetSR(self):
        start = time.time()
        self.statusBar().showMessage('Enhancing Resolution...')
        try:
            SR = self.main.getSuperResolution()
        except Exception as err:
            msgBox = QMessageBox(self)
            msgBox.setText(str(type(err)) + str(err))
            msgBox.exec()
            self.statusBar().showMessage( \
                    'Ready ({:.2f}s)'.format(time.time() - start))
            return
        self.SRViewer.setImage(SR)
        if not self.dockSRViewer.isVisible():
            self.dockSRViewer.setVisible(True)
            if self.dockSRViewer.isFloating():
                self.dockSRViewer.resize( \
                        self.SRViewer.viewerSlice.viewportSizeHint()*1.4)
        self.statusBar().showMessage( \
                'Ready ({:.2f}s)'.format(time.time() - start))
        msgBox = QMessageBox(self)
        msgBox.setText('分辨率增强（层间距）')
        #msgBox.exec()

    @pyqtSlot()
    def actGetReference(self):
        start = time.time()
        self.statusBar().showMessage('Predicting Reference...')
        try:
            #'PatientAge': '0010|1010', \
            img = self.main.getImage()
            self.refViewer.openCSV(os.path.join( \
                    os.path.dirname(__file__), 'plot.csv'))
            age = img.GetMetaData('0010|1010')
            age = int(age[0:3])
            thickness = self.main.getThickness()
        except Exception as err:
            msgBox = QMessageBox(self)
            msgBox.setText(str(type(err)) + str(err))
            msgBox.exec()
            self.statusBar().showMessage( \
                    'Ready ({:.2f}s)'.format(time.time() - start))
            return
        #print(age, *thickness)
        self.refViewer.plotCurve(age, *thickness)
        if not self.dockRefViewer.isVisible():
            self.dockRefViewer.setVisible(True)
            if self.dockRefViewer.isFloating():
                self.dockRefViewer.resize( \
                        self.refViewer.sizeHint()*1.1)
        self.statusBar().showMessage( \
                'Ready ({:.2f}s)'.format(time.time() - start))
        msgBox = QMessageBox(self)
        msgBox.setText('生成参考')
        #msgBox.exec()

if __name__ == '__main__':
    dicomPath = '/mnt/repo/privateData/cartilage_origin/FromXuhua/PD/A102752442'
    if len(sys.argv) > 1:
        dicomPath = sys.argv[1]
    app = QApplication(sys.argv)
    mainWindow = mainWindow()
    mainWindow.actLoadStudy(dicomPath)
    sys.exit(app.exec_())

