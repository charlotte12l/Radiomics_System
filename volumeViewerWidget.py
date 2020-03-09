#!/usr/bin/python3

from PyQt5.QtWidgets import QWidget, QSlider, QLabel, QScrollArea, \
        QHBoxLayout, QVBoxLayout, QSizePolicy
from PyQt5.QtGui import QImage, QPixmap, QPalette
from PyQt5.QtCore import Qt, pyqtSlot, QPoint

import numpy as np
import SimpleITK as sitk
import cv2


class scalableLabel(QScrollArea):
    scaleMin = 0.2
    scaleMax = 10
    wheelScale = 1/480.0
    def __init__(self, parent=None):
        super(scalableLabel, self).__init__(parent)
        self.imageLabel = QLabel(parent)
        self.imageLabel.setBackgroundRole(QPalette.Base)
        self.imageLabel.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.imageLabel.setScaledContents(True)

        self.setBackgroundRole(QPalette.Dark)
        self.setWidget(self.imageLabel)
        self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.__pixmap = None
        self.__scaleFactor = 1 # defalut scaleFactor
        self.__mouseLeftPressing = False
        self.__mouseLeftPos = QPoint(0,0)

    def setScale(self, scale):
        scale = np.clip(scale, self.scaleMin, self.scaleMax)
            
        if self.__pixmap is None:
            return
        self.__scaleFactor = scale
        scaledPixmap = self.__pixmap.scaled(self.__pixmap.size() * scale)
        self.imageLabel.setPixmap(scaledPixmap)
        self.imageLabel.adjustSize()

    def getScale(self):
        return self.__scaleFactor

    def setPixmap(self, pixmap):
        self.__pixmap = pixmap
        scaledPixmap = self.__pixmap.scaled(\
                self.__pixmap.size() * self.__scaleFactor)
        self.imageLabel.setPixmap(scaledPixmap)
        self.imageLabel.adjustSize()

    def wheelEvent(self, event):
        self.setScale(self.getScale() + event.angleDelta().y()*self.wheelScale)

    def mousePressEvent(self, event):
        if (event.button() == Qt.LeftButton):
            self.__mouseLeftPressing = True
            self.__mouseLeftPressPos = event.pos()
        else:
            super(scalableLabel, self).mousePressEvent(event)
            
    def mouseMoveEvent(self, event):
        if self.__mouseLeftPressing and (event.buttons() & Qt.LeftButton):
            currentPos = event.pos()
            hScrollBar = self.horizontalScrollBar()
            vScrollBar = self.verticalScrollBar()
            viewportSize = self.viewport().size()
            delta = currentPos - self.__mouseLeftPressPos
            self.__mouseLeftPressPos = currentPos
            hScrollBar.setValue(hScrollBar.value() - delta.x())
            vScrollBar.setValue(vScrollBar.value() - delta.y())
        else:
            super(scalableLabel, self).mouseMoveEvent(event)
    
    def mouseReleaseEvent(self, event):
        if (event.button() == Qt.LeftButton):
            self.__mouseLeftPressing = False
        else:
            super(scalableLabel, self).mouseReleaseEvent(event)


class volumeSliceViewerWidget(scalableLabel):
    defaultLabelColormap = np.array([ \
            [0  ,0  ,0  ], \
            [0  ,0  ,255], \
            [0  ,255,0  ], \
            [255,0  ,0  ], \
            [0  ,255,255], \
            [255,255,0  ], \
            [255,0  ,255]] + \
            [[0  ,0  ,0 ]]*(256-7), dtype=np.uint8).reshape((256, 1, 3))

    def __init__(self, parent=None, \
            image=None, label=None, \
            index=0, opacity=1, window=[None, None], \
            colormap=None, \
            labelColormap=None):
        '''
        display one slice of a volume
        image: sitk.Image, 3D image only
        window: if None, will be set to [minPixel, maxPixel]
        '''
        super(volumeSliceViewerWidget, self).__init__(parent)
        # widget components
        #self.labelSlice = AspectRatioPixmapLabel(self)
        # init variables
        #assert dim in [0,1,2], "dim="+str(dim)+"is not valid for 3D image"
        #self.__dim = dim
        self.__colormap = colormap
        if labelColormap is None:
            labelColormap = self.defaultLabelColormap
        self.__labelColormap = labelColormap
        # vars updated in setWindow
        self.__window = [None, None]
        # vars updated in setImage
        self.__image = None
        self.__imageArray = None
        self.__labelArray = None
        # vars updated in setIndex
        self.__index = 0
        # vars updated in setOpacity
        self.__opacity = 1
        # execute param
        self.setWindow(window)
        if image is not None:
            self.setImage(image)
            self.setIndex(index)
            if label is not None:
                self.setOpacity(opacity)
                self.setLabel(label)

    def setImage(self, image):
        assert image.GetDimension() == 3, "accepts 3D image only"
        self.__image = image
        self.__label = None
        self.__index = 0
        self.__updateImageArray()
        self.__updateLabelArray()
        self.__updatePixmap()

    def setLabel(self, label):
        if self.__image is None:
            return
        assert label.GetSize() == self.__image.GetSize()
        self.__label = label
        self.__updateLabelArray()
        self.__updatePixmap()

    def setIndex(self, index):
        if self.__image is None:
            return
        index = int(index)
        assert index >= 0 and index < self.__image.GetDepth(), \
                "index ("+str(index)+") out of range"
        self.__index=index
        self.__updatePixmap()

    def setOpacity(self, opacity):
        assert opacity >=0 and opacity <=1, "opacity out of range"
        self.__opacity = opacity
        if self.__image is None or self.__label is None:
            return
        self.__updatePixmap()

    def setWindow(self, window):
        self.__window = window
        if self.__image is None:
            return
        self.__updateImageArray()
        self.__updatePixmap()

    def setColormap(self, colormap):
        self.__colormap = colormap
        if self.__image is None:
            return
        self.__updatePixmap()

    def setLabelColormap(self, labelColormap):
        if labelColormap is None:
            labelColormap = self.defaultLabelColormap
        self.__labelColormap = labelColormap
        if self.__image is None or self.__label is None:
            return
        self.__updatePixmap()

    def __updateImageArray(self):
        array = sitk.GetArrayFromImage(self.__image)
        if self.__window[0] is None:
            minVal = array.min()
        else:
            minVal = self.__window[0]
        if self.__window[1] is None:
            maxVal = array.max()
        else:
            maxVal = self.__window[1]
        self.__imageArray = np.clip( \
                (array - np.float32(minVal))/np.float32(maxVal-minVal), 0, 1)
        self.__imageArray = (255*self.__imageArray).astype(np.uint8)

    def __updateLabelArray(self):
        if self.__label is None:
            self.__labelArray = None
        else:
            array = sitk.GetArrayFromImage(self.__label)
            self.__labelArray = array.astype(np.uint8)
            #minVal = array.min()
            #maxVal = array.max()
            #self.__labelArray = (array - np.float32(minVal))/np.float32(maxVal-minVal)
            #self.__labelArray = (255*self.__labelArray).astype(np.uint8)

    def __updatePixmap(self):
        image = self.__imageArray[self.__index,:,:]
        if self.__colormap is None:
            image = np.stack([image, image, image], axis=-1)
        else:
            image = cv2.applyColorMap(image, self.__colormap)
        if self.__labelArray is not None:
            label = self.__labelArray[self.__index,:,:]
            if self.__labelColormap.shape[-1] == 4:
                alphaMap = self.__labelColormap[:,:,-1].reshape(256,1,1)
            else:
                alphaMap = np.array([0]+[255]*255, \
                        dtype=np.uint8).reshape(256,1,1)
            labelColormap = self.__labelColormap[:,:,0:3]
            alpha = cv2.applyColorMap(label, alphaMap).astype(np.float32)/255
            alpha = alpha * self.__opacity
            label = cv2.applyColorMap(label, labelColormap).astype(np.float32)
            array = image*(1-alpha)+label*alpha
            array = array.astype(np.uint8)
        else:
            array = image
        array = array[:,:,::-1]
        img = QImage(array.copy().data, \
                array.shape[1], array.shape[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(img)
        self.setPixmap(pixmap)

class SliderWithTextWidget(QWidget):
    def __init__(self, parent=None, text=None):
        super(SliderWithTextWidget, self).__init__(parent)
        if text is not None:
            self.labelText = QLabel(self)
            self.labelText.setText(str(text))
        self.labelIndex = QLabel(self)
        self.labelIndex.setText('1/1')
        self.sliderIndex = QSlider(Qt.Horizontal, self)
        self.sliderIndex.setMaximum(1)
        self.sliderIndex.setMinimum(1)
        self.sliderIndex.setValue(1)
        hbox = QHBoxLayout(self)
        if text is not None:
            hbox.addWidget(self.labelText)
        hbox.addWidget(self.sliderIndex)
        hbox.addWidget(self.labelIndex)
        self.sliderIndex.valueChanged.connect(self.__slotIndexChanged)
        self.setMinimum = self.sliderIndex.setMinimum
        self.setValue = self.sliderIndex.setValue
        self.value = self.sliderIndex.value
        self.valueChanged = self.sliderIndex.valueChanged

    def setMaximum(self, *args, **kw):
        self.sliderIndex.setMaximum(*args, **kw)
        self.labelIndex.setText( \
                str(self.sliderIndex.value())+ \
                '/'+str(self.sliderIndex.maximum()))

    @pyqtSlot(int)
    def __slotIndexChanged(self, index):
        self.labelIndex.setText(str(index)+'/'+str(self.sliderIndex.maximum()))
        #self.sliderIndex.setSliderPosition(self.__visibleIndex + 1)


class volumeViewerWidget(QWidget):
    displayPercentile = 0.001
    opacityMax = 100
    indexMouseRightScale = 1/16
    def __init__(self, parent=None, colormap=None):
        super(volumeViewerWidget, self).__init__(parent)
        self.sliderOpacity = SliderWithTextWidget(self, text='Opacity')
        self.sliderIndex   = SliderWithTextWidget(self, text=' Slice ')
        self.viewerSlice =  volumeSliceViewerWidget(self, colormap=colormap)
        vbox = QVBoxLayout(self)
        vbox.addWidget(self.viewerSlice, stretch=1)
        vbox.addWidget(self.sliderIndex)
        vbox.addWidget(self.sliderOpacity)

        self.sliderOpacity.setMinimum(0)
        self.sliderOpacity.setMaximum(self.opacityMax)
        self.sliderOpacity.setValue(self.opacityMax)
        self.sliderOpacity.hide()

        self.__mouseRightPressing = False
        self.__mouseRightPos = QPoint(0,0)
        self.__mouseRightIndex = 0
        
        self.sliderIndex.valueChanged.connect(\
                lambda x: self.viewerSlice.setIndex(x-1))
        self.sliderOpacity.valueChanged.connect(\
                lambda x: self.viewerSlice.setOpacity(x/self.opacityMax))

    def setImage(self, image):
        imageArray = sitk.GetArrayFromImage(image)
        minVal = np.percentile(imageArray.ravel(), \
                self.displayPercentile*100)
        maxVal = np.percentile(imageArray.ravel(), \
                (1 - self.displayPercentile)*100)
        self.viewerSlice.setImage(image)
        self.viewerSlice.setWindow([minVal, maxVal])
        self.sliderIndex.setMinimum(1)
        self.sliderIndex.setMaximum(imageArray.shape[0])
        self.sliderOpacity.hide()

    def setLabel(self, label):
        labelArray = sitk.GetArrayFromImage(label)
        self.viewerSlice.setLabel(label)
        self.sliderOpacity.show()
   

    def mousePressEvent(self, event):
        if (event.button() == Qt.RightButton):
            self.__mouseRightPressing = True
            self.__mouseRightPressPos = event.pos()
            self.__mouseRightIndex = self.sliderIndex.value()
        else:
            super(volumeViewerWidget, self).mousePressEvent(event)
            
    def mouseMoveEvent(self, event):
        if self.__mouseRightPressing and (event.buttons() & Qt.RightButton):
            currentPos = event.pos()
            delta = currentPos - self.__mouseRightPressPos
            self.sliderIndex.setValue( \
                    round(delta.y()*self.indexMouseRightScale) + \
                    self.__mouseRightIndex)
        else:
            super(volumeViewerWidget, self).mouseMoveEvent(event)
    
    def mouseReleaseEvent(self, event):
        if (event.button() == Qt.RightButton):
            self.__mouseRightPressing = False
        else:
            super(volumeViewerWidget, self).mouseReleaseEvent(event)

if __name__ == '__main__':
    import sys, os, time
    from PyQt5.QtWidgets import QApplication
    from ReadSagittalPD import ReadSagittalPDs
    dicomPath='/mnt/repo/privateData/cartilage_origin/FromXuhua/PD/A102747075'
    #dicomPath=sys.argv[1]
    image = ReadSagittalPDs(dicomPath)[0]
    label = sitk.ReadImage('/mnt/workspace/cartilage/PDxuhua/predict/A102747075.nii.gz')
    flipped = sitk.Flip(label, (False, False, True), True)
    flipped = sitk.GetImageFromArray(\
            (sitk.GetArrayFromImage(label)[:,::-1,:]).astype(np.uint8))
    flipped.CopyInformation(label)
    app = QApplication(sys.argv)
    ex = volumeSliceViewerWidget()
    #ex = volumeViewerWidget(colormap = cv2.COLORMAP_SPRING)
    ex = volumeViewerWidget()
    #ex = SliderWithTextWidget(text="TEXT")
    ex.setGeometry(300, 300, 600, 600)
    ex.setWindowTitle('volumeViewerWidget')
    ex.show()
    time.sleep(1)
    ex.setImage(image)
    time.sleep(1)
    ex.setLabel(flipped)
    sys.exit(app.exec_())
