import sys
from PyQt5.QtWidgets import QWidget, QPushButton, QApplication, QLabel
from PyQt5.QtGui import QPalette, QBrush, QPixmap,QImage
import cv2
from volumeViewerWidget import scalableLabel


'''
    def __init__(self, parent=None, \
            image=None, label=None, \
            index=0, opacity=1, window=[None, None], \
            colormap=None, \
            labelColormap=None):

        super(imageViewer, self).__init__(parent)
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
'''
class imageViewer(scalableLabel):
    def __init__(self):
        super(imageViewer,self).__init__()

        # self.lb1 = QLabel(self)
        # pix = QPixmap('background.jpg')

        # self.lb1.setGeometry(0, 0, 300, 200)
        # self.lb1.setStyleSheet("border: 2px solid red")

        # 设置窗口的位置和大小
        self.setGeometry(300, 300, 600, 600)
        self.setWindowTitle('ROC Curve for classification')

    def set_image(self,array):
        img = QImage(array.copy().data, \
                array.shape[1], array.shape[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(img)
        self.setPixmap(pixmap)

        # pix = QPixmap(path)
        # self.lb1.setPixmap(pix)
        # 设置窗口的标题
        # self.setWindowTitle('Example')

        # 显示窗口
        # self.show()


if __name__ == '__main__':
    # 创建应用程序和对象
    app = QApplication(sys.argv)
    ex = imageViewer()
    arr = cv2.imread('TYPE.png')
    ex.set_image(arr)
    ex.show()
    sys.exit(app.exec_())
