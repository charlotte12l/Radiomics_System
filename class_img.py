import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np

class ImageWidget(QtGui.QWidget):
    def __init__(self, parent=None):
        super(ImageWidget, self).__init__(parent)

        # Interpret image data as row-major instead of col-major
        pg.setConfigOptions(imageAxisOrder='row-major')

        pg.mkQApp()
        self.win = pg.GraphicsLayoutWidget()
        self.win.setWindowTitle('pyqtgraph example: Image Analysis')

        # A plot1 area (ViewBox + axes) for displaying the image
        self.plot1 = self.win.addPlot()

        # Item for displaying image data
        self.item = pg.ImageItem()
        self.plot1.addItem(self.item)

        self.win.resize(800, 800)
        self.win.show()

        # Generate image self.data
        # self.data = np.random.normal(size=(200, 100))
        # self.data[20:80, 20:80] += 2.
        # self.data = pg.gaussianFilter(self.data, (3, 3))
        # self.data += np.random.normal(size=(200, 100)) * 0.1
        # self.item.setImage(self.data)

        # set position and scale of image
        # self.item.scale(0.2, 0.2)
        # self.item.translate(-50, 0)

        # zoom to fit imageo
        # self.plot1.setRange(xRange=[5,100],yRange=[5,200])
        # self.plot1.setYRange(yRange=[0,200])
        self.plot1.autoRange()

    def set_image(self,image):
        self.item.setImage(image)
        self.plot1.autoRange()


## Start Qt event loop unless running in interactive mode or using     pyside.
if __name__ == '__main__':

    app = QtGui.QApplication([])
    app.setStyle(QtGui.QStyleFactory.create("Cleanlooks"))

    image_widget = ImageWidget()
    data = np.random.normal(size=(200, 100))
    data[20:80, 20:80] += 2.
    data = pg.gaussianFilter(data, (3, 3))
    data += np.random.normal(size=(200, 100)) * 0.1
    image_widget.set_image(data)

    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore,     'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()