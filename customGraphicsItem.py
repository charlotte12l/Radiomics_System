"""
Demonstrate creation of a custom graphic (a candlestick plot)

"""


import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np

pg.setConfigOptions(imageAxisOrder='row-major')


## Create image to display
arr = np.ones((100, 100), dtype=float)
arr[45:55, 45:55] = 0
arr[25, :] = 5
arr[:, 25] = 5
arr[75, :] = 5
arr[:, 75] = 5
arr[50, :] = 10
arr[:, 50] = 10
arr += np.sin(np.linspace(0, 20, 100)).reshape(1, 100)
arr += np.random.normal(size=(100, 100))

# add an arrow for asymmetry
arr[10, :50] = 10
arr[9:12, 44:48] = 10
arr[8:13, 44:46] = 10

## create GUI
pg.mkQApp()
w = pg.GraphicsLayoutWidget()
w.setWindowTitle('pyqtgraph example: ROI Examples')

v1a = w.addPlot(row=0,col=0, lockAspect=True)
img1a = pg.ImageItem(arr)
v1a.addItem(img1a)
v1a.autoRange()
v1b = w.addPlot(row=1, col=0, lockAspect=True)
img1b = pg.ImageItem()## View ROI
v1b.addItem(img1b)
v1b.autoRange()

w.show()

roi= pg.PolyLineROI([[80, 60], [90, 30], [60, 40]], pen=(6, 9), closed=True,removable=True)
v1a.addItem(roi)

item = QtGui.QGraphicsPathItem()
item.setBrush(pg.mkBrush('r'))
v1a.addItem(item)

# path = pg.arrayToQPath
# path = pg.arrayToQPath(xdata.flatten(), ydata.flatten(), conn.flatten())
# item = QtGui.QGraphicsPathItem(path)
# getArrayRegion(data, img, axes=(0, 1), returnMappedCoords=True, **kwds)

def update(ROI):
    # axes = (0, 1)
    # x_data,y_data = [],[]
    # roi.getArrayRegion(self, data, img, axes=axes, fromBoundingRect=True, **kwds)
    result = ROI.getArrayRegion(arr, img1a, axes=(0, 1))
    # print(np.unique(result))
    img1b.setImage(result)
    # mapped = ROI.getLocalHandlePositions()
    # for i in range(len(mapped)):
    #     x_data.append(mapped[i][1].x())
    #     y_data.append(mapped[i][1].y())

    # result, mapped = ROI.getArrayRegion(arr, img1a, axes=(0, 1), returnCoords =True)
    # print('data:',x_data,y_data)
    # x_data=np.array(x_data)
    # y_data = np.array(y_data)
    # path = pg.arrayToQPath(x_data.flatten(), y_data.flatten())
    # item = QtGui.QGraphicsPathItem(path)
    # item.setBrush(pg.mkBrush('r'))
    # v1a.addItem(item)
    # v1b.addItem(item)
    # print(mapped[0][1])
    # print('sliced:', np.shape(sliced))
    #
    # mask = ROI.renderShapeMask(sliced.shape[axes[1]], sliced.shape[axes[0]])
    # mask = mask.T
    #
    # print('mask:',np.shape(mask))
    # # reshape mask to ensure it is applied to the correct data axes
    # shape = [1] * arr.ndim
    # shape[axes[0]] = sliced.shape[axes[0]]
    # shape[axes[1]] = sliced.shape[axes[1]]
    # mask = mask.reshape(shape)
    #
    # img1b.setImage(sliced * mask, levels=(0, arr.max()))
    v1b.autoRange()

def remove_ROI(evt):
    # evt.removeTimer.stop()
    v1a.removeItem(evt)

roi.sigRemoveRequested.connect(remove_ROI)
roi.sigRegionChanged.connect(update)

if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()

'''
def update(roi):
    img1b.setImage(roi.getArrayRegion(arr, img1a), levels=(0, arr.max()))
    v1b.autoRange()
    
roi.sigRegionChanged.connect(update)
## Create a subclass of GraphicsObject.
## The only required methods are paint() and boundingRect()
## (see QGraphicsItem documentation)
class CandlestickItem(pg.GraphicsObject):
    def __init__(self, data):
        pg.GraphicsObject.__init__(self)
        self.data = data  ## data must have fields: time, open, close, min, max
        self.generatePicture()

    def generatePicture(self):
        ## pre-computing a QPicture object allows paint() to run much more quickly,
        ## rather than re-drawing the shapes every time.
        self.picture = QtGui.QPicture()
        p = QtGui.QPainter(self.picture)
        p.setPen(pg.mkPen('w'))
        w = (self.data[1][0] - self.data[0][0]) / 3.
        for (t, open, close, min, max) in self.data:
            p.drawLine(QtCore.QPointF(t, min), QtCore.QPointF(t, max))
            if open > close:
                p.setBrush(pg.mkBrush('r'))
            else:
                p.setBrush(pg.mkBrush('g'))
            p.drawRect(QtCore.QRectF(t - w, open, w * 2, close - open))
        p.end()

    def paint(self, p, *args):
        p.drawPicture(0, 0, self.picture)

    def boundingRect(self):
        ## boundingRect _must_ indicate the entire area that will be drawn on
        ## or else we will get artifacts and possibly crashing.
        ## (in this case, QPicture does all the work of computing the bouning rect for us)
        return QtCore.QRectF(self.picture.boundingRect())


data = [  ## fields are (time, open, close, min, max).
    (1., 10, 13, 5, 15),
    (2., 13, 17, 9, 20),
    (3., 17, 14, 11, 23),
    (4., 14, 15, 5, 19),
    (5., 15, 9, 8, 22),
    (6., 9, 15, 8, 16),
]
item = CandlestickItem(data)
plt = pg.plot()
plt.addItem(item)
plt.setWindowTitle('pyqtgraph example: customGraphicsItem')

## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    import sys

    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
'''