import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui

#random data with random shape
data = np.random.random((1000,1000))

#get data from an image instead, uncommment this
#from PIL import Image
#inputImage='duck.jpg'
#pilImage=Image.open(inputImage)
#data = np.asarray(pilImage)
#data = np.sum(data,axis=2)

#mask same shape as data
mask = np.zeros(data.shape)
#prepare points for dummy ROI
points = np.array([[10,10],[400,10],[400,600],[10,600],[10,10]])

#quick setup of image plotting
pg.mkQApp()
w = pg.GraphicsLayoutWidget()
# imgold = pg.image(data)
# img = pg.image(data)
v1a = w.addPlot(row=0,col=0, lockAspect=True)
img1a = pg.ImageItem(data)
v1a.addItem(img1a)
v1a.autoRange()

v1b = w.addPlot(row=1, col=0, lockAspect=True)
img1b = pg.ImageItem()## View ROI
v1b.addItem(img1b)
v1b.autoRange()

w.show()

#make dummy ROI from points
roi = pg.PolyLineROI(points,closed=True)
#add ROI
v1a.addItem(roi)
# img.addItem(roi)
# imgold.addItem(roi)
# imgview = img.getImageItem()

#Now do the selection
cols,rows = data.shape
m = np.mgrid[:cols,:rows]
possx = m[0,:,:]# make the x pos array
possy = m[1,:,:]# make the y pos array
possx.shape = cols,rows
possy.shape = cols,rows


def update(roi):
    mpossx = roi.getArrayRegion(possx, img1a).astype(int)
    mpossx = mpossx[np.nonzero(mpossx)]  # get the x pos from ROI
    mpossy = roi.getArrayRegion(possy, img1a).astype(int)
    mpossy = mpossy[np.nonzero(mpossy)]  # get the y pos from ROI
    mask[mpossx, mpossy] = data[mpossx, mpossy]

    # uncomment these lines to see the result. Comment them to see
    # before the result
    # mask=(mask>0).astype(int)
    a = (mask>0).astype(int)
    img1b.setImage(a)
    v1b.autoRange()
    print(np.shape(mask),np.unique(mask))

roi.sigRegionChanged.connect(update)

#for more questions etc, use the help function
#ex, help(img) will tell you all about member functions and vars
# of the class associated to img
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()