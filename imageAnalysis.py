# -*- coding: utf-8 -*-
"""
Demonstrates common image analysis tools.

Many of the features demonstrated here are already provided by the ImageView
widget, but here we present a lower-level approach that provides finer control
over the user interface.
"""

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np
import SimpleITK as sitk

# Interpret image data as row-major instead of col-major
pg.setConfigOptions(imageAxisOrder='row-major')

pg.mkQApp()
win = pg.GraphicsLayoutWidget()
win.setWindowTitle('pyqtgraph example: Image Analysis')

# A plot area (ViewBox + axes) for displaying the image
p_a = win.addPlot(row=0,col=0)
p_a.hideAxis('bottom')
p_a.hideAxis('left')
# p_a.enableAutoScale(False)

p_s = win.addPlot(row=0,col=1)
p_s.hideAxis('bottom')
p_s.hideAxis('left')

p_c = win.addPlot(row=1,col=1)
p_c.hideAxis('bottom')
p_c.hideAxis('left')
#p1 = win.addPlot(row=2,col=0)

# Item for displaying image data
img_s = pg.ImageItem()
img_a= pg.ImageItem()
img_c= pg.ImageItem()
p_s.addItem(img_s)
p_a.addItem(img_a)
p_c.addItem(img_c)

win.resize(800, 800)
win.show()


npa = sitk.GetArrayFromImage(sitk.ReadImage('B2_CESAG.dcm.nii'))
z,y,x = np.shape(npa)
print(np.shape(npa))# (32, 310, 352) z y x
cur_z = z//2
cur_x = x//2
cur_y = y//2

sagittal = np.flipud(npa[z//2,:,:])
axial = np.fliplr(np.rot90(npa[:,y//2,:],1))
coronal = np.fliplr(np.rot90(npa[:,:,x//2],1))
# axial = np.flipud(npa[:,y//2,:])
# coronal = np.flipud(npa[:,:,x//2])

# Generate image data
# data = np.random.normal(size=(200, 100))
# data[20:80, 20:80] += 2.
# data = pg.gaussianFilter(data, (3, 3))
# data += np.random.normal(size=(200, 100)) * 0.1

img_s.setImage(sagittal)
img_a.setImage(axial)
img_c.setImage(coronal)

# img_a.scale(1, 4)
# img_c.scale(1, 0.5)
# set position and scale of image
# img.scale(0.2, 0.2)
# img.translate(-50, 0)

label = pg.LabelItem(justify='right')
win.addItem(label)

vLine_a = pg.InfiniteLine(angle=90, movable=False)
hLine_a = pg.InfiniteLine(angle=0, movable=False)
p_a.addItem(vLine_a, ignoreBounds=True)
p_a.addItem(hLine_a, ignoreBounds=True)

vLine_s = pg.InfiniteLine(angle=90, movable=False)
hLine_s = pg.InfiniteLine(angle=0, movable=False)
p_s.addItem(vLine_s, ignoreBounds=True)
p_s.addItem(hLine_s, ignoreBounds=True)

vLine_c = pg.InfiniteLine(angle=90, movable=False)
hLine_c = pg.InfiniteLine(angle=0, movable=False)
p_c.addItem(vLine_c, ignoreBounds=True)
p_c.addItem(hLine_c, ignoreBounds=True)

'''
def mouseMoved(evt):
    pos = evt[0]  ## using signal proxy turns original arguments into a tuple
    # print('move:',pos)
    # if p1.sceneBoundingRect().contains(pos):
    #     mousePoint = p1.vb.mapSceneToView(pos)
    #     vLine1.setPos(mousePoint.x())
    #     hLine1.setPos(mousePoint.y())
    if p2.sceneBoundingRect().contains(pos):
        mousePoint = p2.vb.mapSceneToView(pos)
        vLine2.setPos(mousePoint.x())
        hLine2.setPos(mousePoint.y())
'''


def mouseCliked(ev):
    # print(ev)
    global cur_x,cur_y,cur_z,sagittal,coronal,axial
    pos = ev[0].scenePos()
    # print('clicked:',pos)
    # itemBoundingRect
    # if p_a.sceneBoundingRect().contains(pos):
    if p_a.sceneBoundingRect().contains(pos):
        mousePoint = p_a.vb.mapSceneToView(pos)
        print(mousePoint)
        if 0<=mousePoint.x()<32 and 0<=mousePoint.y()<352:
            vLine_a.setPos(mousePoint.x())
            hLine_a.setPos(mousePoint.y())

            cur_z =  int(z-mousePoint.x()+0.5)
            cur_x = int(x-mousePoint.y()+0.5)

            sagittal = np.flipud(npa[cur_z, :, :])
            coronal = np.fliplr(np.rot90(npa[:, :, cur_x], 1))
            img_s.setImage(sagittal)
            img_c.setImage(coronal)

            vLine_s.setPos(cur_x)
            hLine_s.setPos(y-cur_y)
            vLine_c.setPos(z-cur_z)
            hLine_c.setPos(y-cur_y)


            label.setText(
                "<span style='font-size: 12pt'>x=%d,   <span style='font-size: 12pt'>y=%d</span>, <span style='font-size: 12pt'>z=%d " % (
                    cur_x, cur_y,cur_z ))
        # ev.accept()
    if p_s.sceneBoundingRect().contains(pos):
        mousePoint = p_s.vb.mapSceneToView(pos)
        vLine_s.setPos(mousePoint.x())
        hLine_s.setPos(mousePoint.y())

        cur_x= int(mousePoint.x()+0.5)
        cur_y = int(y - mousePoint.y() + 0.5)

        axial = np.fliplr(np.rot90(npa[:, cur_y, :], 1))
        coronal = np.fliplr(np.rot90(npa[:, :, cur_x], 1))
        img_a.setImage(axial)
        img_c.setImage(coronal)

        vLine_a.setPos(z-cur_z)
        hLine_a.setPos(x-cur_x)
        vLine_c.setPos(z-cur_z)
        hLine_c.setPos(y-cur_y)

        label.setText(
            "<span style='font-size: 12pt'>x=%d,   <span style='font-size: 12pt'>y=%d</span>, <span style='font-size: 12pt'>z=%d " % (
                cur_x,cur_y,cur_z ))
    #
    if p_c.sceneBoundingRect().contains(pos):
        mousePoint = p_c.vb.mapSceneToView(pos)
        vLine_c.setPos(mousePoint.x())
        hLine_c.setPos(mousePoint.y())

        cur_z =  int(z-mousePoint.x()+0.5)
        cur_y = int(y - mousePoint.y() + 0.5)

        axial = np.fliplr(np.rot90(npa[:, cur_y, :], 1))
        sagittal = np.flipud(npa[cur_z, :, :])
        img_a.setImage(axial)
        img_s.setImage(sagittal)

        vLine_a.setPos(z-cur_z)
        hLine_a.setPos(x-cur_x)
        vLine_s.setPos(cur_x)
        hLine_s.setPos(y-cur_y)

        label.setText(
            "<span style='font-size: 12pt'>x=%d,   <span style='font-size: 12pt'>y=%d</span>, <span style='font-size: 12pt'>z=%d " % (
                cur_x,cur_y,cur_z ))

proxy_a = pg.SignalProxy(p_a.scene().sigMouseClicked, rateLimit=60, slot=mouseCliked)
# zoom to fit imageo
# p_a.autoRange()
p_a.disableAutoRange()
p_a.setAspectLocked(lock=True, ratio=4)

proxy_c = pg.SignalProxy(p_c.scene().sigMouseClicked, rateLimit=60, slot=mouseCliked)
# zoom to fit imageo
# p_c.autoRange()
p_c.disableAutoRange()

proxy_s = pg.SignalProxy(p_s.scene().sigMouseClicked, rateLimit=60, slot=mouseCliked)
# zoom to fit imageo
p_s.autoRange()

## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()