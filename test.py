import SimpleITK as sitk
import numpy as np

a = sitk.ReadImage('.\\nn\\superResolution\\test_SRout.nii')
arr = sitk.GetArrayFromImage(a)
arr = arr[:,::-1,:]
sitk.WriteImage(sitk.GetImageFromArray(arr), 'test_out_trans.nii')
#
# img = sitk.ReadImage('B2_CESAG.dcm.nii')
# print(img.GetDirection())
# npa = sitk.GetArrayFromImage(img)
# print(np.shape(npa))
# img.SetDirection(a.GetDirection())
# npa1=sitk.GetArrayFromImage(img)
# print(np.shape(npa1))
# print(img.GetDirection())
# print(np.unique(npa))
#
# a = npa[15, :, :]
#
# print(np.shape(a))
# print(np.unique(a))

# resample = sitk.ResampleImageFilter()
# resample.SetOutputDirection(image.GetDirection())
# resample.SetOutputOrigin(image.GetOrigin())
# newspacing = [1, 1, 1]
# resample.SetOutputSpacing(newspacing)
# newimage = resample.Execute(image)
'''
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui

pg.setConfigOptions(antialias=True)

w = pg.GraphicsWindow()
w.setWindowTitle('Draggable')


class Graph(pg.GraphItem):
    def __init__(self):
        self.dragPoint = None
        self.dragOffset = None
        pg.GraphItem.__init__(self)

    def setData(self, **kwds):
        self.data = kwds
        if 'pos' in self.data:
            npts = self.data['pos'].shape[0]
            self.data['adj'] = np.column_stack((np.arange(0, npts-1), np.arange(1, npts)))
            self.data['data'] = np.empty(npts, dtype=[('index', int)])
            self.data['data']['index'] = np.arange(npts)
        self.updateGraph()

    def updateGraph(self):
        pg.GraphItem.setData(self, **self.data)

    def mouseDragEvent(self, ev):
        if ev.button() != QtCore.Qt.LeftButton:
            ev.ignore()
            return

        if ev.isStart():
            pos = ev.buttonDownPos()
            pts = self.scatter.pointsAt(pos)
            if len(pts) == 0:
                ev.ignore()
                return
            self.dragPoint = pts[0]
            ind = pts[0].data()[0]
            self.dragOffset = self.data['pos'][ind][1] - pos[1]
        elif ev.isFinish():
            self.dragPoint = None
            return
        else:
            if self.dragPoint is None:
                ev.ignore()
                return

        ind = self.dragPoint.data()[0]
        self.data['pos'][ind][1] = ev.pos()[1] + self.dragOffset
        self.updateGraph()
        ev.accept()


g = Graph()
v = w.addPlot()
v.addItem(g)

x = np.linspace(1, 100, 40)
pos = np.column_stack((x, np.sin(x)))

g.setData(pos=pos, size=10, pxMode=True)

if __name__ == '__main__':
    import sys

    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
'''