#!/usr/bin/python3
import sys
import time

import numpy as np

from matplotlib.backends.qt_compat import QtCore, QtWidgets, is_pyqt5
if is_pyqt5():
    from matplotlib.backends.backend_qt5agg import (
        FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
else:
    from matplotlib.backends.backend_qt4agg import (
        FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
from matplotlib.figure import Figure


class curveWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(curveWidget, self).__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)

        static_canvas = FigureCanvas(Figure(figsize=(5, 5)))
        layout.addWidget(static_canvas)
        #self.addToolBar(NavigationToolbar(static_canvas, self))

        #dynamic_canvas = FigureCanvas(Figure(figsize=(5, 3)))
        #layout.addWidget(dynamic_canvas)
        #self.addToolBar(QtCore.Qt.BottomToolBarArea,
        #                NavigationToolbar(dynamic_canvas, self))

        self._static_ax0, self._static_ax1, self._static_ax2 = \
                static_canvas.figure.subplots(3,1)
        #t = np.linspace(0, 10, 501)
        #self._static_ax.plot(t, np.tan(t), ".")

        #self._dynamic_ax = dynamic_canvas.figure.subplots()
        #self._timer = dynamic_canvas.new_timer(
        #    100, [(self._update_canvas, (), {})])
        #self._timer.start()

    #def _update_canvas(self):
    #    self._dynamic_ax.clear()
    #    t = np.linspace(0, 10, 101)
    #    # Shift the sinusoid as a function of time.
    #    self._dynamic_ax.plot(t, np.sin(t + time.time()))
    #    self._dynamic_ax.figure.canvas.draw()
    
    def plotCurve(self, x, fc, tc, pc):
        atlas = self.__thickness
        self._static_ax0.clear()
        self._static_ax1.clear()
        self._static_ax2.clear()
        self._static_ax0.plot(atlas[0], atlas[1], 'r', label='femur')
        self._static_ax1.plot(atlas[0], atlas[2], 'g', label='tibia')
        self._static_ax2.plot(atlas[0], atlas[3], 'b', label='patella')
        self._static_ax0.legend()
        self._static_ax1.legend()
        self._static_ax2.legend()
        self._static_ax0.plot(x, fc, 'ro')
        self._static_ax1.plot(x, tc, 'go')
        self._static_ax2.plot(x, pc, 'bo')
        self._static_ax0.set_xlabel('age (years old)')
        self._static_ax0.set_ylabel('average thickness')
        self._static_ax0.set_ylim([1.7,2.0])
        self._static_ax1.set_xlabel('age (years old)')
        self._static_ax1.set_ylabel('average thickness')
        self._static_ax1.set_ylim([1.1,1.6])
        self._static_ax2.set_xlabel('age (years old)')
        self._static_ax2.set_ylabel('average thickness')
        self._static_ax2.set_ylim([0.9,2.4])
        #self._static_ax.set_title('population cartilage thickness')
        self._static_ax0.figure.canvas.draw()
        self._static_ax1.figure.canvas.draw()
        self._static_ax2.figure.canvas.draw()

    def openCSV(self, path):
        x,fc,tc,pc = [],[],[],[]
        with open(path) as f:
            for row in f.readlines():
                col1, col2, col3, col4 = row.split(',')
                x.append(float(col1))
                fc.append(float(col2))
                tc.append(float(col3))
                pc.append(float(col4))
        self.__thickness = np.array([x, fc, tc, pc])

if __name__ == "__main__":
    qapp = QtWidgets.QApplication(sys.argv)
    app = curveWidget()
    app.openCSV('plot.csv')
    app.plotCurve(35, 1.6, 1.7, 1.8)
    print(app.sizeHint())
    app.show()
    qapp.exec_()
