import sys
from PyQt5.QtWidgets import QDialog, QApplication
from PyQt5.QtGui import QPainter
from PyQt5 import QtCore

class MyForm(QDialog):
    def __init__(self):
        super().__init__()
        self.setGeometry(30, 30, 500, 300)
        self.pos=[10,100]
        self.dragOffset = None
        #self.pos2=[20,20]
        self.show()

    def paintEvent(self,event):
        qp = QPainter()
        qp.begin(self)
        qp.drawLine(self.pos[0],0,self.pos[0],300)
        qp.drawLine(0, self.pos[1], 100, self.pos[1])
        qp.end()

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
        self.pos =
        self.data['pos'][ind][1] = ev.pos()[1] + self.dragOffset
        self.update()
        ev.accept()


    def mousePressEvent(self,event):
        if event.buttons() & QtCore.Qt.LeftButton:
            self.pos1[0], self.pos1[1] = event.pos().x(),event.pos().y()

    def mouseReleaseEvent(self,event):
        self.pos2[0], self.pos2[1] = event.pos().x(), event.pos().y()
        self.update()

if __name__=="__main__":
    app = QApplication(sys.argv)
    w = MyForm()
    w.show()
    sys.exit(app.exec_())
'''
class MyForm(QDialog):
    def __init__(self):
        super().__init__()
        self.setGeometry(30, 30, 500, 300)
        self.pos1=[10,10]
        self.pos2=[20,20]
        self.show()

    def paintEvent(self,event):
        qp = QPainter()
        qp.begin(self)
        qp.drawLine(self.pos1[0],self.pos1[1],self.pos2[0],self.pos2[1])
        qp.end()

    def mousePressEvent(self,event):
        if event.buttons() & QtCore.Qt.LeftButton:
            self.pos1[0], self.pos1[1] = event.pos().x(),event.pos().y()

    def mouseReleaseEvent(self,event):
        self.pos2[0], self.pos2[1] = event.pos().x(), event.pos().y()
        self.update()

if __name__=="__main__":
    app = QApplication(sys.argv)
    w = MyForm()
    w.show()
    sys.exit(app.exec_())
'''
