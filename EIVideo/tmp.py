# require Python3.6+ & PyQt5 & OpenCV
# pip install PyQt5
# pip install python-opencv

import cv2
import numpy as np
import glob

import sys
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QLabel
from PyQt5.QtGui import QIcon, QPixmap


class WindowCapture(QtWidgets.QMainWindow):
    def __init__(self, files, fp_focus):
        super().__init__()
        self.start_pos = None
        self.stop_pos = None
        self.files = files
        self.file_id = 0
        self.fp_focus = fp_focus
        # setup the scale
        if len(self.files) > 0:
            im_ = cv2.imread(self.files[0], -1)
            self.im_w = im_.shape[1]
            self.im_h = im_.shape[0]
        self.initUI()
    def mousePressEvent(self, QMouseEvent):
        self.start_pos = QMouseEvent.pos()
    def mouseReleaseEvent(self, QMouseEvent):
        self.stop_pos = QMouseEvent.pos()
        d = (self.stop_pos.x() - self.start_pos.x()) \
            *(self.stop_pos.x() - self.start_pos.x()) + \
            (self.stop_pos.y() - self.start_pos.y())* \
            (self.stop_pos.y() - self.start_pos.y())
        d = np.sqrt(d)
        if d < 1: # not moved
            self.saveFocus()
            self.showNextImage()
    def keyPressEvent(self, event):
        if event.key() == ord('q') or event.key() == ord('Q'):
            self.close()
    def initUI(self):
        self.setGeometry(100, 100, 800, 600)
        self.setWindowTitle('Capture Focus')
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.FramelessWindowHint)

        # show the first image
        self.image_container = QLabel(self)
        self.im_ = QPixmap(self.files[self.file_id])
        self.im_ = self.im_.scaled(800, 600, QtCore.Qt.KeepAspectRatio)
        self.image_container.setPixmap(self.im_)
        self.image_container.resize(800, 600)
        self.show()
    def saveFocus(self):
        f_x = int(self.stop_pos.x() * self.im_w / 800.0)
        f_y = int(self.stop_pos.y() * self.im_h / 600.0)
        if self.file_id == 0:
            self.fp_focus.write('%d, %d' % (f_x, f_y))
        else:
            self.fp_focus.write('\n%d, %d' % (f_x, f_y))
    def showNextImage(self):
        if self.file_id == len(self.files)-1:
            self.close()
        else:
            self.file_id += 1
            self.im_ = QPixmap(self.files[self.file_id])
            self.im_ = self.im_.scaled(800, 600, QtCore.Qt.KeepAspectRatio)
            self.image_container.setPixmap(self.im_)

if __name__ == '__main__':
    root_dir = '/Users/liuchen21/Library/Mobile Documents/com~apple~CloudDocs/Documents/PycharmProjects/EIVideo/QEIVideo/result'
    focus_file = 'focus.txt'
    files = glob.glob(root_dir + '/*.png')
    files.sort()
    app = QtWidgets.QApplication(sys.argv)
    fp_focus = open(focus_file, 'wt')
    ex = WindowCapture(files, fp_focus)
    sys.exit(app.exec_())
    fp_focus.close()
