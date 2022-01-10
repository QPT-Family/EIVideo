# Author: AP-Kai
# Datetime: 2022/1/7
# Copyright belongs to the author.
# Please indicate the source for reprinting.


import os
import sys
from QEIVideo.gui.Ui_MainWindow import Ui_MainWindow
from PyQt5.QtWidgets import *


class Demo(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(Demo, self).__init__()
        self.setupUi(self)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    demo = Demo()
    demo.show()
    sys.exit(app.exec())