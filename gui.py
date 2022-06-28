import time
import sys

from PyQt5.QtCore    import QSize, QTimer
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel
from PyQt5.QtGui     import QPixmap

from PIL.ImageQt     import ImageQt

class _MainWindow(QMainWindow):
    def __init__(self, res):
        super().__init__()

        self.lab = QLabel()

        self.setWindowTitle("Orrery")
        self.setFixedSize(QSize(res.h, res.v))
        self.setCentralWidget(self.lab)

class GUI():
    def __init__(self, viewport, res, r_rate = 60, v = False):
        self.viewport = viewport
        self.res      = res
        self.r_rate   = r_rate
        self.v        = v

        self.app = QApplication([])
        self.win = _MainWindow(res)

        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self._refresh)
        self.refresh_count = 0

        if v:
            self.report_timer = QTimer()
            self.report_timer.timeout.connect(self._report)

    def _refresh(self):        
        img = self.viewport.render(self.res)
        # NOTE: Has to be a copy!
        self.win.lab.setPixmap(QPixmap.fromImage(ImageQt(img).copy()))

        if self.v:
            self.refresh_count += 1

    def _report(self):
        print(f'GUI fps: {self.refresh_count}', end='\r')
        self.refresh_count = 0

    def start(self):
        self._refresh()
        self.refresh_timer.start(1000 // self.r_rate)

        if self.v:
           self.report_timer.start(1000)
        
        self.win.show()
        sys.exit(self.app.exec())
