import sys

from PyQt5.QtCore    import QSize, QTimer
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel
from PyQt5.QtGui     import QImage, QPixmap

from utils.common    import Timer

class GUI():
    def __init__(self, viewport, res, r_rate = 60, v = False):
        self.viewport = viewport
        self.res      = res
        self.r_rate   = r_rate
        self.v        = v

        self.app = QApplication([])
        self.win = _MainWindow(res)

        self.refresh_timer = QTimer()
        self.refresh_timer.setSingleShot(True)
        self.refresh_timer.timeout.connect(self._refreshLoop)
        self.refresh_count = 0
        self.refresh_bound = False

        if v:
            self.report_timer = QTimer()
            self.report_timer.timeout.connect(self._report)

    def _refreshLoop(self):
        with Timer() as t:
            img = self.viewport.buffer
            img = QImage(img.data, img.shape[1], img.shape[0], img.strides[0], QImage.Format_RGB888)

            pix = QPixmap.fromImage(img).scaled(self.res.h, self.res.v)
            self.win.lab.setPixmap(pix)

            if self.v:
                self.refresh_count += 1

        # Compensate for the elapsed time in rendering, schedule next refresh
        next_time = (1000 // self.r_rate) - int(t.elap * 1000)
        next_time = next_time if next_time > 0 else 0
        self.refresh_timer.start(next_time)

        self.refresh_bound = (next_time != 0)

    def _report(self):
        print(f'GUI fps: {self.refresh_count}, bound: {self.refresh_bound}', end='\r')
        self.refresh_count = 0

    def start(self):
        self._refreshLoop()

        if self.v:
           self.report_timer.start(1000)
        
        self.win.show()
        sys.exit(self.app.exec())

class _MainWindow(QMainWindow):
    def __init__(self, res):
        super().__init__()

        self.lab = QLabel()

        self.setWindowTitle("Orrery")
        self.setFixedSize(QSize(res.h, res.v))
        self.setCentralWidget(self.lab)