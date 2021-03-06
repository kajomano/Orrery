import sys

from PyQt5.QtCore    import QSize, QTimer
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel
from PyQt5.QtGui     import QImage, QPixmap

from utils.common    import Timer

class GUI():
    def __init__(self, viewport, res, r_rate = 60):
        self.vport  = viewport
        self.res    = res
        self.r_rate = r_rate

        self.app = QApplication([])
        self.win = _MainWindow(res)

        self.refresh_timer = QTimer()
        self.refresh_timer.setSingleShot(True)
        self.refresh_timer.timeout.connect(self._refreshLoop)

    def _refreshLoop(self):
        with Timer() as t:
            img = self.vport.getBuffer()
            img = QImage(img.data, img.shape[1], img.shape[0], img.strides[0], QImage.Format_RGB888)

            pix = QPixmap.fromImage(img).scaled(self.res.h, self.res.v)
            self.win.lab.setPixmap(pix)

        # Compensate for the elapsed time in rendering, schedule next refresh
        next_time = (1000 // self.r_rate) - int(t.elap * 1000)
        next_time = next_time if next_time > 0 else 0
        self.refresh_timer.start(next_time)

    def start(self):
        self._refreshLoop()
        
        self.win.show()
        sys.exit(self.app.exec())

class _MainWindow(QMainWindow):
    def __init__(self, res):
        super().__init__()

        self.lab = QLabel()

        self.setWindowTitle("Orrery")
        self.setFixedSize(QSize(res.h, res.v))
        self.setCentralWidget(self.lab)