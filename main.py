from PyQt5 import QtWidgets, QtGui

from camera import Camera

h_res = 1024
v_res = 768

testcamera = Camera(h_res, v_res)

app = QtWidgets.QApplication([])
label = QtWidgets.QLabel()

# NOTE: Has to be a copy!
label.setPixmap(QtGui.QPixmap.fromImage(testcamera.render().copy()))

label.show()
app.exec()