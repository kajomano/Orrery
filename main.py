from viewport import Viewport
from gui      import GUI
from common   import Resolution

# Settings
res_render = Resolution(640, 480)
res_gui    = Resolution(1024, 768)  

# Instantiation
vp  = Viewport(res_render)
gui = GUI(vp, res_gui, v = True)

# Calls
gui.start()