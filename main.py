from viewport import Viewport
from gui      import GUI
from common   import Resolution

# import cProfile

# Settings
res_render = Resolution(640, 480)
# res_gui    = Resolution(1024, 768)
res_gui    = Resolution(640, 480)

# Instantiation
vp  = Viewport(res_render)
gui = GUI(vp, res_gui, v = True)

# Calls
gui.start()
# cProfile.run('gui.start()')