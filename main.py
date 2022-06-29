from viewport import Viewport
from gui      import GUI
from common   import Resolution

# import cProfile

# Settings
res_render = Resolution(480, 4/3)
# res_gui    = Resolution(768, 4/3)
res_gui    = Resolution(480, 4/3)

# Instantiation
vp  = Viewport(res_render)
# gui = GUI(vp, res_gui, v = True)

# Calls
# gui.start()
# cProfile.run('gui.start()')

vp.render(res_render).show()