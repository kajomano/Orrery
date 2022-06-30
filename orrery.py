from viewport import Viewport
from gui      import GUI
from common   import Resolution

# Notes ========================================================================
# To profile call:
# python -m cProfile -o orrery.prof orrery.py
# snakeviz .\orrery.prof

# Notes ========================================================================
res_vp  = Resolution(480, 4/3)
res_gui = Resolution(1024, 4/3)
# res_gui    = Resolution(480, 4/3)

# Instantiation ================================================================
vp  = Viewport(res_vp)
gui = GUI(vp, res_gui, v = True)

# Calls ========================================================================
gui.start()

# from PIL      import Image
# res_render = Resolution(960, 4/3)
# img = Image.fromarray(vp.buffer, mode = 'RGB')
# img = img.resize(tuple(res_render), Image.Resampling.BILINEAR)        
# img.show()