import numpy as np

from utils.common         import Resolution

from raytracing.scenery   import Scenery
from raytracing.geometry  import Spheres
from raytracing.raytracer import Raytracer

from graphics.viewport    import Viewport
from graphics.gui         import GUI

from utils.linalg         import dot

# Notes ========================================================================
# To profile call:
# python -m cProfile -o orrery.prof orrery.py
# snakeviz .\orrery.prof

# Settings =====================================================================
# Resolution
res_vp  = Resolution(480, 4/3)
res_gui = Resolution(960, 4/3)

# Planets
sun   = Spheres(
    center = np.array([[0, 0, 0]], dtype = np.float32),
    radius = np.array([2], dtype = np.float32)
)

earth = Spheres(
    center = np.array([[-1, -1, -1]], dtype = np.float32),
    radius = np.array([1], dtype = np.float32)
)

# Instantiation ================================================================
viewport = Viewport(res_vp)
scenery  = Scenery() + sun + earth
tracer   = Raytracer(scenery, viewport)

# gui      = GUI(viewport, res_gui, v = True)

# Calls ========================================================================
tracer.trace()

# gui.start()

from PIL import Image
res_render = Resolution(960, 4/3)
img = Image.fromarray(viewport.buffer, mode = 'RGB')
img = img.resize(tuple(res_render), Image.Resampling.BILINEAR)        
img.show()