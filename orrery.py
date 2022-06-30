import numpy as np

from utils.common         import Resolution

from raytracing.scenery   import Scenery
from raytracing.geometry  import Spheres
from raytracing.raytracer import Raytracer

from graphics.viewport    import Viewport
from graphics.gui         import GUI

# Notes ========================================================================
# To profile call:
# python -m cProfile -o orrery.prof orrery.py
# snakeviz .\orrery.prof

# Settings =====================================================================
# Resolution
res_vp  = Resolution(480, 4/3)
res_gui = Resolution(960, 4/3)

# Planets
sun = Spheres(
    center = np.zeros((1, 3), dtype = np.float32),
    radius = np.array([2], dtype = np.float32)
)

# Raytracing
rays_per_pix = 5

# Instantiation ================================================================
viewport = Viewport(res_vp)
scenery  = Scenery() + sun
tracer   = Raytracer(scenery, viewport, rays_per_pix)

# gui      = GUI(viewport, res_gui, v = True)

# Calls ========================================================================
tracer.trace()

# gui.start()

from PIL import Image
res_render = Resolution(960, 4/3)
img = Image.fromarray(viewport.buffer, mode = 'RGB')
img = img.resize(tuple(res_render), Image.Resampling.BILINEAR)        
img.show()