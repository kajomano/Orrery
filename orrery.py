import numpy as np

from utils.common         import Resolution
from utils.linalg         import *

from raytracing.scenery   import Scenery
from raytracing.geometry  import Spheres
from raytracing.raytracer import Rays, Raytracer

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
sun   = Spheres(
    centers = float([[0, 0, 0]]),
    radii   = float([2])
)

earth = Spheres(
    centers = float([[-1, -1, -1]]),
    radii   = float([1])
)

# Instantiation ================================================================
scenery  = Scenery() + sun + earth
tracer   = Raytracer(scenery)

viewport = Viewport(res_vp, tracer)
# gui      = GUI(viewport, res_gui, v = True)

# Calls ========================================================================
viewport.render()

# gui.start()

from PIL import Image
res_render = Resolution(960, 4/3)
img = Image.fromarray(viewport.buffer, mode = 'RGB')
img = img.resize(tuple(res_render), Image.Resampling.BILINEAR)        
img.show()