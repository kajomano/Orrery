from socket import SO_BROADCAST
import numpy as np

from utils.common         import Resolution, Timer
from utils.linalg         import *

from raytracing.scenery   import Scenery
from raytracing.geometry  import Spheres
from raytracing.raytracer import Rays, Raytracer

from graphics.viewport    import Viewport
from graphics.gui         import GUI

# Notes ========================================================================
# To profile call:
# python -m cProfile -o orrery.prof orrery.py
# snakeviz ./orrery.prof

# Settings =====================================================================
# Resolution
res = Resolution(1440, 16/9)

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

viewport = Viewport(res, tracer)
# gui      = GUI(viewport, res, v = True)

# Calls ========================================================================
with Timer() as t:
    viewport.render()    
print(t)

# gui.start()

from PIL import Image
img = Image.fromarray(viewport.buffer, mode = 'RGB')
img = img.resize(tuple(res), Image.Resampling.BILINEAR)
img.show()
# img.save("rt_image_001.png")