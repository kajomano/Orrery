import numpy as np

from utils.common         import Resolution, Timer
from utils.linalg         import *

from raytracing.scene     import Scene
from raytracing.geometry  import Spheres
from raytracing.raytracer import Rays, Raytracer

from interfaces.viewport  import Viewport
from interfaces.gui       import GUI

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
scene  = Scene() + sun + earth
vport  = Viewport(res)
tracer = Raytracer(scene, vport)

# gui      = GUI(vport, res, v = True)

# Calls ========================================================================
with Timer() as t:
    tracer.render()    
print(t)

# gui.start()

from PIL import Image
img = Image.fromarray(vport.buffer, mode = 'RGB')
img = img.resize(tuple(res), Image.Resampling.BILINEAR)
img.show()
# img.save("rt_image_001.png")