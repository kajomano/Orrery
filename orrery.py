# NOTE: torch version > 1.12.0
from telnetlib import DM
from unittest.case import doModuleCleanups
import torch

from utils.consts      import ftype
from utils.common        import Resolution, Timer
from utils.torch         import DmModule

from raytracing.scene    import Object
from raytracing.geometry import Sphere
from raytracing.rays     import Rays
from raytracing.tracer   import DiffuseTracer, PathTracer

from interfaces.viewport import Viewport
from interfaces.gui      import GUI

from multiprocessing     import Process

# Notes ========================================================================
# To profile call:
# python -m cProfile -o orrery.prof orrery.py
# snakeviz ./orrery.prof

# Settings =====================================================================
# Resolution
res = Resolution(1440)
dev = 'cpu'

# Planets
class Sun(Object, Sphere, DmModule):
    def __init__(self):
        super().__init__(
            center = torch.tensor([-10, 6, 0], dtype = ftype),
            radius = 6,
            albedo = torch.tensor([1.0, 0.7, 0.0], dtype = ftype),
            fuzz   = 0.0
        )

class Earth(Object, Sphere, DmModule):
    def __init__(self):
        super().__init__(
            center = torch.tensor([0, 0, 0], dtype = ftype),
            radius = 2,
            albedo = torch.tensor([0.2, 0.5, 0.8], dtype = ftype),
            fuzz   = 0.3
        )

class Moon(Object, Sphere, DmModule):
    def __init__(self):
        super().__init__(
            center = torch.tensor([-1, -1, -1], dtype = ftype),
            radius = 1,
            albedo = torch.tensor([0.3, 0.3, 0.3], dtype = ftype),
            fuzz   = 1.0
        )

class Ground(Object, Sphere, DmModule):
    def __init__(self):
        super().__init__(
            center = torch.tensor([0, 0, -1000], dtype = ftype),
            radius = 1000 - 2,
            albedo = torch.tensor([0.4, 0.6, 0.5], dtype = ftype),
            fuzz   = 1.0
        )

# Instantiation ================================================================
scene  = Sun() + Earth() + Moon() + Ground()

# tracer = DiffuseTracer(scene)
tracer = PathTracer(scene)

vport  = Viewport(res)

gui    = GUI(vport, res)

# Move to GPU ==================================================================
scene.to(dev)
tracer.to(dev)

# Calls ========================================================================
# with Timer() as t:
#     tracer.render(vport)
# print(t)

if __name__ == '__main__':
    p = Process(target = tracer.render, args = (vport,))
    p.start()

    gui.start()
    p.join()

# from PIL import Image
# img = Image.fromarray(vport.getBuffer(), mode = 'RGB')
# img.show()
# img.save("rt_image_007.png")