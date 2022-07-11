# NOTE: torch version > 1.12.0
import torch

from utils.settings      import ftype
from utils.common        import Resolution, Timer

from raytracing.geometry import Sphere
from raytracing.scene    import Object, LightSource
from raytracing.rays     import Rays
from raytracing.tracer   import DiffuseTracer, PathTracer

from interfaces.viewport import Viewport
from interfaces.gui      import GUI

# Notes ========================================================================
# To profile call:
# python -m cProfile -o orrery.prof orrery.py
# snakeviz ./orrery.prof

# Settings =====================================================================
# Resolution
res = Resolution(360)
dev = 'cpu'

# Planets
class Sun(Object, Sphere):
    def __init__(self):
        super().__init__(
            center = torch.tensor([-10, 6, 0], dtype = ftype),
            radius = 6,
            albedo = torch.tensor([1.0, 0.7, 0.0], dtype = ftype),
            fuzz   = 0.1
        )

class Earth(Object, Sphere):
    def __init__(self):
        super().__init__(
            center = torch.tensor([0, 0, 0], dtype = ftype),
            radius = 2,
            albedo = torch.tensor([0.2, 0.5, 0.8], dtype = ftype),
            fuzz   = 1.0
        )

class Moon(Object, Sphere):
    def __init__(self):
        super().__init__(
            center = torch.tensor([-1, -1, -1], dtype = ftype),
            radius = 1,
            albedo = torch.tensor([0.3, 0.3, 0.3], dtype = ftype),
            fuzz   = 1.0
        )

class Ground(Object, Sphere):
    def __init__(self):
        super().__init__(
            center = torch.tensor([0, 0, -1000], dtype = ftype),
            radius = 1000 - 2,
            albedo = torch.tensor([0.4, 0.6, 0.5], dtype = ftype),
            fuzz   = 0.8
        )

# Instantiation ================================================================
scene  = Sun() + Earth() + Moon() + Ground()
vport  = Viewport(res)

# tracer = DiffuseTracer(scene)
tracer = PathTracer(scene)

# gui    = GUI(vport, res, v = True)

# Move to GPU ==================================================================
scene.to(dev)
vport.to(dev)
tracer.to(dev)

# Calls ========================================================================
with Timer() as t:
    tracer.render(vport)
print(t)

# with Timer() as t:
#     for _ in range(10):
#         tracer.render(vport)
# print(t / 10)

# gui.start()

from PIL import Image
img = Image.fromarray(vport.getBuffer(), mode = 'RGB')
img.show()
# img.save("rt_image_007.png")