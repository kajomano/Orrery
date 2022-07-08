# NOTE: torch version > 1.12.0
import torch

from utils.settings      import ftype
from utils.common        import Resolution, Timer

from raytracing.geometry import Sphere
from raytracing.scene    import Object, LightSource
from raytracing.rays     import Rays
from raytracing.tracer   import DiffuseTracer

from interfaces.viewport import Viewport
from interfaces.gui      import GUI

# Notes ========================================================================
# To profile call:
# python -m cProfile -o orrery.prof orrery.py
# snakeviz ./orrery.prof

# Settings =====================================================================
# Resolution
res = Resolution(720)
dev = 'cpu'

# Planets
class Earth(Object, Sphere):
    def __init__(self):
        super().__init__(
            center = torch.tensor([0, 0, 0], dtype = ftype),
            radius = 2
        )

class Ball(Object, Sphere):
    def __init__(self):
        super().__init__(
            center = torch.tensor([0, 0, -1000], dtype = ftype),
            radius = 1000 - 2
        )

# Instantiation ================================================================
scene  = Earth() + Ball()
vport  = Viewport(res)
tracer = DiffuseTracer(scene, vport)
# gui    = GUI(vport, res, v = True)

# Move to GPU ==================================================================
scene.to(dev)
vport.to(dev)
tracer.to(dev)

# Calls ========================================================================
tracer.render()

# with Timer() as t:
#     for _ in range(10):
#         tracer.render()
# print(t / 10)

# gui.start()

from PIL import Image
img = Image.fromarray(vport.getBuffer(), mode = 'RGB')
img = img.resize(tuple(res), Image.Resampling.BILINEAR)
img.show()
# img.save("rt_image_002.png")