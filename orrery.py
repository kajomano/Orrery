# NOTE: torch version > 1.12.0
import torch

from utils.settings      import ftype
from utils.common        import Resolution, Timer

from raytracing.geometry import Spheres
from raytracing.scene    import Scene
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
dev = 'cuda:0'

# Planets
ball = Spheres(
    centers = torch.tensor([[0, 0, 0]], dtype = ftype),
    radii   = torch.tensor([2], dtype = ftype)
)

earth = Spheres(
    centers = torch.tensor([[0, 0, -1000]], dtype = ftype),
    radii   = torch.tensor([1000 - 2], dtype = ftype)
)

# Instantiation ================================================================
scene  = Scene() + earth + ball
vport  = Viewport(res)
tracer = DiffuseTracer(scene, vport)
gui    = GUI(vport, res, v = True)

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