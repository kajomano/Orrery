# NOTE: torch version > 1.12.0
import torch

from utils.settings      import ftype
from utils.common        import Resolution, Timer

from raytracing.geometry import Spheres
from raytracing.scene    import Scene
from raytracing.rays     import Rays
from raytracing.tracer   import DiffuseTracer

from interfaces.viewport import Viewport, ViewportParams
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
ball = Spheres(
    centers = torch.tensor([[0, 0, 0]], dtype = ftype),
    radii   = torch.tensor([0.9], dtype = ftype)
)

earth = Spheres(
    centers = torch.tensor([[0, 0, -1000]], dtype = ftype),
    radii   = torch.tensor([1000 - 0.9], dtype = ftype)
)

# Instantiation ================================================================
scene  = Scene() + earth + ball

vport_params = ViewportParams(
    eye_pos      = torch.tensor([0.0, -5.0, -0.3], dtype = ftype),
    view_target  = torch.tensor([0.0, 0.0, -0.3], dtype = ftype)
)
vport  = Viewport(res, vport_params)

tracer = DiffuseTracer(scene, vport)

gui    = GUI(vport, res, v = True)

# Move to GPU ==================================================================
scene.to(dev)
vport.to(dev)
tracer.to(dev)

# Calls ========================================================================
with Timer() as t:
    for _ in range(1):
        tracer.render()
print(t / 1)

# gui.start()

from PIL import Image
img = Image.fromarray(vport.buffer, mode = 'RGB')
img = img.resize(tuple(res), Image.Resampling.BILINEAR)
img.show()
# img.save("rt_image_002.png")