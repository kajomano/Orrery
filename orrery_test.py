# NOTE: torch version > 1.12.0
import torch

from utils.torch            import ftype
from utils.common           import Resolution, Timer

import raytracing.geometry  as geom
import raytracing.materials as mat
from raytracing.tracer      import SimpleTracer

from interfaces.viewport    import Viewport

# Settings =====================================================================
res = Resolution(720)
dev = 'cpu'

# Scene ========================================================================
class Ground(geom.Sphere, mat.Metal):
    def __init__(self):
        super().__init__(
            center = torch.tensor([0, 0, -1000], dtype = ftype),
            radius = 1000,
            albedo = torch.tensor([0.35, 0.78, 0.52], dtype = ftype),
            fuzz   = 0.7
        )

class Sun(geom.Sphere, mat.Shiny):
    def __init__(self):
        super().__init__(
            center = torch.tensor([-4.5, 0, 2], dtype = ftype),
            radius = 2,
            albedo = torch.tensor([0.9, 0.7, 0.0], dtype = ftype)
        )

class Earth(geom.Sphere, mat.Glass):
    def __init__(self):
        super().__init__(
            center = torch.tensor([0, 0, 2], dtype = ftype),
            radius = 2,
            albedo = torch.tensor([0.2, 0.5, 0.8], dtype = ftype),
            eta    = 1.5
        )

scene = Ground() + Sun() + Earth()
scene.buildHierarchy()

# Instantiation ================================================================
tracer = SimpleTracer(scene)
vport  = Viewport(res)

# Move to GPU ==================================================================
scene.to(dev)
tracer.to(dev)
vport.to(dev)

# Calls ========================================================================
with Timer() as t:
    tracer.render(vport)
print(t)

# from PIL import Image
# img = Image.fromarray(vport.getBuffer().numpy(), mode = 'RGB')
# img.show()
