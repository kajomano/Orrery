# NOTE: torch version > 1.12.0
import torch

from utils.consts           import ftype
from utils.common           import Resolution, Timer
from utils.torch            import DmModule

from raytracing.scene       import Object, Scene
import raytracing.geometry  as geom
import raytracing.materials as mat
from raytracing.rays        import Rays
from raytracing.tracer      import DiffuseTracer, PathTracer

from interfaces.viewport    import Viewport
from interfaces.gui         import GUI

from multiprocessing        import Process

# Notes ========================================================================
# To profile call:
# python -m cProfile -o orrery.prof orrery.py
# snakeviz ./orrery.prof

# Settings =====================================================================
# Resolution
res = Resolution(1440)
dev = 'cuda:0'

# Planets
class Ground(Object, geom.Sphere, mat.Metal):
    def __init__(self):
        super().__init__(
            center = torch.tensor([0, 0, -1000], dtype = ftype),
            radius = 1000 - 2,
            albedo = torch.tensor([1.0, 1.0, 1.0], dtype = ftype),
            fuzz   = 0.7
        )

class Sun(Object, geom.Sphere, mat.Shiny):
    def __init__(self):
        super().__init__(
            center = torch.tensor([-4, 0, 0], dtype = ftype),
            radius = 2,
            albedo = torch.tensor([1.0, 0.7, 0.0], dtype = ftype)
        )

class Earth(Object, geom.Sphere, mat.Diffuse):
    def __init__(self):
        super().__init__(
            center = torch.tensor([0, 0, 0], dtype = ftype),
            radius = 2,
            albedo = torch.tensor([0.2, 0.5, 0.8], dtype = ftype)
        )

class Moon(Object, geom.Sphere, mat.Metal):
    def __init__(self):
        super().__init__(
            center = torch.tensor([4, 0, 0], dtype = ftype),
            radius = 2,
            albedo = torch.tensor([0.3, 0.3, 0.3], dtype = ftype),
            fuzz   = 0.5
        )

# Instantiation ================================================================
scene  = Scene() + Ground() + Sun() + Earth() + Moon()

# tracer = DiffuseTracer(scene)
tracer = PathTracer(scene, samples = 100)

vport  = Viewport(res)

# gui    = GUI(vport, res)

# Move to GPU ==================================================================
scene.to(dev)
tracer.to(dev)

# Calls ========================================================================
with Timer() as t:
    tracer.render(vport)
print(t)

# if __name__ == '__main__':
#     p = Process(target = tracer.render, args = (vport,))
#     p.start()

#     gui.start()
#     p.join()

# from PIL import Image
# img = Image.fromarray(vport.getBuffer(), mode = 'RGB')
# img.show()
# img.save("rt_image_009.png")