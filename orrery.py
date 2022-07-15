# NOTE: torch version > 1.12.0
import torch

from utils.consts           import ftype
from utils.common           import Resolution, Timer

from raytracing.scene       import Object
import raytracing.geometry  as geom
import raytracing.materials as mat
from raytracing.tracer      import SimpleTracer, PathTracer

from interfaces.viewport    import Viewport
from interfaces.gui         import GUI

from multiprocessing        import Process

# Notes ========================================================================
# To profile call:
# python -m cProfile -o orrery.prof orrery.py
# snakeviz ./orrery.prof

# Settings =====================================================================
# Resolution
res = Resolution(360)
dev = 'cpu'

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
            albedo = torch.tensor([0.9, 0.7, 0.0], dtype = ftype)
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
            fuzz   = 0.0
        )

class Minmus(Object, geom.Sphere, mat.Glowing):
    def __init__(self):
        super().__init__(
            center   = torch.tensor([1, -2, -1.5], dtype = ftype),
            radius   = 0.5,
            albedo   = torch.tensor([0.6, 1.0, 0.4], dtype = ftype),
            glow_min = 0.8,
            glow_max = 1.6
        )

# Instantiation ================================================================
scene  = Ground() + Sun() + Earth() + Moon() + Minmus()

# tracer = SimpleTracer(scene)
tracer = PathTracer(scene, samples = 10)

vport  = Viewport(res)

# gui    = GUI(vport, res)

# Move to GPU ==================================================================
scene.to(dev)
tracer.to(dev)

# Calls ========================================================================
if __name__ == '__main__':
    with Timer() as t:
        # tracer.render(vport)

        p = Process(target = tracer.render, args = (vport,))
        p.start()
        p.join()
    print(t)

    from PIL import Image
    img = Image.fromarray(vport.getBuffer(), mode = 'RGB')
    img.show()
    # img.save("rt_image_009.png")
