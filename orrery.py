# NOTE: torch version > 1.12.0
import torch

from utils.torch            import ftype
from utils.common           import Resolution, Timer
from utils.rand             import randInCircle

from raytracing.scene       import Object
import raytracing.geometry  as geom
import raytracing.materials as mat
from raytracing.tracer      import SimpleTracer, PathTracer

from interfaces.viewport    import Viewport
from interfaces.gui         import GUI

import multiprocessing      as mp

if __name__ == '__main__':
    mp.set_start_method('spawn')

# Notes ========================================================================
# To profile call:
# python -m cProfile -o orrery.prof orrery.py
# snakeviz ./orrery.prof

# Settings =====================================================================
res = Resolution(1440)
dev = 'cpu'

# Scene ========================================================================
class Ground(Object, geom.Sphere, mat.Metal):
    def __init__(self):
        super().__init__(
            center = torch.tensor([0, 0, -1000], dtype = ftype),
            radius = 1000,
            albedo = torch.tensor([0.35, 0.78, 0.52], dtype = ftype),
            fuzz   = 0.7
        )

class Sun(Object, geom.Sphere, mat.Shiny):
    def __init__(self):
        super().__init__(
            center = torch.tensor([-4.5, 0, 2], dtype = ftype),
            radius = 2,
            albedo = torch.tensor([0.9, 0.7, 0.0], dtype = ftype)
        )

class Earth(Object, geom.Sphere, mat.Glass):
    def __init__(self):
        super().__init__(
            center = torch.tensor([0, 0, 2], dtype = ftype),
            radius = 2,
            albedo = torch.tensor([0.2, 0.5, 0.8], dtype = ftype),
            eta    = 1.5
        )

class Moon(Object, geom.Sphere, mat.Metal):
    def __init__(self):
        super().__init__(
            center = torch.tensor([4.5, 0, 2], dtype = ftype),
            radius = 2,
            albedo = torch.tensor([0.3, 0.3, 0.3], dtype = ftype),
            fuzz   = 0.2
        )

class RandDiffuse(Object, geom.Sphere, mat.Diffuse):
    def __init__(self, center):
        super().__init__(
            center   = center,
            radius   = 0.5,
            albedo   = torch.rand([3], dtype = ftype)
        )

class RandShiny(Object, geom.Sphere, mat.Shiny):
    def __init__(self, center):
        super().__init__(
            center   = center,
            radius   = 0.5,
            albedo   = torch.rand([3], dtype = ftype)
        )

class RandGlowing(Object, geom.Sphere, mat.Glowing):
    def __init__(self, center):
        super().__init__(
            center   = center,
            radius   = 0.5,
            albedo   = torch.rand([3], dtype = ftype),
            glow_min = 1.2,
            glow_max = 3.0
        )

class RandGlass(Object, geom.Sphere, mat.Glass):
    def __init__(self, center):
        super().__init__(
            center   = center,
            radius   = 0.5,
            albedo   = torch.rand([3], dtype = ftype),
            eta      = 1.5
        )

scene = Sun() + Earth() + Moon()

def testObj(candidate):
    for obj in scene.obj_list:
        if torch.norm(obj.cent[:2] - candidate.cent[:2]) < (obj.rad + candidate.rad):
            return(False)
    return(True)

for i in range(20):
    for type in [RandDiffuse, RandShiny, RandGlowing, RandGlass]:
        while True:
            rand_loc = torch.tensor(list(randInCircle(1, dev)) + [0.0]) * 10
            rand_loc[2] = 0.5 

            candidate = type(rand_loc)
            
            if testObj(candidate):
                scene += candidate
                break

scene += Ground()

# Instantiation ================================================================
# tracer = SimpleTracer(scene)
tracer = PathTracer(scene, samples = 100)

# Move to GPU ==================================================================
scene.to(dev)
tracer.to(dev)    

# Calls ========================================================================
if __name__ == '__main__':
    vport = Viewport(res)
    vport.to(dev)

    with Timer() as t:
        tracer.render(vport)

        # p = mp.Process(target = tracer.render, args = (vport,))
        # p.start()
        # p.join()
    print(t)

    from PIL import Image
    img = Image.fromarray(vport.getBuffer().numpy(), mode = 'RGB')
    img.show()
    img.save("rt_image_013.png")
