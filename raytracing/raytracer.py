import numpy as np

from utils.linalg import view

class Raytracer():
    def __init__(self, scenery, viewport, rays_per_pix = 5):
        self.scen = scenery
        self.vp   = viewport
        self.rpp  = rays_per_pix

    def trace(self):
        hit_mask = self.scen.spheres.intersect(self.vp.rays)
        hit_mask = view(hit_mask, (self.vp.res.v, self.vp.res.h, 1))
        
        self.vp.buffer = hit_mask.repeat(3, 2).astype(np.uint8) * 255
        