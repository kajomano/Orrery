import numpy as np

from utils.linalg import view

class Raytracer():
    def __init__(self, scenery, viewport):
        self.scen = scenery
        self.vp   = viewport

    def trace(self):
        hit_mask, rs = self.scen.spheres.intersect(self.vp.rays)
        hit_mask = view(hit_mask, (self.vp.res.v, self.vp.res.h, 2))
        
        self.vp.buffer[hit_mask[:, :, 0], :] = [255, 0, 0]
        self.vp.buffer[hit_mask[:, :, 1], :] = [0, 0, 255]
        