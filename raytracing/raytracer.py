import numpy as np

from utils.linalg import *

# TODO: rename this file to Rays
# TODO: allow for multidim rays
class Rays():
    def __init__(self, origins, directions, norm_dir = True):
        self.orig = origins
        self.dir  = norm(directions, axis = 1) if norm_dir else directions

    def __len__(self):
        return(self.orig.shape[0])

    def __call__(self, ts):
        return(self.orig + ts * self.dir)

    def __getitem__(self, ids):
        return(Rays(
            origins    = self.orig[ids, :],
            directions = self.dir[ids, :],
            norm_dir   = False
        ))

class RayHits():
    def __init__(self, 
        hit_mask = np.full((0), False), 
        ts       = float_zero((0,)), 
        ps       = float_zero((0, 3)), 
        ns       = float_zero((0, 3))
    ):
        # TODO: put ts, ps and ns into a single tensor
        self.hit_mask = hit_mask
        self.ts       = ts
        self.ps       = ps
        self.ns       = ns

    def __add__(self, other):
        ts_comp = self.ts < other.ts
      
        return(RayHits(
            np.where(ts_comp, self.hit_mask, other.hit_mask),
            np.where(ts_comp, self.ts, other.ts),
            np.where(unsqueeze(ts_comp, 1), self.ps, other.ps),
            np.where(unsqueeze(ts_comp, 1), self.ns, other.ns)
        ))

class Raytracer():
    def __init__(self, scenery, viewport):
        self.scen  = scenery
        self.vport = viewport

    def trace(self, rays):
        hits_1 = self.scen.spheres[0:1].intersect(rays)
        hits_2 = self.scen.spheres[1:2].intersect(rays)

        hits = hits_1 + hits_2

        # hits  = self.scen.spheres.intersect(rays)
        final = np.zeros((len(rays), 3), dtype = np.uint8)

        # Test shading
        final[hits.hit_mask, :] = unsqueeze(hits.ts[hits.hit_mask], 1)*150
        # final[hits_1.hit_mask, :] = [[255, 0, 0]]
        # final[hits_2.hit_mask, :] = [[0, 0, 255]]

        return(final)

    def render(self):
        ray_hits = self.trace(self.vport.rays)
        self.vport.buffer = view(ray_hits, (self.vport.res.v, self.vport.res.h, 3))