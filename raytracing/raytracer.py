import numpy as np

from utils.linalg import *

class Rays():
    def __init__(self, origins, directions, norm_dir = True):
        self.orig = origins
        self.dir  = norm(directions, axis = 1) if norm_dir else directions

    def __call__(self, ts):
        return(self.orig + ts * self.dir)

    def __getitem__(self, ids):
        return(Rays(
            origins    = self.orig[ids, :],
            directions = self.dir[ids, :],
            norm_dir   = False
        ))

class Raytracer():
    def __init__(self, scenery):
        self.scen = scenery

    def trace(self, rays):
        hits = self.scen.spheres.intersect(rays)

        results = np.zeros((rays.orig.shape[0], 3), dtype = np.uint8)

        # Test shading
        results[hits.hit_mask, :] = unsqueeze(hits.ts, 1)*150

        return(results)