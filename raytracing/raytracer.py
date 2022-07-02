import numpy as np

from utils.linalg import *

# TODO: rename this file to Rays

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

class Raytracer():
    def __init__(self, scenery):
        self.scen = scenery

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