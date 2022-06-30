import numpy as np

from utils.linalg import unsqueeze, norm, dot, cat, sqrt

class Rays():
    def __init__(self, origins, directions):
        self.orig = origins
        self.dir  = norm(directions, axis = 1)

class Spheres():
    def __init__(self, center = None, radius = None):
        self.cent = center
        self.rad  = radius

    def __add__(self, spheres):
        if not isinstance(spheres, Spheres):
            raise Exception("Non-matching geometrical types!")            

        if self.cent is None:
            self.cent = spheres.cent
        else:
            self.cent = cat(self.cent, spheres.cent)

        if self.rad is None:
            self.rad = spheres.rad
        else:
            self.rad = cat(self.rad, spheres.rad)

        return(self)

    def intersect(self, rays):
        # TODO: All this is broken, only works because there is only a single 
        # sphere!

        oc   = unsqueeze(rays.orig, 1) - unsqueeze(self.cent, 0)
        d_oc = dot(unsqueeze(rays.dir, 1), oc)
        oc_2 = dot(oc, oc)
        d_2  = dot(rays.dir, rays.dir)
        r_2  = self.rad * self.rad

        # Hit mask
        disc     = d_oc*d_oc - unsqueeze(d_2, 1)*(oc_2 - unsqueeze(r_2, 0))
        hit_mask = disc >= 0

        # # R
        # disc[~hit_mask] = 0
        # disc = 

        # rs_1 = -d_oc
        # rs_2 = -d_oc 
        # print(np.min(disc))

        return(hit_mask, -1)
