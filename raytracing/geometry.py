import numpy as np

from utils.linalg import norm, dot, cat, sqrt

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
        oc   = rays.orig - self.cent
        d_oc = dot(rays.dir, oc)
        oc_2 = dot(oc, oc)
        d_2  = dot(rays.dir, rays.dir)
        r_2  = self.rad * self.rad

        disc     = d_oc*d_oc - d_2*(oc_2 - r_2)
        hit_mask = disc >= 0

        # ...

        return(hit_mask)
        

