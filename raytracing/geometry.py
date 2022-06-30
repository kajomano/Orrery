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
        d_2  = unsqueeze(dot(rays.dir, rays.dir), 1)
        r_2  = unsqueeze(self.rad * self.rad, 0)

        # Hit mask
        disc     = d_oc*d_oc - d_2*(oc_2 - r_2)
        hit_mask = disc >= 0

        # Choose the smallest of the positive hits
        disc[~hit_mask] = np.nan
        disc = sqrt(disc)

        # TODO: We should only care about d_oc > 0 (the two vectors are facing 
        # the same way), but does this allow for the simplification of not
        # caring about the (-d_oc + disc) / d_2 possible solution?
        rs_p = ((-d_oc + disc) / d_2)
        rs_m = ((-d_oc - disc) / d_2)
        rs   = cat(unsqueeze(rs_p, 2), unsqueeze(rs_m, 2), axis = 2)
        rs[rs < 0] = np.nan
        rs   = np.min(rs, axis = 2)

        # WHAT
        # TODO: remove this, this is ugly
        hit_mask = np.min(rs, axis = 1, keepdims = True) == rs

        print(hit_mask.shape)
        # print(np.sum(np.argmin(rs, axis = 1)))

        # TODO: got stuck at this scatter op

        # hit_mask[:] = False
        # hit_mask = np.put_along_axis(hit_mask, np.argmin(rs, axis = 1, keepdims = True), True, axis = 1)

        # rs[rs < 0] = np.nan


        

        # Choose the smallest of the positive hits
        # hit_mask[rs < 0] = False
        # rs[~hit_mask]    = np.inf
        # print(np.argmin(rs, axis = 1))

        # TODO: return rs?, hit points and normals

        return(hit_mask, -1)
