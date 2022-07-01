import numpy as np

from utils.linalg import *

# TODO: define & or + operation to collate Intersects
class Intersects():
    def __init__(self, hit_mask, ts, ps, ns):
        self.hit_mask = hit_mask
        self.ts       = ts
        self.ps       = ps
        self.ns       = ns

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
        oc   = unsqueeze(rays.orig, 1) - unsqueeze(self.cent, 0)
        d_oc = dot(unsqueeze(rays.dir, 1), oc)
        oc_2 = dot(oc, oc)
        d_2  = unsqueeze(dot(rays.dir, rays.dir), 1)
        r_2  = unsqueeze(self.rad * self.rad, 0)
        disc = d_oc*d_oc - d_2*(oc_2 - r_2)        

        # Create hitmask of the smallest of the positive hits
        hit_mask_1 = disc >= 0
        disc[~hit_mask_1] = np.inf
        disc = sqrt(disc)

        # NOTE: As we only care about d_oc > 0 (the two vectors are facing the 
        # same way), this means that ((-d_oc + disc) / d_2) will always give the
        # away-facing intersection.
        ts_p = ((-d_oc + disc) / d_2)
        ts_m = ((-d_oc - disc) / d_2)
        ts   = cat(unsqueeze(ts_p, 2), unsqueeze(ts_m, 2), axis = 2)

        ts[ts < 0] = np.inf
        ts = np.min(ts, axis = 2)

        hit_mask_2 = np.full_like(hit_mask_1, False)
        np.put_along_axis(hit_mask_2, np.argmin(ts, axis = 1, keepdims = True), True, axis = 1)

        hit_mask = hit_mask_1 & hit_mask_2

        # Calculate intersect points
        ts_sub   = unsqueeze(ts[hit_mask], 1)
        rays_sub = rays[np.any(hit_mask, axis = 1)]

        ps = rays_sub(ts_sub)

        # Calculate surface normals
        hit_ids = np.nonzero(hit_mask)[1]
        hit_cents = self.cent[hit_ids, :]

        ns = norm(ps - hit_cents)

        ts = ts[hit_mask]
        hit_mask = np.any(hit_mask, axis = 1)

        return(Intersects(hit_mask, ts, ps, ns))
