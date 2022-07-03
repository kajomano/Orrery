import numpy as np

from utils.linalg import *

# TODO: move this to RayTracer, rename to RayHits()
class Intersects():
    def __init__(self, 
        hit_mask = np.full((0), False), 
        ts       = float_zero((0,)), 
        ps       = float_zero((0, 3)), 
        ns       = float_zero((0, 3))
    ):
        self.hit_mask = hit_mask
        self.ts       = ts
        self.ps       = ps
        self.ns       = ns

    def __add__(self, other):
        ts_comp = self.ts < other.ts
      
        return(Intersects(
            np.where(ts_comp, self.hit_mask, other.hit_mask),
            np.where(ts_comp, self.ts, other.ts),
            np.where(unsqueeze(ts_comp, 1), self.ps, other.ps),
            np.where(unsqueeze(ts_comp, 1), self.ns, other.ns)
        ))
       

class Spheres():
    def __init__(self, 
        centers = float_zero((0, 3)), 
        radii   = float_zero((0,))
    ):
        self.cent = centers
        self.rad  = radii

    def __len__(self):
        return(self.cent.shape[0])

    def __add__(self, other):
        self.cent = cat(self.cent, other.cent)
        self.rad = cat(self.rad, other.rad)

        return(self)

    def __getitem__(self, ids):
        return(Spheres(
            centers = self.cent[ids, :],
            radii   = self.rad[ids]
        ))

    def intersect(self, rays):
        if not len(self):
            return(Intersects()) 

        # TODO: replace self dots with length^2

        oc   = unsqueeze(rays.orig, 1) - unsqueeze(self.cent, 0)
        d_oc = dot(unsqueeze(rays.dir, 1), oc)
        oc_2 = dot(oc, oc)
        d_2  = unsqueeze(dot(rays.dir, rays.dir), 1)
        r_2  = unsqueeze(self.rad * self.rad, 0)
        disc = d_oc*d_oc - d_2*(oc_2 - r_2)        

        # Create hitmask of the smallest of the positive hits ==================
        hit_mask_valid = disc >= 0
        disc[~hit_mask_valid] = np.inf
        disc = sqrt(disc)

        # NOTE: As we only care about d_oc > 0 (the two vectors are facing the 
        # same way), this means that ((-d_oc + disc) / d_2) will always give the
        # away-facing intersection.
        ts_p = ((-d_oc + disc) / d_2)
        ts_m = ((-d_oc - disc) / d_2)
        ts   = cat(unsqueeze(ts_p, 2), unsqueeze(ts_m, 2), axis = 2)

        # Filter pure negative hits
        hit_mask_valid[np.all(ts < 0, axis = 2)] = False

        # TODO: add valid range for t: eps < t < inf

        # Find smallest positive hits
        ts[ts < 0] = np.inf
        ts = np.min(ts, axis = 2)

        hit_mask_smallest = np.full_like(hit_mask_valid, False)
        ts_smallest       = np.argmin(ts, axis = 1, keepdims = True)
        np.put_along_axis(hit_mask_smallest, ts_smallest, True, axis = 1)

        hit_mask_wide = hit_mask_valid & hit_mask_smallest
        hit_mask      = np.any(hit_mask_wide, axis = 1)

        if not np.any(hit_mask):
            return(Intersects())

        # Calculate intersect points
        ts_sub   = unsqueeze(ts[hit_mask_wide], 1)
        rays_sub = rays[hit_mask]

        ps = rays_sub(ts_sub)

        # Calculate surface normals
        hit_cents = self.cent[np.nonzero(hit_mask_wide)[-1], :]

        ns = norm(ps - hit_cents)

        # Get the results back to full size
        ps_full = float_zero((len(rays), 3))
        ps_full[hit_mask, :] = ps

        ns_full = float_zero((len(rays), 3))
        ns_full[hit_mask, :] = ns

        ts_full = squeeze(np.take_along_axis(ts, ts_smallest, axis = 1), 1)

        return(Intersects(hit_mask, ts_full, ps_full, ns_full))
