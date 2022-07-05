import torch
from torch.nn.functional import normalize

from utils.settings  import ftype, t_min
from utils.torch     import DmModule
from raytracing.rays import RayHits

class Spheres(DmModule):
    def __init__(self, 
        centers = torch.zeros((0, 3), dtype = ftype), 
        radii   = torch.zeros((0,), dtype = ftype)
    ):
        if len(centers.shape) != 2 or \
        len(radii.shape) != 1 or \
        centers.shape[0] != radii.shape[0] or \
        centers.shape[1] != 3 or \
        centers.dtype != ftype or \
        radii.dtype != ftype:
            raise Exception("Invalid sphere parameters!")
            
        self.cent = centers
        self.rad  = radii

        super().__init__()

    def __len__(self):
        return(self.cent.shape[0])

    def __add__(self, other):
        self.cent = torch.cat((self.cent, other.cent), dim = 0)
        self.rad  = torch.cat((self.rad, other.rad), dim = 0)

        return(self)

    def __getitem__(self, ids):
        return(Spheres(
            centers = self.cent[ids, :],
            radii   = self.rad[ids]
        ))

    def intersect(self, rays):
        if not len(self):
            return(RayHits(rays))

        oc   = self.cent.unsqueeze(0) - rays.orig.unsqueeze(1) 
        # NOTE: dot product on the last axis
        d_oc = torch.einsum('ijk,ijk->ij', rays.dir.unsqueeze(1), oc)
        oc_2 = torch.sum(torch.pow(oc, 2), dim = 2)
        r_2  = torch.pow(self.rad, 2).unsqueeze(0)
        disc = torch.pow(d_oc, 2) - oc_2 + r_2

        obj_hit_mask = disc >= 0

        ts_p = d_oc + torch.sqrt(disc.clamp(0))
        ts_n = d_oc - torch.sqrt(disc.clamp(0))
        obj_ts = torch.where((ts_n < ts_p) & (ts_n > t_min), ts_n, ts_p)

        # Filter for hits behind the origin (purely negative)
        obj_hit_mask[obj_ts < t_min] = False   

        if not torch.any(obj_hit_mask):
            return(RayHits(rays))

        obj_ts[~obj_hit_mask] = torch.inf

        # Start assembling the hits struct
        hits = RayHits(rays, mask = torch.any(obj_hit_mask, dim = 1))

        # Calculate object hit ids
        ts_hit, hit_ids = torch.min(obj_ts[hits.mask, :], dim = 1)

        # Fill out rest of the hit details
        hits.ts[hits.mask]           = ts_hit
        hits.details[hits.mask, :3]  = rays[hits.mask](ts_hit)
        hits.details[hits.mask, 3:6] = normalize(hits.details[hits.mask, :3] - self.cent[hit_ids, :], dim = 1)

        return(hits)
