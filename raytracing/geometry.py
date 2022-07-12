import torch
from torch.nn.functional import normalize

from utils.settings   import ftype, t_min

class Sphere:
    def __init__(self, center, radius, **kwargs):
        if center.shape != torch.Size([3]) or \
        center.dtype != ftype or \
        not isinstance(radius, (int, float)):
            raise Exception("Invalid sphere parameters!")
            
        self.cent = center.view(1, 3)
        self.rad  = radius

        super().__init__(**kwargs)

    def intersect(self, hits):
        oc   = self.cent - hits.rays.orig
        # NOTE: dot product on the last axis
        d_oc = torch.einsum('ij,ij->i', hits.rays.dir, oc)
        oc_2 = torch.sum(torch.pow(oc, 2), dim = 1)
        r_2  = pow(self.rad, 2)
        disc = torch.pow(d_oc, 2) - oc_2 + r_2

        mask = disc >= 0

        ts_p = d_oc + torch.sqrt(disc.clamp(0))
        ts_n = d_oc - torch.sqrt(disc.clamp(0))
        ts   = torch.where((ts_n < ts_p) & (ts_n > t_min), ts_n, ts_p)

        mask[ts < t_min] = False

        if not torch.any(mask):
            return(False)

        ts[~mask] = torch.inf

        hits.mask           = mask
        hits.ts             = ts
        hits.det[mask, :3]  = hits.rays[mask](ts[mask])
        hits.det[mask, 3:6] = normalize(hits.det[mask, :3] - self.cent, dim = 1)

        return(True)