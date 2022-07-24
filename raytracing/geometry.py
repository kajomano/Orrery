import torch
from torch.nn.functional import normalize

from utils.torch      import ftype
from utils.consts     import t_min

from raytracing.rays  import RayHits
from raytracing.scene import Object, AlignedBox

class Sphere(Object):
    def __init__(self, center, radius, **kwargs):
        if center.shape != torch.Size([3]) or \
        center.dtype != ftype or \
        not isinstance(radius, (int, float)):
            raise Exception("Invalid sphere parameters!")
            
        self.cent = center.view(1, 3)
        self.rad  = radius

        super().__init__(**kwargs)

    def genAlignedBox(self):
        rads = torch.full([3], self.rad, dtype = ftype, device = self.device)

        return(AlignedBox(
            self.cent + rads, 
            self.cent - rads,
            self
        ))

    def intersect(self, rays):
        oc   = self.cent - rays.orig
        # NOTE: dot product on the last axis
        d_oc = torch.einsum('ij,ij->i', rays.dirs, oc)
        oc_2 = torch.sum(torch.pow(oc, 2), dim = 1)
        r_2  = pow(self.rad, 2)
        disc = torch.pow(d_oc, 2) - oc_2 + r_2

        hit_mask = disc >= 0

        ts_p = d_oc + torch.sqrt(disc.clamp(0))
        ts_n = d_oc - torch.sqrt(disc.clamp(0))
        face = ts_n > t_min
        ts   = torch.where(face, ts_n, ts_p)

        hit_mask[ts < t_min] = False

        if not torch.any(hit_mask):
            return(None)

        ts[~hit_mask] = torch.inf

        ps   = rays[hit_mask](ts[hit_mask])
        ns   = normalize(ps - self.cent, dim = 1)
        face = face[hit_mask]

        hits = RayHits(
            rays     = rays,
            hit_mask = hit_mask,
            ts       = ts,
            ps       = ps,
            ns       = ns,
            face     = face
        )

        return(hits)    