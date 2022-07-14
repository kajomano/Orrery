import torch

from utils.torch     import DmModule
from raytracing.rays import RayBounces

class Material(DmModule):
    def __init__(self, albedo, **kwargs):
        self.albedo = albedo.view(1, 3)

        super().__init__(**kwargs)

    def bounceTo(self, hits, directions):
        n_dir = torch.einsum('ij,ij->i', hits.ns, directions)

        bncs = RayBounces(
            hits     = hits,
            bnc_mask = n_dir > 0,
            out_dirs = directions.repeat(hits.ns.shape[0], 1),            
            alb      = torch.clamp_min(self.albedo * n_dir.view(-1, 1), 0)
        )

        return(bncs)
