import torch
from torch.nn.functional import normalize

from utils.consts    import ftype, eps
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

class Lambertian(Material):
    def _randomUnitVect(self, hits):
        return(normalize(torch.randn((hits.ns.shape[0], 3), dtype = ftype, device = self.device), dim = 1))

    def bounce(self, hits):
        out_dirs = normalize(hits.ns + self._randomUnitVect(hits) + eps, dim = 1)

        bncs = RayBounces(
            hits     = hits,
            bnc_mask = torch.ones((hits.ns.shape[0],), dtype = torch.bool, device = self.device),
            out_dirs = out_dirs,
            alb      = self.albedo.repeat(hits.ns.shape[0], 1)
        )

        return(bncs)

# def _genScatterDir(self, hits):
#     fuz_size = hits.det[:, 9].view(-1, 1)
#     ray_norm = -torch.einsum("ij,ij->i", hits.rays.dir[hits.mask, :], hits.det[:, 3:6]).view(-1, 1)
#     ray_nout = torch.maximum(fuz_size, ray_norm)
#     ray_corr = torch.sqrt((1 - torch.pow(ray_nout, 2)) / ((1 + eps) - torch.pow(ray_norm, 2)))

#     ray_dir  = ray_corr * (hits.rays.dir[hits.mask, :] + ray_norm * hits.det[:, 3:6]) + hits.det[:, 3:6] * ray_nout

#     rand_dir = normalize(torch.randn((hits.mask.shape[0], 3), dtype = ftype, device = self.device), dim = 1)        
#     rand_dir *= fuz_size - eps # NOTE: safeguard against 0, 0, 0 ray_rand

#     ray_rand = normalize(ray_dir + rand_dir, dim = 1)

#     return(ray_rand)