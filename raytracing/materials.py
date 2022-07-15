import torch
from torch.nn.functional import normalize

from utils.consts    import ftype, eps
from utils.torch     import DmModule

from raytracing.rays import RayBounces

class Material(DmModule):
    def __init__(self, albedo, **kwargs):
        self.alb = albedo.view(1, 3)

        super().__init__(**kwargs)

    def bounceTo(self, hits, tracer):
        n_dir = torch.clamp_min(torch.einsum('ij,ij->i', hits.ns, tracer.dir_light), 0)
        col   = torch.lerp(tracer.col_ambnt, tracer.col_light, n_dir.view(-1, 1))

        bncs = RayBounces(
            hits     = hits,
            bnc_mask = n_dir > 0,
            out_dirs = tracer.dir_light.repeat(hits.ns.shape[0], 1),
            alb      = self.alb * col
        )

        return(bncs)

class Diffuse(Material):
    def _randomUnitVect(self, hits):
        return(normalize(torch.randn((hits.ns.shape[0], 3), dtype = ftype, device = self.device), dim = 1))

    def bounce(self, hits):
        out_dirs = normalize(hits.ns + self._randomUnitVect(hits) + eps, dim = 1)

        bncs = RayBounces(
            hits     = hits,
            bnc_mask = torch.ones((hits.ns.shape[0],), dtype = torch.bool, device = self.device),
            out_dirs = out_dirs,
            alb      = self.alb.repeat(hits.ns.shape[0], 1)
        )

        return(bncs)

class Shiny(Diffuse):
    def bounce(self, hits):
        out_dirs = hits.rays.dirs[hits.hit_mask, :] - 2 * torch.einsum('ij,ij->i', hits.rays.dirs[hits.hit_mask, :], hits.ns).view(-1, 1) * hits.ns

        bncs = RayBounces(
            hits     = hits,
            bnc_mask = torch.ones((hits.ns.shape[0],), dtype = torch.bool, device = self.device),
            out_dirs = out_dirs,
            alb      = self.alb.repeat(hits.ns.shape[0], 1)
        )

        return(bncs)

class Metal(Diffuse):
    def __init__(self, fuzz, **kwargs):
        self.fuzz = fuzz

        super().__init__(**kwargs)

    # NOTE: turns out my fuzzy-bounce formulation looks better, because it avoids
    # absorbed rays at shallow angles. If absorption can be kept out in other materials,
    # the bnc_mask and connected complexity could be removed.

    # def bounce(self, hits):
    #     out_dirs = hits.rays.dirs[hits.hit_mask, :] - 2 * torch.einsum('ij,ij->i', hits.rays.dirs[hits.hit_mask, :], hits.ns).view(-1, 1) * hits.ns
    #     out_dirs += self._randomUnitVect(hits) * self.fuzz
    #     out_dirs = normalize(out_dirs)

    #     bnc_mask = torch.einsum('ij,ij->i', out_dirs, hits.ns) > 0

    #     bncs = RayBounces(
    #         hits     = hits,
    #         bnc_mask = bnc_mask,
    #         out_dirs = out_dirs,
    #         alb      = self.alb * bnc_mask.view(-1, 1)
    #     )

    #     return(bncs)

    def bounce(self, hits):
        ray_norm = -torch.einsum("ij,ij->i", hits.rays.dirs[hits.hit_mask, :], hits.ns).view(-1, 1)
        ray_nout = torch.clamp_min(ray_norm, self.fuzz)
        ray_corr = torch.sqrt((1 - torch.pow(ray_nout, 2)) / ((1 + eps) - torch.pow(ray_norm, 2)))

        ray_dir  = ray_corr * (hits.rays.dirs[hits.hit_mask, :] + ray_norm * hits.ns) + hits.ns * ray_nout

        rand_dir = normalize(torch.randn((hits.ns.shape[0], 3), dtype = ftype, device = self.device), dim = 1)        
        rand_dir *= self.fuzz - eps # NOTE: safeguard against 0, 0, 0 ray_rand

        ray_rand = normalize(ray_dir + rand_dir, dim = 1)

        bncs = RayBounces(
            hits     = hits,
            bnc_mask = torch.ones((hits.ns.shape[0],), dtype = torch.bool, device = self.device),
            out_dirs = ray_rand,
            alb      = self.alb.repeat(hits.ns.shape[0], 1)
        )

        return(bncs)


class Glowing(Material):
    def __init__(self, col_glow, **kwargs):
        self.col_glow = col_glow.view(1, 3)

        super().__init__(albedo = col_glow / 255, **kwargs)

    def bounce(self, hits):
        bncs = RayBounces(
            hits     = hits,
            bnc_mask = torch.zeros((hits.ns.shape[0],), dtype = torch.bool, device = self.device),
            out_dirs = torch.zeros((hits.ns.shape[0], 3), dtype = ftype, device = self.device),
            alb      = self.col_glow.repeat(hits.ns.shape[0], 1)
        )

        return(bncs)