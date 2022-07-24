import torch

from utils.torch import ftype

class Rays():
    def __init__(self, origins, directions):
        self.orig   = origins
        self.dirs   = directions
        self.device = origins.device

    def __len__(self):
        return(self.orig.shape[0])

    def __getitem__(self, ids):
        return(Rays(
            origins    = self.orig[ids, :],
            directions = self.dirs[ids, :]
        ))

    def __call__(self, ts):
        return(self.orig + ts.view(-1, 1) * self.dirs)

class RayHits():
    def __init__(self, rays, hit_mask, ts, ns, ps, face):
        self.rays     = rays
        self.hit_mask = hit_mask
        self.ts       = ts   # t (distance between ray_orig and P) [len(rays)]
        self.ps       = ps   # P (hit points)                      [n, 3] 
        self.ns       = ns   # N (surface normals at Ps)           [n, 3]
        self.face     = face # True if front face hit              [n,]

class RayBounces():
    def __init__(self, hits, bnc_mask, out_dirs, alb):
        self.hits     = hits
        self.bnc_mask = bnc_mask # Bounce mask                [n,]
        self.out_dirs = out_dirs # Scattered ray directions   [n, 3]
        self.alb      = alb      # Transferred albedo         [n, 3]

class RayBounceAggr():
    def __init__(self, rays):
        self.rays = rays

        self.hit_mask = torch.zeros((len(rays),), dtype = torch.bool, device = rays.device)
        self.ts       = torch.full((len(rays),), torch.inf, dtype = ftype, device = rays.device)
        self.ps       = torch.zeros((len(rays), 3), dtype = ftype, device = rays.device)
        self.face     = torch.zeros((len(rays),), dtype = torch.bool, device = rays.device)

        self.bnc_mask = torch.zeros((len(rays),), dtype = torch.bool, device = rays.device)
        self.out_dirs = torch.zeros((len(rays), 3), dtype = ftype, device = rays.device)
        self.alb      = torch.zeros((len(rays), 3), dtype = ftype, device = rays.device)

    def aggregate(self, bncs, ray_ids):
        ts_comp = bncs.hits.ts < self.ts[ray_ids]
        ts_hits = ts_comp[bncs.hits.hit_mask]
        ts_ids  = ray_ids[ts_comp]

        self.hit_mask[ts_ids] = True
        self.ts[ts_ids]       = bncs.hits.ts[ts_comp]
        self.ps[ts_ids]       = bncs.hits.ps[ts_hits]
        self.face[ts_ids]     = bncs.hits.face[ts_hits]

        self.bnc_mask[ts_ids] = bncs.bnc_mask[ts_hits]
        self.out_dirs[ts_ids] = bncs.out_dirs[ts_hits]
        self.alb[ts_ids]      = bncs.alb[ts_hits]

        return(self)

    def generateRays(self):
        rays = Rays(
            origins    = self.ps[self.bnc_mask, :],
            directions = self.out_dirs[self.bnc_mask, :]
        )

        return(rays)
