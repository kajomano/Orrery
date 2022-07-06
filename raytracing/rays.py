import torch
from torch.nn.functional import normalize

from utils.settings import ftype
from utils.torch    import DmModule

class Rays(DmModule):
    def __init__(self, origins, directions, _manual = True):
        if _manual and \
        origins.shape != directions.shape or \
        len(origins.shape) != 2 or \
        origins.shape[-1] != 3 or \
        origins.dtype != ftype or \
        directions.dtype != ftype:
            raise Exception('Invalid ray parameters!')

        self.orig = origins

        if not _manual:
            self.dir = normalize(directions, dim = 1)
            super().__init__()
        else:
            self.dir    = directions
            self.device = origins.device 

    def __len__(self):
        return(self.orig.shape[0])

    def __getitem__(self, ids):
        return(Rays(
            origins    = self.orig[ids, :],
            directions = self.dir[ids, :],
            _manual    = False
        ))

    def __call__(self, ts):
        return(self.orig + ts.view(-1, 1) * self.dir)

class RayHits(DmModule):
    def __init__(self, rays, mask = None, details = None):
        self.rays = rays
        self.mask = torch.zeros((len(rays),), dtype = torch.bool, device = rays.device) if mask is None else mask
        self.det  = torch.zeros((len(rays), 7), dtype = ftype, device = rays.device) if details is None else details
        self.det[:, 0] = torch.inf
        # NOTE:
        # det[:, 0]   = t (distance between ray_orig and P)
        # det[:, 1:4] = P (hit point)
        # det[:, 4:7] = N (surface normal at P)

        self.device  = rays.device

    # def __mul__(self, other):
    #     ts_comp = self.ts < other.ts
      
    #     return(RayHits(
    #         np.where(ts_comp, self.hit_mask, other.hit_mask),
    #         np.where(ts_comp, self.ts, other.ts),
    #         np.where(unsqueeze(ts_comp, 1), self.ps, other.ps),
    #         np.where(unsqueeze(ts_comp, 1), self.ns, other.ns)
    #     ))