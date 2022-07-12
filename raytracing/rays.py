import torch
from torch.nn.functional import normalize

from utils.consts import ftype
from utils.torch  import DmModule

class Rays(DmModule):
    def __init__(self, origins, directions, _manual = False):
        # TODO: parameter check!
        self.orig   = origins
        self.dir    = directions if _manual else normalize(directions, dim = 1)
        self.device = origins.device

    def __len__(self):
        return(self.orig.shape[0])

    def __getitem__(self, ids):
        return(Rays(
            origins    = self.orig[ids, :],
            directions = self.dir[ids, :],
            _manual    = True
        ))

    def __call__(self, ts):
        return(self.orig + ts.view(-1, 1) * self.dir)

class RayHits(DmModule):
    def __init__(self, rays, mask = None, ts = None, details = None):
        self.rays = rays
        self.mask = torch.zeros((len(rays),), dtype = torch.bool, device = rays.device) if mask is None else mask
        self.ts   = torch.full((len(rays),), torch.inf, dtype = ftype, device = rays.device) if ts is None else ts
        self.det  = torch.zeros((len(rays), 10), dtype = ftype, device = rays.device) if details is None else details
        # NOTE:
        # ts          = t (distance between ray_orig and P)
        # det[:, 0:3] = P (hit point)
        # det[:, 3:6] = N (surface normal at P)
        # det[:, 6:9] = albedo
        # det[:, 9]   = fuzziness

        self.device  = rays.device

    def __mul__(self, other):
        ts_comp = (other.ts < self.ts) * other.mask

        self.mask[ts_comp]   = True
        self.ts[ts_comp]     = other.ts[ts_comp]
        self.det[ts_comp, :] = other.det[ts_comp, :]
        
        return(self)

    def squish(self):
        self.mask = self.mask.nonzero().view(-1)
        self.ts   = self.ts[self.mask]
        self.det  = self.det[self.mask, :]