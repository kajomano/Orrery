import numpy as np

from utils.linalg import *

class Rays():
    def __init__(self, origins, directions, norm_dir = True):
        self.orig  = origins
        self.dir   = norm(directions, axis = 1) if norm_dir else directions
        self.shape = origins.shape

    def view(self, shape):
        return(Rays(
            origins    = np.reshape(self.orig, list(shape) + [3]),
            directions = np.reshape(self.dir, list(shape) + [3]),
            norm_dir   = False
        ))

    def __getitem__(self, *args):
        args = args[0]
        if isinstance(args, np.ndarray) or isinstance(args, slice):
            return(Rays(
                origins    = self.orig[args, :],
                directions = self.dir[args, :],
                norm_dir   = False
            ))
        elif len(args) == 2:
            return(Rays(
                origins    = self.orig[args[0], args[1], :],
                directions = self.dir[args[0], args[1], :],
                norm_dir   = False
            ))
        else:
            raise Exception('Invalid number of ray slices!')

    def __call__(self, ts):
        return(self.orig + ts * self.dir)

class RayHits():
    def __init__(self, 
        hit_mask = np.full((0), False), 
        ts       = float_zero((0,)), 
        ps       = float_zero((0, 3)), 
        ns       = float_zero((0, 3))
    ):
        # TODO: put ts, ps and ns into a single tensor
        self.hit_mask = hit_mask
        self.ts       = ts
        self.ps       = ps
        self.ns       = ns

    def __mul__(self, other):
        ts_comp = self.ts < other.ts
      
        return(RayHits(
            np.where(ts_comp, self.hit_mask, other.hit_mask),
            np.where(ts_comp, self.ts, other.ts),
            np.where(unsqueeze(ts_comp, 1), self.ps, other.ps),
            np.where(unsqueeze(ts_comp, 1), self.ns, other.ns)
        ))