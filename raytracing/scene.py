from math import inf
import torch

from utils.torch      import DmModule
from utils.consts     import t_min

from raytracing.rays  import RayBounceAggr

class Object(DmModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class AlignedBox(DmModule):
    def __init__(self, maxes, mins, children):
        self.maxes = maxes.view(1, 3)
        self.mins  = mins.view(1, 3)

        self.leaf = isinstance(children, Object)
        if self.leaf:
            self.children = [children]
        else:
            self.children = children

    def to(self, device):
        for child in self.children:
            child.to(device)

        super().to(device)

        return(self)

    def __add__(self, other):
        return(AlignedBox(
            torch.maximum(self.maxes, other.maxes),
            torch.minimum(self.mins, other.mins),
            [self, other]
        ))

    def intersect(self, rays):
        # TODO: Nans from 0/0!!!!
        t_0 = (self.mins - rays.orig) / rays.dirs
        t_1 = (self.maxes - rays.orig) / rays.dirs

        t_smalls, _  = torch.max(torch.minimum(t_0, t_1), dim = 1)
        t_bigs, _    = torch.min(torch.maximum(t_0, t_1), dim = 1)

        return(torch.logical_and(t_smalls <= t_bigs, t_bigs >= t_min))

# TODO: add addition operator and offset and rotation parameters, so that local
# smaller scenes can be combined into the bigger scenes
class Scene(DmModule):
    def __init__(self, **kwargs):
        self.obj_list = []
        self.bvh      = None

        super().__init__(**kwargs)

    def __add__(self, other):
        if isinstance(other, Object):
            self.obj_list.append(other)
        elif isinstance(other, Scene):
                self.obj_list += other.obj_list
        else:
            raise Exception("Invalid type added to scene!")

        return(self)

    def to(self, device):
        for obj in self.obj_list:
            obj.to(device)
        
        if self.bvh is not None:
            self.bvh.to(device)

        super().to(device)

        return(self)

    def _buildRecursive(self, bv_list, bv_ids, cents, maxes, mins, depth, max_depth):
        if len(bv_ids) == 1:
            return(bv_list[bv_ids.item()])

        if depth >= max_depth:
            maxes, _ = torch.max(maxes, dim = 0)
            mins, _  = torch.min(mins, dim = 0)

            return(AlignedBox(
                maxes,
                mins,
                [bv_list[bv_ids[idx].item()] for idx in range(len(bv_ids))]
            ))

        max, _ = torch.max(maxes, dim = 0)
        min, _ = torch.min(mins, dim = 0)
        axis   = torch.argmax(max - min).item()
        sorts  = torch.argsort(cents[:, axis], dim = 0)

        half   = len(bv_ids) // 2

        left_ids  = sorts[:half]
        right_ids = sorts[half:]

        left  = self._buildRecursive(bv_list, bv_ids[left_ids], cents[left_ids], maxes[left_ids], mins[left_ids], depth + 1, max_depth)
        right = self._buildRecursive(bv_list, bv_ids[right_ids], cents[right_ids], maxes[right_ids], mins[right_ids], depth + 1, max_depth)

        return(left + right)

    # TODO: better (actual) bvh building algorithm
    def build(self, max_depth = inf):
        bv_list = [obj.genAlignedBox() for obj in self.obj_list]
        bv_ids  = torch.arange(len(bv_list), device = self.device)

        maxes = torch.cat([bv.maxes for bv in bv_list], dim = 0)
        mins  = torch.cat([bv.mins for bv in bv_list], dim = 0)
        cents = mins + (maxes - mins) / 2

        self.bvh = self._buildRecursive(bv_list, bv_ids, cents, maxes, mins, 0, max_depth)

    def _traverseRecursive(self, bv, rays, ray_ids, bncs_aggr, tracer = None):
        hit_mask = bv.intersect(rays)

        if not torch.any(hit_mask):
            return(None)

        if bv.leaf:
            for obj in bv.children:
                hits = obj.intersect(rays[hit_mask])

                if(hits is not None):
                    bncs = obj.bounce(hits) if tracer is None else obj.bounceTo(hits, tracer)
                    bncs_aggr.aggregate(bncs, ray_ids[hit_mask])
        else:
            for child in bv.children:
                self._traverseRecursive(child, rays[hit_mask], ray_ids[hit_mask], bncs_aggr, tracer)

    def traverse(self, rays, tracer = None):
        bncs_aggr = RayBounceAggr(rays)
        ray_ids   = torch.arange(len(rays), dtype = torch.long, device = self.device)

        self._traverseRecursive(self.bvh, rays, ray_ids, bncs_aggr, tracer)

        # for obj in self.obj_list:
        #     hits = obj.intersect(rays)

        #     if(hits is not None):
        #         bncs = obj.bounce(hits) if tracer is None else obj.bounceTo(hits, tracer)
        #         bncs_aggr.aggregate(bncs, ray_ids)
        
        return(bncs_aggr)
