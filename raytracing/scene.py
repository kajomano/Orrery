from math import inf
import torch

from utils.torch      import DmModule, ftype
from utils.consts     import t_min

from raytracing.rays  import RayBounceAggr

class Object(DmModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class AlignedBox(DmModule):
    def __init__(self, maxes, mins, left, right = None):
        self.maxes = maxes.view(1, 3)
        self.mins  = mins.view(1, 3)
        self.left  = left
        self.right = right
        self.leaf  = isinstance(left, Object)

    def to(self, device):
        self.left.to(device)
        if not self.leaf:
            self.right.to(device)

        super().to(device)

        return(self)

    def __add__(self, other):
        return(AlignedBox(
            torch.maximum(self.maxes, other.maxes),
            torch.minimum(self.mins, other.mins),
            self,
            other
        ))

    def intersect(self, rays):
        # TODO: Nans from 0/0!!!!
        t_0 = (self.mins - rays.orig) / rays.dirs
        t_1 = (self.maxes - rays.orig) / rays.dirs

        t_smalls, _ = torch.max(torch.minimum(t_0, t_1), dim = 1)
        t_bigs, _   = torch.min(torch.maximum(t_0, t_1), dim = 1)

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

    def _splitCost(self, exts_sorted):
        mins, _ = torch.min(exts_sorted[:, 0, :], dim = 0)
        maxs, _ = torch.max(exts_sorted[:, 1, :], dim = 0)
        ranges  = maxs - mins

        area = (ranges[0] * ranges[1] + ranges[1] * ranges[2] + ranges[2] * ranges[0]).item()

        # NOTE: I don't grok this heuristic, explanation from:
        # https://jacco.ompf2.com/2022/04/18/how-to-build-a-bvh-part-2-faster-rays/
        return(area * len(exts_sorted))

    def _buildRecursive(self, bv_list, bv_ids, exts):
        if len(bv_ids) == 1:
            return(bv_list[bv_ids.item()])

        # Find axis and split with lowest area
        cost_lowest = inf
        for ax in range(3):
            # Sort extents by centroid positions in the axis
            ids_sorted  = torch.argsort(exts[:, 2, ax]) # centroids
            exts_sorted = exts[ids_sorted, :2, :]

            # Find the left and right bounding box area of the split
            for split in range(len(bv_ids) - 1):
                cost = self._splitCost(exts_sorted[:(split + 1)]) + self._splitCost(exts_sorted[(split + 1):])

                if cost < cost_lowest:
                    left_ids    = ids_sorted[:(split + 1)]
                    right_ids   = ids_sorted[(split + 1):]
                    cost_lowest = cost

        left  = self._buildRecursive(bv_list, bv_ids[left_ids], exts[left_ids])
        right = self._buildRecursive(bv_list, bv_ids[right_ids], exts[right_ids])

        return(left + right)

    def build(self):
        bv_list = [obj.genAlignedBox() for obj in self.obj_list]
        bv_ids  = torch.arange(len(bv_list), device = self.device)

        # Tensorise mins and maxes
        mins  = torch.cat([bv.mins for bv in bv_list], dim = 0)
        maxes = torch.cat([bv.maxes for bv in bv_list], dim = 0)
        cents = mins + (maxes - mins) / 2

        # Compile extents into a single tensor
        exts = torch.cat(
            [
                mins.view(-1, 1, 3),
                maxes.view(-1, 1, 3),
                cents.view(-1, 1, 3)
            ],
            dim = 1
        )

        # Call the recursive algorithm
        self.bvh = self._buildRecursive(bv_list, bv_ids, exts)

    def _traverseRecursive(self, bv, rays, ray_ids, bncs_aggr, tracer = None):
        if bv.leaf:
            hits = bv.left.intersect(rays)

            if(hits is not None):
                bncs = bv.left.bounce(hits) if tracer is None else bv.left.bounceTo(hits, tracer)
                bncs_aggr.aggregate(bncs, ray_ids)
        else:
            hit_mask_left = bv.left.intersect(rays)
            if torch.any(hit_mask_left):
               self._traverseRecursive(bv.left, rays[hit_mask_left], ray_ids[hit_mask_left], bncs_aggr, tracer)

            hit_mask_right = bv.right.intersect(rays)
            if torch.any(hit_mask_right):
               self._traverseRecursive(bv.right, rays[hit_mask_right], ray_ids[hit_mask_right], bncs_aggr, tracer)

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
