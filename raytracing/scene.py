import torch

from utils.torch         import DmModule, ftype

from raytracing.rays     import RayBounceAggr

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

        t_mins, _  = torch.max(torch.minimum(t_0, t_1), dim = 1)
        t_maxes, _ = torch.min(torch.maximum(t_0, t_1), dim = 1)

        return(t_mins <= t_maxes)

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

    def _sortBVs(self, bv_ids, maxes, mins, centers):
        maxes = maxes[bv_ids]
        mins  = mins[bv_ids]

        max, _ = torch.max(maxes, dim = 0)
        min, _ = torch.min(mins, dim = 0)
        axis   = torch.argmax(max - min).item()

        return(torch.argsort(centers[bv_ids, axis]))

    def _buildBVHrecursive(self, bv_list, maxes, mins, centers, bv_ids, depth, max_depth):
        if len(bv_ids) == 1:
            return(bv_list[bv_ids.item()])

        if depth >= max_depth:
            maxes, _ = torch.max(maxes[bv_ids, :], dim = 0)
            mins, _  = torch.min(mins[bv_ids, :], dim = 0)

            return(AlignedBox(
                maxes,
                mins,
                [bv_list[bv_ids[idx].item()] for idx in range(len(bv_ids))]
            ))

        sorted_ids = self._sortBVs(bv_ids, maxes, mins, centers)

        half  = len(bv_ids) // 2
        left  = self._buildBVHrecursive(bv_list, maxes, mins, centers, sorted_ids[:half], depth + 1, max_depth)
        right = self._buildBVHrecursive(bv_list, maxes, mins, centers, sorted_ids[half:], depth + 1, max_depth)

        return(left + right)

    # TODO: better (actual) bvh building algorithm
    def buildBVH(self, max_depth = 5):
        bv_list  = [obj.genAlignedBox() for obj in self.obj_list]

        maxes   = torch.cat([bv.maxes for bv in bv_list], dim = 0)
        mins    = torch.cat([bv.mins for bv in bv_list], dim = 0)
        centers = mins + (maxes - mins) / 2
        bv_ids  = torch.arange(len(bv_list), dtype = torch.long, device = self.device)

        self.bvh = self._buildBVHrecursive(bv_list, maxes, mins, centers, bv_ids, 0, max_depth)

    def _traverseRecursive(self, bv, rays, ray_ids, bncs_aggr, tracer = None):
        hit_mask = bv.intersect(rays)

        if not torch.any(hit_mask):
            return(None)

        if bv.leaf:
            for obj in bv.children:
                hits = obj.intersect(rays[hit_mask])

                if(hits is not None):
                    bncs = obj.bounce(hits, self) if tracer is None else obj.bounceTo(hits, tracer)
                    bncs_aggr.aggregate(bncs, ray_ids[hit_mask])
        else:
            for child in bv.children:
                self._traverseRecursive(child, rays[hit_mask], ray_ids[hit_mask], bncs_aggr, tracer)

    def traverse(self, rays, tracer = None):
        bncs_aggr = RayBounceAggr(rays)
        ray_ids   = torch.arange(len(rays), dtype = torch.long, device = self.device)

        for obj in self.bvh.children[0].children:
            hits = obj.children[0].intersect(rays)

            if(hits is not None):
                bncs = obj.children[0].bounceTo(hits, tracer)
                bncs.alb[:] = torch.tensor([[0, 255, 0]], dtype = ftype, device = self.device)
                bncs_aggr.aggregate(bncs, ray_ids)

        for obj in self.bvh.children[1].children:
            hits = obj.children[0].intersect(rays)

            if(hits is not None):
                bncs = obj.children[0].bounceTo(hits, tracer)
                bncs.alb[:] = torch.tensor([[255, 0, 0]], dtype = ftype, device = self.device)
                bncs_aggr.aggregate(bncs, ray_ids)

        # ----------------------------------------------------------------------

        # for obj in self.bvh.children[0].children[0].children:
        #     hits = obj.children[0].intersect(rays)

        #     if(hits is not None):
        #         bncs = obj.children[0].bounceTo(hits, tracer)
        #         bncs.alb[:] = torch.tensor([[0, 255, 0]], dtype = ftype, device = self.device)
        #         bncs_aggr.aggregate(bncs, ray_ids)

        # for obj in self.bvh.children[0].children[1].children:
        #     hits = obj.children[0].intersect(rays)

        #     if(hits is not None):
        #         bncs = obj.children[0].bounceTo(hits, tracer)
        #         bncs.alb[:] = torch.tensor([[255, 0, 0]], dtype = ftype, device = self.device)
        #         bncs_aggr.aggregate(bncs, ray_ids)

        # for obj in self.bvh.children[1].children[0].children:
        #     hits = obj.children[0].intersect(rays)

        #     if(hits is not None):
        #         bncs = obj.children[0].bounceTo(hits, tracer)
        #         bncs.alb[:] = torch.tensor([[0, 0, 255]], dtype = ftype, device = self.device)
        #         bncs_aggr.aggregate(bncs, ray_ids)

        # for obj in self.bvh.children[1].children[1].children:
        #     hits = obj.children[0].intersect(rays)

        #     if(hits is not None):
        #         bncs = obj.children[0].bounceTo(hits, tracer)
        #         bncs.alb[:] = torch.tensor([[0, 0, 0]], dtype = ftype, device = self.device)
        #         bncs_aggr.aggregate(bncs, ray_ids)

        # self._traverseRecursive(self.bvh, rays, ray_ids, bncs_aggr, tracer)
        
        return(bncs_aggr)
