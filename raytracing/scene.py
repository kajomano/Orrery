import torch

from utils.torch     import DmModule

from raytracing.rays import RayBounceAggr

class Object(DmModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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

    # TODO: better (actual) bvh building algorithm
    def buildBVH(self):
        aabb_list = [obj.genAlignedBox() for obj in self.obj_list]
        self.bvh  = aabb_list[0] + aabb_list[1] + aabb_list[2]
        self.bvh.children = aabb_list

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

        self._traverseRecursive(self.bvh, rays, ray_ids, bncs_aggr, tracer)
        
        return(bncs_aggr)

    # def traverse(self, rays, tracer = None):
    #     bncs_aggr = RayBounceAggr(rays)

    #     for obj in self.obj_list:
    #         hits = obj.intersect(rays)

    #         if(hits is not None):
    #             bncs = obj.bounce(hits, self) if tracer is None else obj.bounceTo(hits, tracer)
    #             bncs_aggr.aggregate(bncs)

    #     return(bncs_aggr)
