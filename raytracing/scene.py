from utils.torch     import DmModule

from raytracing.rays import RayBounceAggr

class Object(DmModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __add__(self, other):
        if isinstance(other, Object):
            return(Scene() + self + other)
        elif isinstance(other, Scene):
            return(other + self)
        else:
            raise Exception("Invalid type added to object!")

# TODO: add addition operator and offset and rotation parameters, so that local
# smaller scenes can be combined into the bigger scenes
class Scene():
    def __init__(self):
        self.obj_list = []

        super().__init__()

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

    def buildHierarchy(self):
        pass
        # genAlignedBox

    def traverse(self, rays, tracer = None):
        bncs_aggr = RayBounceAggr(rays)

        for obj in self.obj_list:
            hits = obj.intersect(rays)

            if(hits is not None):
                bncs = obj.bounce(hits, self) if tracer is None else obj.bounceTo(hits, tracer)
                bncs_aggr.aggregate(bncs)

        return(bncs_aggr)
