import torch

from utils.consts import ftype

class Object:
    def __init__(self, albedo, fuzz, **kwargs):
        # TODO: argcheck!
        self.mat = torch.empty((1, 4), dtype = ftype)
        self.mat[0, 0:3] = albedo
        self.mat[0, 3]   = fuzz

        super().__init__(**kwargs)

    def __add__(self, other):
        if isinstance(other, Object):
            return(Scene() + self + other)
        elif isinstance(other, Scene):
            return(other + self)
        else:
            raise Exception("Invalid type added to object!")

    def intersect(self, hits):        
        hits.det[:, 6:10] = self.mat

        return(super().intersect(hits))

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