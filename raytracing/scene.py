class Object:
    def __init__(self, albedo, fuzz, **kwargs):
        # TODO: argcheck!
        self.alb = albedo.view(1, 3)
        self.fuz = fuzz

        super().__init__(**kwargs)

    def __add__(self, other):
        if isinstance(other, Object):
            return(Scene(self, other))
        elif isinstance(other, Scene):
            return(other + self)
        else:
            raise Exception("Invalid type added to object!")

    def intersect(self, rays):
        hits = super().intersect(rays)
        hits.det[:, 6:9] = self.alb
        hits.det[:, 9]   = self.fuz

        return(hits)

class LightSource(Object):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

# TODO: add addition operator and offset and rotation parameters, so that local
# smaller scenes can be combined into the bigger scenes
class Scene():
    def __init__(self, *args):
        self.obj_list = []
        self.ls_list  = []

        for arg in args:
            self + arg

        super().__init__()

    def __add__(self, other):
        if isinstance(other, Object):
            if isinstance(other, LightSource):
                self.ls_list.append(other)
            self.obj_list.append(other)
        elif isinstance(other, Scene):
                self.obj_list += other.obj_list
                self.ls_list  += other.ls_list
        else:
            raise Exception("Invalid type added to scene!")

        return(self)

    def to(self, device):
        for obj in self.obj_list:
            obj.to(device)