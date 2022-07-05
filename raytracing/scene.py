from raytracing.geometry import Spheres
from utils.torch         import DmModule

# TODO: add addition operator and offset and rotation parameters, so that local
# smaller scenes can be combined into the bigger scenes
class Scene(DmModule):
    def __init__(self):
        self.spheres = Spheres()

        super().__init__()

    def __add__(self, object):
        if isinstance(object, Spheres):
            self.spheres += object
        else:
            raise Exception("Invalid object type added to scene!")

        return(self)