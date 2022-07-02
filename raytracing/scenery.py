from raytracing.geometry import Spheres

# TODO: rename to Scene
# TODO: add addition operator and offset and rotation parameters, so that local
# smaller scenes can be combined into the bigger scenes
class Scenery():
    def __init__(self):
        self.spheres = Spheres()

    def __add__(self, object):
        if isinstance(object, Spheres):
            self.spheres += object
        else:
            raise Exception("Invalid object type!")

        return(self)