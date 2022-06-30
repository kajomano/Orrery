from raytracing.geometry import Spheres

class Scenery():
    def __init__(self):
        self.spheres = Spheres()

    def __add__(self, object):
        if isinstance(object, Spheres):
            self.spheres += object
        else:
            raise Exception("Unknown object type")

        return(self)