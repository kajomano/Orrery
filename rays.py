import numpy as np

class Rays():
    def __init__(self, origin, direction):
        self.orig = origin
        self.dir  = np.linalg.norm(direction, axis = 0)

    # def __call__(self, t):
    #     return(self.orig + t * self.dir)