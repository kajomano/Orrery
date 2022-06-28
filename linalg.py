import numpy as np

def norm(vec):
    return(vec / np.linalg.norm(vec, axis = 0))

def cross(vec_a, vec_b):
    return(np.cross(vec_a, vec_b))