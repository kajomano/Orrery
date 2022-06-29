import numpy as np

def norm(vec, axis = 0):
    return(vec / np.linalg.norm(vec, axis = axis, keepdims = True))

def cross(vec_a, vec_b):
    return(np.cross(vec_a, vec_b))