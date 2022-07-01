import numpy as np

def float(vec_list):
    return(np.array(vec_list, dtype = np.float32))

def view(vec, shape):
    return(np.reshape(vec, shape))

def squeeze(vec, axes):
    return(np.squeeze(vec, axes))

def unsqueeze(vec, axes):
    return(np.expand_dims(vec, axes))

def cat(vec_a, vec_b, axis = 0):
    return(np.concatenate([vec_a, vec_b], axis = axis))

def length(vec, axis = -1):
    return(np.linalg.norm(vec, axis = axis, keepdims = True))

def sqrt(vec):
    return(np.sqrt(vec))

def norm(vec, axis = -1):
    return(vec / np.linalg.norm(vec, axis = axis, keepdims = True))

def cross(vec_a, vec_b):
    return(np.cross(vec_a, vec_b))

# TODO: Figure out the Einstein notation!
# TODO: Ask Benedek
# return(np.einsum('ij, ij->i', vec_a, vec_b, optimize = True))
def dot(vec_a, vec_b):
    return(np.sum(vec_a*vec_b, axis = -1))

