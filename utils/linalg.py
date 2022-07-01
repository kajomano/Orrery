import numpy as np

float_type = np.float32
eps        = 1e-08

# Creator functions ============================================================
def float(vec_list):
    return(np.array(vec_list, dtype = float_type))

def float_zero(shape):
    return(np.zeros(shape, dtype = float_type))

# Dimension control ============================================================
def view(vec, shape):
    return(np.reshape(vec, shape))

def squeeze(vec, axes):
    return(np.squeeze(vec, axes))

def unsqueeze(vec, axes):
    return(np.expand_dims(vec, axes))

# Concatenation ================================================================
def cat(vec_a, vec_b, axis = 0):
    return(np.concatenate([vec_a, vec_b], axis = axis))

# Vector lengths ===============================================================
def sqrt(vec):
    return(np.sqrt(vec))

def norm(vec, axis = -1):
    return(vec / (np.linalg.norm(vec, axis = axis, keepdims = True) + eps))

# Vector algebra ===============================================================
def cross(vec_a, vec_b):
    return(np.cross(vec_a, vec_b))

# TODO: Figure out the Einstein notation!
# TODO: Ask Benedek
# return(np.einsum('ij, ij->i', vec_a, vec_b, optimize = True))
def dot(vec_a, vec_b):
    return(np.sum(vec_a*vec_b, axis = -1))

