import numpy as np

from utils.linalg         import *
from raytracing.raytracer import Rays

class ViewportParams():
    def __init__(
        self, 
        height       = 2.0,
        aspect_ratio = 16/9,# 4/3,
        focal_len    = 1.0, 
        eye_pos      = float([0.0, -5.0, 0.0]),
        view_target  = float([0.0, 0.0, 0.0])
    ):
        self.width       = height * aspect_ratio
        self.height      = height
        self.focal       = focal_len
        self.eye_pos     = eye_pos
        self.view_target = view_target

class Viewport():
    def __init__(self, res, params = ViewportParams()):
        self.res    = res

        self.buffer = np.zeros([res.v, res.h, 3], dtype = np.uint8)

        self.config(params)

    def config(self, params):
        pix_width  = params.width / (self.res.h - 1)
        pix_height = params.height / (self.res.v - 1)

        # NOTE: This presumes up will always be up
        view_dir  = norm(params.view_target - params.eye_pos)

        h_norm    = norm(cross(view_dir, float([0.0, 0.0, 1.0])))
        v_norm    = norm(cross(view_dir, h_norm))

        left_top  = (params.eye_pos + params.focal * view_dir) - (params.width / 2 * h_norm) - (params.height / 2 * v_norm)
        left_top  = unsqueeze(left_top, (0, 1))

        h_step    = pix_width * h_norm
        v_step    = pix_height * v_norm

        h_offset  = unsqueeze(np.arange(self.res.h, dtype = np.float32), (0, 2)) * unsqueeze(h_step, (0, 1))
        v_offset  = unsqueeze(np.arange(self.res.v, dtype = np.float32), (1, 2)) * unsqueeze(v_step, (0, 1))

        rays_orig = left_top + h_offset + v_offset
        rays_dir  = rays_orig - unsqueeze(params.eye_pos, (0, 1))

        # TODO: write complex subscript op for Rays
        # TODO: keep them in buffer-shape for now
        rays_orig = view(rays_orig, (-1, 3))
        rays_dir  = view(rays_dir, (-1, 3))

        self.rays = Rays(rays_orig, rays_dir)