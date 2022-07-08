import numpy as np
import torch
from torch.nn.functional import normalize

from utils.settings  import ftype
from utils.torch     import DmModule
from raytracing.rays import Rays

class ViewportParams():
    def __init__(
        self, 
        height       = 2.0,
        aspect_ratio = 16/9,
        focal_len    = 1.0, 
        eye_pos      = torch.tensor([0.0, -5.0, 0.0], dtype = ftype),
        view_target  = torch.tensor([0.0, 0.0, 0.0], dtype = ftype)
    ):
        self.width       = height * aspect_ratio
        self.height      = height
        self.focal       = focal_len
        self.eye_pos     = eye_pos
        self.view_target = view_target

class Viewport(DmModule):
    def __init__(self, res, params = ViewportParams()):
        self.res = res

        # NOTE: the buffer is strictly a numpy array, to enforce memory
        # placement and correct dtype
        self.buffer = np.zeros([res.v, res.h, 3], dtype = np.uint8)

        super().__init__()

        self.config(params)        

    def config(self, params):
        self.pix_width  = params.width / (self.res.h - 1)
        self.pix_height = params.height / (self.res.v - 1)
        
        view_dir  = normalize(params.view_target - params.eye_pos, dim = 0)

        # NOTE: This presumes up will always be up
        h_norm    = normalize(torch.cross(view_dir, torch.tensor([0.0, 0.0, 1.0], dtype = ftype)), dim = 0)
        v_norm    = normalize(torch.cross(view_dir, h_norm), dim = 0)

        left_top  = (params.eye_pos + params.focal * view_dir) - (params.width / 2 * h_norm) - (params.height / 2 * v_norm)
        left_top  = left_top.view(1, 1, 3)

        h_step    = (self.pix_width * h_norm).view(1, 1, 3)
        v_step    = (self.pix_height * v_norm).view(1, 1, 3)

        h_offset  = h_step * torch.arange(self.res.h, dtype = ftype).view(1, self.res.h, 1)
        v_offset  = v_step * torch.arange(self.res.v, dtype = ftype).view(self.res.v, 1, 1)

        rays_orig = left_top + h_offset + v_offset
        rays_dir  = rays_orig - params.eye_pos.view(1, 3)

        self.rays = Rays(rays_orig.view(-1, 3), rays_dir.view(-1, 3)).to(self.device)

    def getBuffer(self):
        return(self.buffer)

    def setBuffer(self, buffer):
        self.buffer = buffer

    def getBufferWithCrosshair(self, off_h = 0, off_v = 0):
        mid_h = self.res.h // 2 + off_h
        mid_v = self.res.v // 2 - off_v      

        self.buffer[(mid_v - 1):(mid_v + 2), (mid_h - 5):(mid_h + 6), :] = 255
        self.buffer[(mid_v - 5):(mid_v + 6), (mid_h - 1):(mid_h + 2), :] = 255

        self.buffer[mid_v, (mid_h - 4):(mid_h + 5), :] = 0
        self.buffer[(mid_v - 4):(mid_v + 5), mid_h, :] = 0

        ray_orig = self.rays.orig.view(self.res.v, self.res.h, 3)[mid_v, mid_h, :]
        ray_dir  = self.rays.dir.view(self.res.v, self.res.h, 3)[mid_v, mid_h, :]

        print(f'origin: {ray_orig[0]:.4f}, {ray_orig[1]:.4f}, {ray_orig[2]:.4f}')
        print(f'direc.: {ray_dir[0]:.4f}, {ray_dir[1]:.4f}, {ray_dir[2]:.4f}')

        return(self.buffer)
