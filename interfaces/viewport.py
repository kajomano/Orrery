import numpy as np

import torch
from torch.nn.functional import normalize

from utils.consts    import pi
from utils.torch     import DmModule, ftype
from utils.rand      import randInCircle, randInSquare

from raytracing.rays import Rays

from copy            import deepcopy
from multiprocessing import Lock, shared_memory
from math            import tan

class ViewportParams():
    def __init__(
        self,
        cam_pos    = np.array([0.0, -10.0, -1.0], dtype = np.single),
        cam_target = np.array([0.0, 0.0, 0.0],    dtype = np.single),
        vfov       = 60,
        r_lens     = 0.05
    ):
        # TODO: params check!
        self.cam_pos    = cam_pos
        self.cam_target = cam_target
        self.vfov       = vfov
        self.r_lens     = r_lens

class Viewport(DmModule):
    def __init__(self, res, params = ViewportParams()):
        self.res = res

        self.setParams(params)
        self._createBuffer()

        # self.buffer_lock = Lock()

        # buff_tmp = np.zeros([res.v, res.h, 3], dtype = np.uint8)
        # self.buff_shm = shared_memory.SharedMemory(create = True, size = buff_tmp.nbytes)
        # buff_ptr = np.ndarray(buff_tmp.shape, dtype = np.uint8, buffer = self.buff_shm.buf)
        # buff_ptr[:] = buff_tmp[:]
        # self.buff_ptr = None

    def setParams(self, params):
        self.params = deepcopy(params)

        theta  = params.vfov / 360 * 2 * pi
        height = tan(theta / 2) * 2
        width  = height * self.res.ar

        view_dir   = params.cam_target - params.cam_pos
        focus_dist = np.linalg.norm(view_dir, axis = 0)
        view_dir  /= focus_dist

        # NOTE: This presumes up will always be up
        self.h_norm  = np.cross(view_dir, np.array([0.0, 0.0, 1.0], dtype = np.single)) 
        self.h_norm /= np.linalg.norm(self.h_norm, axis = 0)
        self.v_norm  = np.cross(view_dir, self.h_norm)
        self.v_norm /= np.linalg.norm(self.v_norm, axis = 0)

        left_top = params.cam_pos + (view_dir - (width / 2 * self.h_norm) - (height / 2 * self.v_norm)) * focus_dist
        left_top = left_top.reshape(1, 1, 3)

        self.h_step = (width / (self.res.h - 1) * focus_dist) * self.h_norm
        self.v_step = (height / (self.res.v - 1) * focus_dist) * self.v_norm

        h_offset  = self.h_step.reshape(1, 1, 3) * np.arange(self.res.h, dtype = np.single).reshape(1, self.res.h, 1)
        v_offset  = self.v_step.reshape(1, 1, 3) * np.arange(self.res.v, dtype = np.single).reshape(self.res.v, 1, 1)            

        self.pixels = (left_top + h_offset + v_offset).reshape(-1, 3)

    def __len__(self):
        return(self.pixels.shape[0])

    def getRays(self, rand = True):
        orig = torch.tensor(self.params.cam_pos, dtype = ftype, device = self.device).view(1, 3).repeat(len(self), 1)
        dirs = torch.tensor(self.pixels, dtype = ftype, device = self.device)

        if rand:
            h_norm = torch.tensor(self.h_norm, dtype = ftype, device = self.device).view(1, 3)
            v_norm = torch.tensor(self.v_norm, dtype = ftype, device = self.device).view(1, 3)

            h_step = torch.tensor(self.h_step, dtype = ftype, device = self.device).view(1, 3)
            v_step = torch.tensor(self.v_step, dtype = ftype, device = self.device).view(1, 3)

            orig_rh, orig_rv = randInCircle(len(self), self.device)
            orig += (orig_rh * h_norm * self.params.r_lens) + (orig_rv * v_norm * self.params.r_lens)

            dirs_rh, dirs_rv = randInSquare(len(self), self.device)
            dirs += (dirs_rh * h_step) + (dirs_rv * v_step)

        rays = Rays(
            origins    = orig,
            directions = normalize(dirs - orig, dim = 1)
        )

        return(rays)

    def _createBuffer(self):
        self.buffer = np.zeros([self.res.v, self.res.h, 3], dtype = np.uint8)

    def getBuffer(self):
        return(self.buffer)


    # def getBuffer(self):
    #     if self.buff_ptr is None:
    #         # TODO: create flag is missing!
    #         self.buff_ptr = np.ndarray([self.res.v, self.res.h, 3], dtype = np.uint8, buffer = self.buff_shm.buf)
    #     return(self.buff_ptr)

    # TODO: clean up shm
    # def __del__(self):
    #     self.buff_shm.close()

    #     if current_process().name == 'MainProcess':
    #         print("cleaned main")
    #         self.buff_shm.unlink()
    #     else:
    #         print("cleaned off")

    # def getBufferWithCrosshair(self, off_h = 0, off_v = 0):
    #     mid_h = self.res.h // 2 + off_h
    #     mid_v = self.res.v // 2 - off_v      

    #     self.buffer[(mid_v - 1):(mid_v + 2), (mid_h - 5):(mid_h + 6), :] = 255
    #     self.buffer[(mid_v - 5):(mid_v + 6), (mid_h - 1):(mid_h + 2), :] = 255

    #     self.buffer[mid_v, (mid_h - 4):(mid_h + 5), :] = 0
    #     self.buffer[(mid_v - 4):(mid_v + 5), mid_h, :] = 0

    #     ray_orig = self.rays.orig.view(self.res.v, self.res.h, 3)[mid_v, mid_h, :]
    #     ray_dir  = self.rays.dir.view(self.res.v, self.res.h, 3)[mid_v, mid_h, :]

    #     print(f'origin: {ray_orig[0]:.4f}, {ray_orig[1]:.4f}, {ray_orig[2]:.4f}')
    #     print(f'direc.: {ray_dir[0]:.4f}, {ray_dir[1]:.4f}, {ray_dir[2]:.4f}')

    #     return(self.buffer)
