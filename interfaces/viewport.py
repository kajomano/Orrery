import torch
from torch.nn.functional import normalize

from utils.consts    import pi
from utils.torch     import DmModule, ftype
from utils.rand      import randInCircle, randInSquare

from raytracing.rays import Rays

from copy            import deepcopy
from multiprocessing import shared_memory
from math            import tan

class ViewportParams():
    def __init__(
        self,
        cam_pos    = torch.tensor([10.0, -10.0, 3.0], dtype = ftype),
        # cam_pos    = torch.tensor([0.0, -10.0, 30.0], dtype = ftype),
        cam_target = torch.tensor([0.0, 0.0, 0.0],    dtype = ftype),
        vfov       = 40,
        r_lens     = 0.05
    ):
        # TODO: params check!
        self.cam_pos    = cam_pos
        self.cam_target = cam_target
        self.vfov       = vfov
        self.r_lens     = r_lens

class Viewport(DmModule):
    def __init__(self, res, params = ViewportParams(), **kwargs):
        self.res = res

        self.params_init = False
        self.setParams(params)

        self._initBuffer()

        super().__init__(**kwargs)

    def setParams(self, params):
        self.params = deepcopy(params)

        theta  = params.vfov / 360 * 2 * pi
        height = tan(theta / 2) * 2
        width  = height * self.res.ar

        view_dir   = params.cam_target - params.cam_pos
        focus_dist = torch.norm(view_dir)
        view_dir  /= focus_dist

        # NOTE: This presumes up will always be up
        h_norm  = normalize(torch.cross(view_dir, torch.tensor([0.0, 0.0, 1.0], dtype = ftype)), dim = 0)
        v_norm  = normalize(torch.cross(view_dir, h_norm), dim = 0)

        left_top = params.cam_pos + (view_dir - (width / 2 * h_norm) - (height / 2 * v_norm)) * focus_dist
        left_top = left_top.view(1, 1, 3)

        h_step = (width / (self.res.h - 1) * focus_dist) * h_norm
        v_step = (height / (self.res.v - 1) * focus_dist) * v_norm

        h_offset = h_step.view(1, 1, 3) * torch.arange(self.res.h, dtype = ftype).view(1, self.res.h, 1)
        v_offset = v_step.view(1, 1, 3) * torch.arange(self.res.v, dtype = ftype).view(self.res.v, 1, 1)            

        pixels = (left_top + h_offset + v_offset).view(-1, 3)

        self._setParams([
            params.cam_pos,
            pixels,
            h_norm,
            v_norm,
            h_step,
            v_step,
            torch.tensor([params.r_lens], dtype = ftype)
        ])

    def getParams(self):
        return(deepcopy(self.params))

    def _setParams(self, params_list):
        if not self.params_init:
            self.params_shm_list = [shared_memory.SharedMemory(create = True, size = param.element_size() * param.numel()) for param in params_list]
            self.params_init = True

        for i, param in enumerate(self._getParams()):
            param.view(params_list[i].shape).copy_(params_list[i])
        
    def _getParams(self):
        return([torch.frombuffer(param_shm.buf, dtype = ftype) for param_shm in self.params_shm_list])

    def __len__(self):
        return(self.res.v * self.res.h)

    def getRays(self, rand = True):
        orig, pixs, h_norm, v_norm, h_step, v_step, r_lens = self._getParams()

        # NOTE: [:3] subsets because of a bug in pytorch
        orig = orig.to(self.device)[:3].view(1, 3).repeat(len(self), 1)
        pixs = pixs.to(self.device).view(-1, 3)

        if rand:
            h_norm = h_norm[:3].to(self.device).view(1, 3)
            v_norm = v_norm[:3].to(self.device).view(1, 3)

            h_step = h_step[:3].to(self.device).view(1, 3)
            v_step = v_step[:3].to(self.device).view(1, 3)

            r_lens = r_lens[:1].to(self.device)

            orig_rh, orig_rv = randInCircle(len(self), self.device)
            orig = orig + (orig_rh * h_norm * r_lens) + (orig_rv * v_norm * r_lens)

            pixs_rh, pixs_rv = randInSquare(len(self), self.device)
            pixs = pixs + (pixs_rh * h_step) + (pixs_rv * v_step)

        rays = Rays(
            origins    = orig,
            directions = normalize(pixs - orig, dim = 1)
        )

        return(rays)

    def _initBuffer(self):
        buff_tmp      = torch.zeros([self.res.v, self.res.h, 3], dtype = torch.uint8)
        self.buff_shm = shared_memory.SharedMemory(create = True, size = buff_tmp.element_size() * buff_tmp.numel())
        self.getBuffer().copy_(buff_tmp)

    def getBuffer(self):
        return(torch.frombuffer(self.buff_shm.buf, dtype = torch.uint8).view(self.res.v, self.res.h, 3))