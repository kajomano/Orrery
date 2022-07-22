import torch
from torch.nn.functional import normalize

from raytracing.rays import Rays, RayBounceAggr

from utils.consts    import ftype
from utils.torch     import DmModule
from utils.rand      import randInCircle, randInSquare

class RayTracer(DmModule):
    def __init__(self, 
        scene, 
        # col_sky     = torch.tensor([8, 22, 38],  dtype = ftype),
        # col_horizon = torch.tensor([35, 58, 84], dtype = ftype),
        # col_ground  = torch.tensor([21, 28, 36], dtype = ftype),
        col_sky     = torch.tensor([93, 156, 222],  dtype = ftype),
        col_horizon = torch.tensor([220, 226, 232], dtype = ftype),
        col_ground  = torch.tensor([220, 226, 232], dtype = ftype),
        **kwargs
    ):
        # TODO: parameter check!
        self.scene     = scene
        self.col_sky   = col_sky
        self.col_hrzn  = col_horizon
        self.col_grnd  = col_ground

        self.buffer = None

        super().__init__(**kwargs)

    def trace(self, rays):
        bncs_aggr = RayBounceAggr(rays)

        for obj in self.scene.obj_list:
            hits = obj.intersect(rays)

            if(hits is not None):
                bncs = obj.bounce(hits)
                bncs_aggr.aggregate(bncs)

        return(bncs_aggr)

    def _shadeNohits(self, bncs_aggr):
        rays_z = bncs_aggr.rays.dirs[~bncs_aggr.hit_mask, 2].view(-1, 1)

        return(torch.where(
            rays_z > 0,
            torch.lerp(self.col_hrzn, self.col_sky, rays_z),
            torch.lerp(self.col_hrzn, self.col_grnd, rays_z)
        ))

    def _initBuffer(self, vport):
        self.buffer = torch.zeros((len(vport), 3), dtype = ftype, device = self.device)

    def _dumpBuffer(self, vport):
        self.buffer = torch.sqrt(self.buffer / 255) * 255 # Gamma correction
        self.buffer = torch.clamp_max(self.buffer, 255)   # Basic HDR to LDR conversion

        vport.getBuffer()[:] = self.buffer.type(torch.uint8).view(vport.res.v, vport.res.h, 3).cpu().numpy()[:]

    def _getParams(self, vport):
        rays_target = torch.tensor(vport.rays_target, dtype = ftype, device = self.device)
        eye_pos     = torch.tensor(vport.eye_pos, dtype = ftype, device = self.device).view(1, 3)

        h_norm = torch.tensor(vport.h_norm, dtype = ftype, device = self.device).view(1, 3)
        v_norm = torch.tensor(vport.v_norm, dtype = ftype, device = self.device).view(1, 3)

        h_step = torch.tensor(vport.h_step, dtype = ftype, device = self.device).view(1, 3)
        v_step = torch.tensor(vport.v_step, dtype = ftype, device = self.device).view(1, 3)

        return(rays_target, eye_pos, h_norm, v_norm, h_step, v_step, vport.aperture)
  
# ==============================================================================
class SimpleTracer(RayTracer):
    def __init__(self, 
        scene, 
        dir_light = torch.tensor([1, 0, 1],       dtype = ftype),
        col_light = torch.tensor([200, 200, 200], dtype = ftype),
        col_ambnt = torch.tensor([50, 50, 50],    dtype = ftype),
        **kwargs
    ):
        # TODO: parameter check!
        self.dir_light = normalize(dir_light, dim = 0).view(1, 3)
        self.col_light = col_light
        self.col_ambnt = col_ambnt

        super().__init__(scene, **kwargs)

    def render(self, vport):
        self._initBuffer(vport)

        # TODO: this is broken!
        rays_target, eye_pos, *_ = self._getParams(vport)

        rays = Rays(
            eye_pos.repeat(rays_target.shape[0], 1),
            normalize(rays_target - eye_pos, dim = 1),
            _manual = True
        )

        bncs_aggr = RayBounceAggr(rays)

        for obj in self.scene.obj_list:
            hits = obj.intersect(rays)

            if(hits is not None):
                bncs = obj.bounceTo(hits, self)
                bncs_aggr.aggregate(bncs)

        if not torch.any(bncs_aggr.hit_mask):
            self.buffer = self._shadeNohits(bncs_aggr)
            return()

        self.buffer[~bncs_aggr.hit_mask, :] = self._shadeNohits(bncs_aggr)
        self.buffer[bncs_aggr.hit_mask, :]  = bncs_aggr.alb[bncs_aggr.hit_mask, :]

        self._dumpBuffer(vport)

# ==============================================================================
class PathTracer(RayTracer):
    def __init__(self,
        scene,
        samples   = 100,
        max_depth = 5,
        **kwargs
    ):
        # TODO: parameter check!
        self.samples   = samples
        self.max_depth = max_depth

        super().__init__(scene, **kwargs)

    def _shadeRecursive(self, depth, rays, idx, samp_buffer):
        if depth >= self.max_depth:
            samp_buffer[idx, :] = 0
            return()

        bncs_aggr = self.trace(rays)

        if not torch.any(bncs_aggr.hit_mask):
            samp_buffer *= self._shadeNohits(bncs_aggr)
            return()

        samp_buffer[idx[~bncs_aggr.hit_mask], :] *= self._shadeNohits(bncs_aggr)
        samp_buffer[idx[bncs_aggr.hit_mask], :]  *= bncs_aggr.alb[bncs_aggr.hit_mask, :]

        rays_rand = bncs_aggr.generateRays()
        self._shadeRecursive(depth + 1, rays_rand, idx[bncs_aggr.bnc_mask], samp_buffer)        

    def render(self, vport):
        self._initBuffer(vport)        
        rays_target, eye_pos, h_norm, v_norm, h_step, v_step, aperture = self._getParams(vport)

        idx = torch.arange(len(vport), dtype = torch.long, device = self.device)
        samp_buffer = torch.ones((len(vport), 3), dtype = ftype, device = self.device)

        for sample in range(self.samples):
            r_lens = aperture / 2
            rand_h, rand_v = randInCircle(len(vport), self.device)
            orig_rand = eye_pos + (rand_h * h_norm * r_lens) + (rand_v * v_norm * r_lens)

            rand_h, rand_v = randInSquare(len(vport), self.device)
            target_rand = rays_target + (rand_h * h_step) + (rand_v * v_step)

            rays_rand = Rays(
                origins    = orig_rand,
                directions = normalize(target_rand - orig_rand, dim = 1),
                _manual    = True
            )

            samp_buffer.fill_(1)
            self._shadeRecursive(0, rays_rand, idx, samp_buffer)
            self.buffer += samp_buffer

            print(sample)

        self.buffer /= self.samples
        self._dumpBuffer(vport)