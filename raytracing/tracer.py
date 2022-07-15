import torch
from torch.nn.functional import normalize

from raytracing.rays import Rays, RayBounceAggr #RayHitsAggr

from utils.consts    import ftype, eps
from utils.common    import Resolution
from utils.torch     import DmModule

class RayTracer(DmModule):
    def __init__(self, 
        scene, 
        col_sky     = torch.tensor([4, 19, 42],     dtype = ftype),
        col_horizon = torch.tensor([82, 131, 189],  dtype = ftype),
        col_ground  = torch.tensor([194, 212, 224], dtype = ftype),
        **kwargs
    ):
        # TODO: parameter check!
        self.scene     = scene
        self.col_sky   = col_sky
        self.col_hrzn  = col_horizon
        self.col_grnd  = col_ground

        self.buffer = None

        super().__init__(**kwargs)

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
        self.buffer = torch.sqrt(self.buffer / 255) * 255
        self.buffer = torch.clamp_max(self.buffer, 255)

        vport.getBuffer()[:] = self.buffer.type(torch.uint8).view(vport.res.v, vport.res.h, 3).cpu().numpy()[:]

    def _getParams(self, vport):
        rays_orig = torch.tensor(vport.rays_orig, dtype = ftype, device = self.device)
        eye_pos   = torch.tensor(vport.eye_pos, dtype = ftype, device = self.device)

        h_step    = torch.tensor(vport.h_step, dtype = ftype, device = self.device)
        v_step    = torch.tensor(vport.v_step, dtype = ftype, device = self.device)

        return(rays_orig, eye_pos, h_step, v_step)
  
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
        rays_orig, eye_pos, *_ = self._getParams(vport)

        rays = Rays(
            rays_orig,
            normalize(rays_orig - eye_pos, dim = 1),
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

        bncs_aggr = RayBounceAggr(rays)

        for obj in self.scene.obj_list:
            hits = obj.intersect(rays)

            if(hits is not None):
                bncs = obj.bounce(hits)
                bncs_aggr.aggregate(bncs)

        if not torch.any(bncs_aggr.hit_mask):
            # self.buffer *= self._shadeNohits(bncs_aggr)
            samp_buffer *= self._shadeNohits(bncs_aggr)
            return()

        # self.buffer[idx[~bncs_aggr.hit_mask], :] *= self._shadeNohits(bncs_aggr)
        # self.buffer[idx[bncs_aggr.hit_mask], :]  *= bncs_aggr.alb[bncs_aggr.hit_mask, :]

        samp_buffer[idx[~bncs_aggr.hit_mask], :] *= self._shadeNohits(bncs_aggr)
        samp_buffer[idx[bncs_aggr.hit_mask], :]  *= bncs_aggr.alb[bncs_aggr.hit_mask, :]

        rays_rand = bncs_aggr.generateRays()
        self._shadeRecursive(depth + 1, rays_rand, idx[bncs_aggr.bnc_mask], samp_buffer)        

    def render(self, vport):
        self._initBuffer(vport)        
        rays_orig, eye_pos, h_step, v_step = self._getParams(vport)

        idx = torch.arange(len(vport), dtype = torch.long, device = self.device)
        samp_buffer = torch.ones((len(vport), 3), dtype = ftype, device = self.device)

        for sample in range(self.samples):
            orig_rand = rays_orig.clone()
            orig_rand += (torch.rand((len(vport), 1), dtype = ftype, device = self.device) - 0.5) * h_step.view(1, 3)
            orig_rand += (torch.rand((len(vport), 1), dtype = ftype, device = self.device) - 0.5) * v_step.view(1, 3)

            rays_rand = Rays(
                origins    = orig_rand,
                directions = normalize(orig_rand - eye_pos.view(1, 3), dim = 1),
                _manual    = True
            )

            samp_buffer.fill_(1)
            self._shadeRecursive(0, rays_rand, idx, samp_buffer)
            self.buffer += samp_buffer

            print(sample)

        self.buffer /= self.samples
        self._dumpBuffer(vport)