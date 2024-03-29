import torch
from torch.nn.functional import normalize

from utils.torch  import DmModule, ftype
from utils.common import Timer

class RayTracer(DmModule):
    def __init__(self, 
        col_sky     = torch.tensor([8, 22, 38],  dtype = ftype),
        col_horizon = torch.tensor([35, 58, 84], dtype = ftype),
        col_ground  = torch.tensor([21, 28, 36], dtype = ftype),
        # col_sky     = torch.tensor([93, 156, 222],  dtype = ftype),
        # col_horizon = torch.tensor([220, 226, 232], dtype = ftype),
        # col_ground  = torch.tensor([220, 226, 232], dtype = ftype),
        **kwargs
    ):
        # TODO: parameter check!
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
        self.buffer = torch.sqrt(self.buffer / 255) * 255 # Gamma correction
        self.buffer = torch.clamp_max(self.buffer, 255)   # Basic HDR to LDR conversion

        vport.getBuffer().copy_(self.buffer.type(torch.uint8).view(vport.res.v, vport.res.h, 3).cpu())
  
# ==============================================================================
class SimpleTracer(RayTracer):
    def __init__(self,
        dir_light = torch.tensor([1, 0, 1],       dtype = ftype),
        col_light = torch.tensor([200, 200, 200], dtype = ftype),
        col_ambnt = torch.tensor([50, 50, 50],    dtype = ftype),
        **kwargs
    ):
        # TODO: parameter check!
        self.dir_light = normalize(dir_light, dim = 0).view(1, 3)
        self.col_light = col_light
        self.col_ambnt = col_ambnt

        super().__init__(**kwargs)

    def render(self, scene, vport):
        scene.to(self.device)
        vport.to(self.device)

        self._initBuffer(vport)

        bncs_aggr = scene.traverse(vport.getRays(rand = False), self)

        if not torch.any(bncs_aggr.hit_mask):
            self.buffer = self._shadeNohits(bncs_aggr)
            return()

        self.buffer[~bncs_aggr.hit_mask, :] = self._shadeNohits(bncs_aggr)
        self.buffer[bncs_aggr.hit_mask, :]  = bncs_aggr.alb[bncs_aggr.hit_mask, :]

        self._dumpBuffer(vport)

# ==============================================================================
class PathTracer(RayTracer):
    def __init__(self,
        samples   = 100,
        max_depth = 5,
        **kwargs
    ):
        # TODO: parameter check!
        self.samples   = samples
        self.max_depth = max_depth

        super().__init__(**kwargs)

    def _shadeRecursive(self, scene, depth, rays, pix_ids, samp_buffer):
        if depth >= self.max_depth:
            samp_buffer[pix_ids, :] = 0
            return(len(rays))

        bncs_aggr = scene.traverse(rays)

        if not torch.any(bncs_aggr.hit_mask):
            samp_buffer *= self._shadeNohits(bncs_aggr)
            return(len(rays))

        samp_buffer[pix_ids[~bncs_aggr.hit_mask], :] *= self._shadeNohits(bncs_aggr)
        samp_buffer[pix_ids[bncs_aggr.hit_mask], :]  *= bncs_aggr.alb[bncs_aggr.hit_mask, :]

        rays_rand = bncs_aggr.generateRays()
        n_rays = self._shadeRecursive(scene, depth + 1, rays_rand, pix_ids[bncs_aggr.bnc_mask], samp_buffer)

        return(len(rays) + n_rays) 

    def render(self, scene, vport):
        scene.to(self.device)
        vport.to(self.device)

        self._initBuffer(vport)        

        pix_ids     = torch.arange(len(vport), dtype = torch.long, device = self.device)
        samp_buffer = torch.ones((len(vport), 3), dtype = ftype, device = self.device)

        for sample in range(self.samples):
            with Timer() as t:
                rays = vport.getRays()

                samp_buffer.fill_(1)
                n_rays = self._shadeRecursive(scene, 0, rays, pix_ids, samp_buffer)
                self.buffer += samp_buffer

            print(f'{(sample + 1):04}', f'{(n_rays / t.elap / 1e6):.04} MR/s', t, sep = " - ")

        self.buffer /= self.samples
        self._dumpBuffer(vport)