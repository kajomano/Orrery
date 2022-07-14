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
        self.scene    = scene
        self.col_sky  = col_sky
        self.col_hrzn = col_horizon
        self.col_grnd = col_ground

        super().__init__(**kwargs)

    def _correctGamma(self, buffer):
        return(torch.sqrt(buffer / 255) * 255)

    def _shadeNohits(self, bncs_aggr):
        rays_z = bncs_aggr.rays.dirs[~bncs_aggr.hit_mask, 2].view(-1, 1)

        return(torch.where(
            rays_z > 0,
            ((1 - rays_z)*self.col_hrzn + rays_z*self.col_sky),
            ((1 + rays_z)*self.col_hrzn - rays_z*self.col_grnd)
        ))
  
# ==============================================================================
class DiffuseTracer(RayTracer):
    def __init__(self, 
        scene, 
        light_dir = torch.tensor([1, 0, 1],      dtype = ftype),
        light_col = torch.tensor([200, 200, 200], dtype = ftype),
        **kwargs
    ):
        self.light_dir = normalize(light_dir, dim = 0).view(1, 3)
        self.light_col = light_col

        super().__init__(scene, **kwargs)

    def render(self, vport):
        buffer    = torch.zeros((len(vport), 3), dtype = ftype, device = self.device)
        rays_orig = torch.tensor(vport.rays_orig, dtype = ftype, device = self.device)
        eye_pos   = torch.tensor(vport.eye_pos, dtype = ftype, device = self.device)

        rays = Rays(
            rays_orig,
            normalize(rays_orig - eye_pos, dim = 1),
            _manual = True
        )

        bncs_aggr = RayBounceAggr(rays)

        for obj in self.scene.obj_list:
            hits = obj.intersect(rays)

            if(hits is not None):
                bncs = obj.bounceTo(hits, self.light_dir)
                bncs_aggr.aggregate(bncs)

        shadow_rays = bncs_aggr.generateRays()
        shadow = torch.zeros((len(shadow_rays),), dtype = torch.bool, device = self.device)
        for obj in self.scene.obj_list:
            hits = obj.intersect(shadow_rays)
            if hits is not None:
                shadow = torch.logical_or(hits.hit_mask, shadow)

        bncs_aggr.alb[bncs_aggr.bnc_mask, :] *= ~shadow.view(-1, 1)

        buffer[~bncs_aggr.hit_mask, :] = self._shadeNohits(bncs_aggr)
        buffer[bncs_aggr.hit_mask, :]  = bncs_aggr.alb[bncs_aggr.hit_mask, :] * self.light_col

        vport_buffer = vport.getBuffer()
        vport_buffer[:] = buffer.type(torch.uint8).view(vport.res.v, vport.res.h, 3).cpu().numpy()[:]

# # ==============================================================================
# class PathTracerParams(RayTracerParams):
#     def __init__(
#         self,
#         samples   = 100,
#         max_depth = 5,
#         ambient   = torch.tensor([0, 0, 0], dtype = ftype),
#         **kwargs
#     ):
#         # TODO: parameter check!
#         self.samples   = samples
#         self.max_depth = max_depth
#         self.ambient   = ambient.view(1, 3)

#         super().__init__(**kwargs)

# class PathTracer(RayTracer):
#     def __init__(self, scene, params = PathTracerParams()):
#         super().__init__(scene, params)

#     def _genScatterDir(self, hits):
#         fuz_size = hits.det[:, 9].view(-1, 1)
#         ray_norm = -torch.einsum("ij,ij->i", hits.rays.dir[hits.mask, :], hits.det[:, 3:6]).view(-1, 1)
#         ray_nout = torch.maximum(fuz_size, ray_norm)
#         ray_corr = torch.sqrt((1 - torch.pow(ray_nout, 2)) / ((1 + eps) - torch.pow(ray_norm, 2)))

#         ray_dir  = ray_corr * (hits.rays.dir[hits.mask, :] + ray_norm * hits.det[:, 3:6]) + hits.det[:, 3:6] * ray_nout

#         rand_dir = normalize(torch.randn((hits.mask.shape[0], 3), dtype = ftype, device = self.device), dim = 1)        
#         rand_dir *= fuz_size - eps # NOTE: safeguard against 0, 0, 0 ray_rand

#         ray_rand = normalize(ray_dir + rand_dir, dim = 1)

#         return(ray_rand)

#     def _shadeRecursive(self, depth, rays, idx):
#         if depth > self.params.max_depth:
#             self.buffer[idx, :] = self.params.ambient
#             return()

#         hits = self.trace(rays)

#         if not torch.any(hits.mask):
#             self.buffer[idx, :] = self._shadeNohits(hits)
#             return()

#         self.buffer[idx[~hits.mask], :] = self._shadeNohits(hits)

#         hits.squish()

#         rays_rand = Rays(
#             origins    = hits.det[:, 0:3],
#             directions = self._genScatterDir(hits),
#             _manual    = True
#         )

#         idx = idx[hits.mask]

#         self._shadeRecursive(depth + 1, rays_rand, idx)
#         self.buffer[idx, :] *= hits.det[:, 6:9] # albedo

#     def render(self, vport):
#         glob_buffer  = torch.zeros((len(vport), 3), dtype = ftype, device = self.device)
#         self.buffer  = torch.zeros((len(vport), 3), dtype = ftype, device = self.device)
#         vport_buffer = vport.getBuffer()
#         idx          = torch.arange(len(vport), dtype = torch.long, device = self.device)

#         rays_orig    = torch.tensor(vport.rays_orig, dtype = ftype, device = self.device)
#         eye_pos      = torch.tensor(vport.eye_pos, dtype = ftype, device = self.device)

#         h_step       = torch.tensor(vport.h_step, dtype = ftype, device = self.device)
#         v_step       = torch.tensor(vport.v_step, dtype = ftype, device = self.device)

#         for sample in range(self.params.samples):
#             orig_rand = rays_orig.clone()
#             orig_rand += (torch.rand((len(vport), 1), dtype = ftype, device = self.device) - 0.5) * h_step.view(1, 3)
#             orig_rand += (torch.rand((len(vport), 1), dtype = ftype, device = self.device) - 0.5) * v_step.view(1, 3)

#             rays_rand = Rays(
#                 origins    = orig_rand,
#                 directions = normalize(orig_rand - eye_pos.view(1, 3), dim = 1),
#                 _manual    = True
#             )

#             self._shadeRecursive(0, rays_rand, idx)
#             glob_buffer += self.buffer
#             print(sample)

#         vport_buffer[:] = self._correctGamma(glob_buffer / self.params.samples).type(torch.uint8).view(vport.res.v, vport.res.h, 3).cpu().numpy()[:]