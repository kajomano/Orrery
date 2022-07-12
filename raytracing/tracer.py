import torch
from torch.nn.functional import normalize

from raytracing.rays import Rays, RayHits

from utils.settings  import ftype, n_min
from utils.common    import Resolution
from utils.torch     import DmModule

class RayTracerParams(DmModule):
    def __init__(
        self,
        sky_col     = torch.tensor([4, 19, 42],     dtype = ftype),
        horizon_col = torch.tensor([82, 131, 189],  dtype = ftype),
        ground_col  = torch.tensor([194, 212, 224], dtype = ftype)
    ):
        # TODO: parameter check!
        self.sky_col  = sky_col.view(1, 3)
        self.hori_col = horizon_col.view(1, 3)
        self.grnd_col = ground_col.view(1, 3)

        super().__init__()

class RayTracer(DmModule):
    def __init__(self, scene, params = RayTracerParams()):
        self.scene  = scene
        self.params = params

        super().__init__()

    def trace(self, rays):
        hits = RayHits(rays)
        for obj in self.scene.obj_list:
            hits *= obj.intersect(rays)

        return(hits)

    def _correctGamma(self, buffer):
        return(torch.sqrt(buffer / 255) * 255)

    def _shadeNohits(self, hits):
        rays_z = hits.rays.dir[~hits.mask, 2].view(-1, 1)

        return(torch.where(
            rays_z > 0,
            ((1 - rays_z)*self.params.hori_col + rays_z*self.params.sky_col),
            ((1 + rays_z)*self.params.hori_col - rays_z*self.params.grnd_col)
        ))
  
# Diffuse tracer ===============================================================
class DiffuseTracerParams(RayTracerParams):
    def __init__(
        self,
        light_dir = torch.tensor([1, -0.3, 0.3],  dtype = ftype),
        light_col = torch.tensor([200, 200, 200], dtype = ftype),
        ambient   = torch.tensor([0, 0, 0],    dtype = ftype),
        **kwargs
    ):
        # TODO: parameter check!
        self.light_dir = normalize(light_dir, dim = 0).view(1, 3)
        self.light_col = light_col.view(1, 3)
        self.ambient   = ambient.view(1, 3)
        
        super().__init__(**kwargs)

class DiffuseTracer(RayTracer):
    def __init__(self, scene, params = DiffuseTracerParams()):
        super().__init__(scene, params)

    def render(self, vport):
        buffer = torch.zeros((len(vport), 3), dtype = ftype, device = self.device)

        rays = Rays(
            vport.rays_orig,
            normalize(vport.rays_orig - vport.eye_pos, dim = 1),
            _manual = True
        )

        hits = self.trace(rays)

        if not torch.any(hits.mask):
            buffer = self._shadeNohits(hits)
            return()

        buffer[~hits.mask, :] = self._shadeNohits(hits)

        hits.squish()

        light_dot = torch.einsum('ij,ij->i', self.params.light_dir, hits.det[:, 3:6]).view(-1, 1)
        buffer[hits.mask, :] = torch.where(
            (light_dot > 0),
            ((1 - light_dot)*self.params.ambient + light_dot*self.params.light_col*hits.det[:, 6:9]),
            self.params.ambient
        )        

        vport.setBuffer(buffer.type(torch.uint8).view(vport.res.v, vport.res.h, 3).cpu().numpy())

# Pointlight tracer ============================================================
class PathTracerParams(RayTracerParams):
    def __init__(
        self,
        samples   = 100,
        max_depth = 5,
        ambient   = torch.tensor([0, 0, 0], dtype = ftype),
        **kwargs
    ):
        # TODO: parameter check!
        self.samples   = samples
        self.max_depth = max_depth
        self.ambient   = ambient.view(1, 3)

        super().__init__(**kwargs)

class PathTracer(RayTracer):
    def __init__(self, scene, params = PathTracerParams()):
        super().__init__(scene, params)

    def _genScatterDir(self, hits):
        fuz_size = hits.det[:, 9].view(-1, 1)
        ray_norm = -torch.einsum("ij,ij->i", hits.rays.dir[hits.mask, :], hits.det[:, 3:6]).view(-1, 1)
        ray_nout = torch.maximum(fuz_size, ray_norm)
        ray_corr = torch.sqrt((1 - torch.pow(ray_nout, 2)) / (1 - torch.pow(ray_norm, 2)))

        ray_dir  = ray_corr * (hits.rays.dir[hits.mask, :] + ray_norm * hits.det[:, 3:6]) + hits.det[:, 3:6] * ray_nout

        rand_dir = normalize(torch.randn((hits.mask.shape[0], 3), dtype = ftype, device = self.device), dim = 1)        
        rand_dir *= fuz_size - n_min # NOTE: safeguard against 0, 0, 0 ray_rand

        ray_rand = normalize(ray_dir + rand_dir, dim = 1)

        return(ray_rand)

    def _shadeRecursive(self, depth, rays, idx):
        if depth > self.params.max_depth:
            self.buffer[idx, :] = self.params.ambient
            return()

        hits = self.trace(rays)

        if not torch.any(hits.mask):
            self.buffer[idx, :] = self._shadeNohits(hits)
            return()

        self.buffer[idx[~hits.mask], :] = self._shadeNohits(hits)

        hits.squish()

        rays_rand = Rays(
            origins    = hits.det[:, 0:3],
            directions = self._genScatterDir(hits),
            _manual    = True
        )

        idx = idx[hits.mask]

        self._shadeRecursive(depth + 1, rays_rand, idx)
        self.buffer[idx, :] *= hits.det[:, 6:9] # albedo

    def render(self, vport):
        glob_buffer = torch.zeros((len(vport), 3), dtype = ftype, device = self.device)
        self.buffer = torch.zeros((len(vport), 3), dtype = ftype, device = self.device)
        idx         = torch.arange(len(vport), dtype = torch.long, device = self.device)

        for _ in range(self.params.samples):
            orig_rand = vport.rays_orig.clone()
            orig_rand += (torch.rand((len(vport), 1), dtype = ftype, device = self.device) - 0.5) * vport.h_step.view(1, 3)
            orig_rand += (torch.rand((len(vport), 1), dtype = ftype, device = self.device) - 0.5) * vport.v_step.view(1, 3)

            rays_rand = Rays(
                origins    = orig_rand,
                directions = normalize(orig_rand - vport.eye_pos.view(1, 3), dim = 1),
                _manual    = True
            )

            self._shadeRecursive(0, rays_rand, idx)
            glob_buffer += self.buffer

        glob_buffer = self._correctGamma((glob_buffer / self.params.samples))
        vport.setBuffer(glob_buffer.type(torch.uint8).view(vport.res.v, vport.res.h, 3).cpu().numpy())
