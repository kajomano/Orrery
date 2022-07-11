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
        light_dir   = torch.tensor([1, -0.3, 0.3],  dtype = ftype),
        light_col   = torch.tensor([120, 150, 180], dtype = ftype),
        ambient_col = torch.tensor([25, 30, 40],    dtype = ftype),
        **kwargs
    ):
        # TODO: parameter check!
        self.light_dir   = normalize(light_dir, dim = 0).view(1, 3)
        self.light_col   = light_col.view(1, 3)
        self.ambient_col = ambient_col.view(1, 3)
        
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

        light_dot = torch.einsum('ij,ij->i', self.params.light_dir, hits.det[hits.mask, 3:6])
        buffer[hits.mask, :] = torch.where(
            (light_dot > 0).view(-1, 1),
            ((1 - light_dot)*self.params.ambient_col + light_dot*self.params.light_col),
            self.params.ambient_col
        )        

        vport.setBuffer(buffer.type(torch.uint8).view(vport.res.v, vport.res.h, 3).cpu().numpy())

# Pointlight tracer ============================================================
class PathTracerParams(RayTracerParams):
    def __init__(
        self,
        samples     = 100,
        max_depth   = 5,
        ambient_col = torch.tensor([0, 0, 0], dtype = ftype),
        **kwargs
    ):
        # TODO: parameter check!
        self.samples     = samples
        self.max_depth   = max_depth
        self.ambient_col = ambient_col.view(1, 3)

        super().__init__(**kwargs)

class PathTracer(RayTracer):
    def __init__(self, scene, params = PathTracerParams()):
        super().__init__(scene, params)

    # TODO: better, simpler method for a linear transition between fuzz and shine
    def _genScatterDir(self, hits):
        n_hits   = torch.sum(hits.mask)
        rand_dir = normalize(torch.randn((n_hits, 3), dtype = ftype, device = self.device), dim = 1)
        # NOTE: safeguard against 0, 0, 0 rand_dir
        rand_dir *= 1.0 - n_min
        rand_dir += hits.det[hits.mask, 3:6]
        rand_dir = torch.where(
            torch.rand((n_hits, 1), dtype = ftype, device = self.device) < hits.det[hits.mask, 9].view(-1, 1), # fuzziness
            rand_dir,
            hits.rays.dir[hits.mask, :] - 2*torch.einsum("ij,ij->i", hits.rays.dir[hits.mask, :], hits.det[hits.mask, 3:6]).view(-1, 1)*hits.det[hits.mask, 3:6]
        )       
        rand_dir = normalize(rand_dir, dim = 1)

        return(rand_dir)

    def _shadeRecursive(self, depth, rays, idx):
        if depth > self.params.max_depth:
            self.buffer[idx, :] = self.params.ambient_col
            return()

        hits = self.trace(rays)

        if not torch.any(hits.mask):
            self.buffer[idx, :] = self._shadeNohits(hits)
            return()

        self.buffer[idx[~hits.mask], :] = self._shadeNohits(hits)

        rays_rand = Rays(
            origins    = hits.det[hits.mask, 0:3],
            directions = self._genScatterDir(hits),
            _manual    = True
        )

        idx = idx[hits.mask]

        self._shadeRecursive(depth + 1, rays_rand, idx)
        self.buffer[idx, :] *= hits.det[hits.mask, 6:9] # albedo

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
