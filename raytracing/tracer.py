import torch
from torch.nn.functional import normalize

from raytracing.rays import Rays, RayHits

from utils.settings  import ftype
from utils.common    import Resolution
from utils.torch     import DmModule

class RayTracerParams(DmModule):
    def __init__(
        self,
        sky_col     = torch.tensor([4, 19, 42],     dtype = torch.uint8),
        horizon_col = torch.tensor([82, 131, 189],  dtype = torch.uint8),
        ground_col  = torch.tensor([194, 212, 224], dtype = torch.uint8)
    ):
        # TODO: parameter check!
        self.sky_col  = sky_col
        self.hori_col = horizon_col
        self.grnd_col = ground_col

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

    def _shadeNohits(self, hits):
        rays_z = hits.rays.dir[~hits.mask, 2].view(-1, 1)

        return(torch.where(
            rays_z > 0,
            ((1 - rays_z)*self.params.hori_col.view(1, 3) + rays_z*self.params.sky_col.view(1, 3)),
            ((1 + rays_z)*self.params.hori_col.view(1, 3) - rays_z*self.params.grnd_col.view(1, 3))
        ))
  
# Diffuse tracer ===============================================================
class DiffuseTracerParams(RayTracerParams):
    def __init__(
        self,
        light_dir   = torch.tensor([1, -0.3, 0.3],  dtype = ftype),
        light_col   = torch.tensor([120, 150, 180], dtype = torch.uint8),
        ambient_col = torch.tensor([25, 30, 40],    dtype = torch.uint8),
        **kwargs
    ):
        # TODO: parameter check!
        self.light_dir   = normalize(light_dir, dim = 0)
        self.light_col   = light_col
        self.ambient_col = ambient_col
        
        super().__init__(**kwargs)

class DiffuseTracer(RayTracer):
    def __init__(self, scene, params = DiffuseTracerParams()):
        super().__init__(scene, params)

    def render(self, vport):
        buffer = torch.zeros((len(vport), 3), dtype = ftype, device = self.device)

        # Trace primary rays
        rays = Rays(
            vport.rays_orig,
            normalize(vport.rays_orig - vport.eye_pos.view(1, 3), dim = 1),
            _manual = True
        )

        hits = self.trace(rays)

        # Diffuse shade hits
        light_dot = torch.einsum('ij,ij->i', self.params.light_dir.view(1, 3), hits.det[hits.mask, 3:6]).view(-1, 1)

        buffer[hits.mask, :] = torch.where(
            light_dot > 0,
            ((1 - light_dot)*self.params.ambient_col.view(1, 3) + light_dot*self.params.light_col.view(1, 3)),
            self.params.ambient_col.view(1, 3)
        )

        # Shade nohits
        buffer[~hits.mask, :] += self._shadeNohits(hits)

        buffer = buffer.type(torch.uint8)
        vport.setBuffer(buffer.view(vport.res.v, vport.res.h, 3).cpu().numpy())

# Pointlight tracer ============================================================
class PathTracerParams(RayTracerParams):
    def __init__(
        self,
        samples     = 100,
        max_depth   = 5,
        ambient_col = torch.tensor([25, 30, 40], dtype = torch.uint8),
        **kwargs
    ):
        # TODO: parameter check!
        self.samples     = samples
        self.max_depth   = max_depth
        self.ambient_col = ambient_col

        super().__init__(**kwargs)

class PathTracer(RayTracer):
    def __init__(self, scene, params = PathTracerParams()):
        super().__init__(scene, params)

    # TODO: try if the other generation method is faster
    def _genRandOnSphere(self, hits):
        return(normalize(torch.rand((torch.sum(hits.mask), 3), dtype = ftype, device = self.device), dim = 1))

    def _shadeRecursive(self, rays, buffer, depth):
        if depth >= self.params.max_depth:
            return(self.params.ambient_col.view(1, 3))

        # Trace current rays
        hits = self.trace(rays)

        # Generate random new rays
        rays_rand = Rays(
            origins    = hits.det[hits.mask, 0:3],
            directions = normalize(hits.det[hits.mask, 3:6] + self._genRandOnSphere(hits), dim = 1),
            _manual    = True
        )
        

    def render(self, vport):
        buf_glob = torch.zeros((len(vport), 3), dtype = ftype, device = self.device)

        for _ in range(self.params.samples):
            orig_rand = vport.rays_orig.clone()
            orig_rand += (torch.rand((len(vport), 1), dtype = ftype, device = self.device) - 0.5) * vport.h_step.view(1, 3)
            orig_rand += (torch.rand((len(vport), 1), dtype = ftype, device = self.device) - 0.5) * vport.v_step.view(1, 3)

            rays_rand = Rays(
                origins    = orig_rand,
                directions = normalize(orig_rand - vport.eye_pos.view(1, 3), dim = 1),
                _manual    = True
            )

            buf_samp = torch.zeros((len(vport), 3), dtype = ftype, device = self.device)
            self._shadeRecursive(rays_rand, buf_samp, 0)

            # # Shade nohits
            # buffer[~hits.mask, :] += self._shadeNohits(hits)

        buf_glob = (buf_glob / self.params.samples).type(torch.uint8)
        vport.setBuffer(buf_glob.view(vport.res.v, vport.res.h, 3).cpu().numpy())
