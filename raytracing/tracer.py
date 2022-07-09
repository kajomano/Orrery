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

    def _shadeNohits(self, hits, buffer):
        rays_z = hits.rays.dir[~hits.mask, 2].view(-1, 1)

        buffer[~hits.mask, :] += torch.where(
            rays_z > 0,
            ((1 - rays_z)*self.params.hori_col.view(1, 3) + rays_z*self.params.sky_col.view(1, 3)),
            ((1 + rays_z)*self.params.hori_col.view(1, 3) - rays_z*self.params.grnd_col.view(1, 3))
        )
  
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
        buffer = torch.zeros((len(vport.rays), 3), dtype = ftype, device = self.device)

        # Trace primary rays
        hits = self.trace(vport.rays)

        # Shade background (nohits)
        self._shadeNohits(hits, buffer)

        # Diffuse shade hits
        light_dot = torch.einsum('ij,ij->i', self.params.light_dir.view(1, 3), hits.det[hits.mask, 3:6]).view(-1, 1)

        buffer[hits.mask, :] = torch.where(
            light_dot > 0,
            ((1 - light_dot)*self.params.ambient_col.view(1, 3) + light_dot*self.params.light_col.view(1, 3)),
            self.params.ambient_col.view(1, 3)
        )

        buffer = buffer.type(torch.uint8)
        vport.setBuffer(buffer.view(vport.res.v, vport.res.h, 3).cpu().numpy())

# Pointlight tracer ============================================================
class PathTracerParams(RayTracerParams):
    def __init__(
        self,
        samples     = 100,
        # TODO: remove these
        light_dir   = torch.tensor([1, -0.3, 0.3],  dtype = ftype),
        light_col   = torch.tensor([120, 150, 180], dtype = torch.uint8),
        ambient_col = torch.tensor([25, 30, 40],    dtype = torch.uint8),
        **kwargs
    ):
        # TODO: remove these
        self.light_dir   = normalize(light_dir, dim = 0)
        self.light_col   = light_col

        # TODO: parameter check!
        self.samples     = samples    
        self.ambient_col = ambient_col

        super().__init__(**kwargs)

class PathTracer(RayTracer):
    def __init__(self, scene, params = PathTracerParams()):
        super().__init__(scene, params)

    def render(self, vport):
        buffer = torch.zeros((len(vport.rays), 3), dtype = ftype, device = self.device)

        for _ in range(self.params.samples):
            rays_orig = vport.rays.orig
            rays_orig += (torch.rand((len(vport.rays), 1), dtype = ftype, device = self.device) - 1.0) * vport.pix_width * vport.h_norm.view(1, 3)
            rays_orig += (torch.rand((len(vport.rays), 1), dtype = ftype, device = self.device) - 1.0) * vport.pix_height * vport.v_norm.view(1, 3)

            rays_rand = Rays(
                origins    = rays_orig,
                directions = vport.rays.dir
            )

            hits = self.trace(rays_rand)
            self._shadeNohits(hits, buffer)

            # Diffuse shade hits
            light_dot = torch.einsum('ij,ij->i', self.params.light_dir.view(1, 3), hits.det[hits.mask, 3:6]).view(-1, 1)

            buffer[hits.mask, :3] += torch.where(
                light_dot > 0,
                ((1 - light_dot)*self.params.ambient_col.view(1, 3) + light_dot*self.params.light_col.view(1, 3)),
                self.params.ambient_col.view(1, 3)
            )

        buffer = (buffer / self.params.samples).type(torch.uint8)
        vport.setBuffer(buffer.view(vport.res.v, vport.res.h, 3).cpu().numpy())

    # def renderTiles(self, tile_size = Resolution(100)):
    #     tiles_h = self.vport.res.h // tile_size.h
    #     tiles_h += 1 if self.vport.res.h % tile_size.h else 0

    #     tiles_v = self.vport.res.v // tile_size.v
    #     tiles_v += 1 if self.vport.res.v % tile_size.v else 0

    #     start_v = 0
    #     for _ in range(tiles_v):
    #         start_h = 0
    #         for _ in range(tiles_h):
    #             end_v = min(self.vport.res.v, start_v + tile_size.v)
    #             end_h = min(self.vport.res.h, start_h + tile_size.h)

    #             slice_v = slice(start_v, end_v)
    #             slice_h = slice(start_h, end_h)

    #             self.shade(self.vport.rays[slice_h, slice_v].view([-1]))

    #             start_h += tile_size.h
    #         start_v += tile_size.v