import torch
from torch.nn.functional import normalize

from utils.settings import ftype
from utils.common   import Resolution
from utils.torch    import DmModule

class RayTracerParams(DmModule):
    def __init__(
        self,
        sky_col     = torch.tensor([28, 20, 97],   dtype = torch.uint8),
        horizon_col = torch.tensor([249, 177, 92], dtype = torch.uint8),
        ground_col  = torch.tensor([16, 19, 45],   dtype = torch.uint8) 
    ):
        # TODO: parameter check!

        self.sky_col  = sky_col
        self.hori_col = horizon_col
        self.grnd_col = ground_col

        super().__init__()

class RayTracer(DmModule):
    def __init__(self, scene, viewport, params = RayTracerParams()):
        self.scene  = scene
        self.vport  = viewport
        self.params = params

        super().__init__()

    def trace(self, rays):
        hits = self.scene.spheres.intersect(rays)

        return(hits)

    def render(self):
        self.vport.buffer = self._shade(self.vport.rays).view(self.vport.res.v, self.vport.res.h, 3).cpu().numpy()

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

    def _shadeNohits(self, hits, buffer):
        rays_y = hits.rays.dir[~hits.mask, 2].view(-1, 1)

        buffer[~hits.mask, :] = torch.where(
            rays_y > 0,
            ((1 - rays_y)*self.params.hori_col.view(1, 3) + rays_y*self.params.sky_col.view(1, 3)),
            self.params.grnd_col.view(1, 3)
        ).type(torch.uint8)
  

class DiffuseTracerParams(RayTracerParams):
    def __init__(
        self,
        light_dir   = torch.tensor([1, -0.3, 0.3],  dtype = ftype),
        light_col   = torch.tensor([100, 220, 120], dtype = torch.uint8),
        ambient_col = torch.tensor([50, 80, 120],   dtype = torch.uint8),
        *args,
        **kwargs
    ):
        # TODO: parameter check!
        if len(args) or len(kwargs):
            super().__init__(args, kwargs)
        else:
            super().__init__() 

        self.light_dir   = normalize(light_dir, dim = 0)
        self.light_col   = light_col
        self.ambient_col = ambient_col

class DiffuseTracer(RayTracer):
    def __init__(self, scene, viewport, params = DiffuseTracerParams()):
        super().__init__(scene, viewport, params)

    def _shade(self, rays):
        buffer = torch.zeros((len(rays), 3), dtype = torch.uint8, device = self.device)

        # Trace primary rays
        hits = self.trace(rays)

        # Shade background (nohits)
        self._shadeNohits(hits, buffer)

        # Diffuse shade hits
        light_dot = torch.einsum('ij,ij->i', self.params.light_dir.view(1, 3), hits.details[hits.mask, 3:]).view(-1, 1)

        buffer[hits.mask, :] = torch.where(
            light_dot > 0,
            ((1 - light_dot)*self.params.ambient_col.view(1, 3) + light_dot*self.params.light_col.view(1, 3)),
            self.params.ambient_col.view(1, 3)
        ).type(torch.uint8)

        return(buffer)