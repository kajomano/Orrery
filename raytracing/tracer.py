import numpy as np

from utils.linalg import *
from utils.common import Resolution

class RayTracerParams():
    def __init__(
        self,
        sky_col     = float([[28, 20, 97]]),
        horizon_col = float([[249, 177, 92]]),
        ground_col  = float([[41, 16, 45]])      
    ):
        self.sky_col  = sky_col,
        self.hori_col = horizon_col,
        self.grnd_col = ground_col    

class RayTracer():
    def __init__(self, scene, viewport, params = RayTracerParams()):
        self.scene  = scene
        self.vport  = viewport
        self.params = params

    def trace(self, rays):
        hits = self.scene.spheres.intersect(rays)

        return(hits)

    def render(self):
        self.vport.buffer = np.reshape(
            self.shade(self.vport.rays.view([-1])), 
            [self.vport.res.v, self.vport.res.h, 3]
        )

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

    def _shadeNohits(self, rays, ray_hits, buffer):
        rays_y = rays.dir[~ray_hits.hit_mask, 2]

        print(rays.shape[0])
        print(np.sum(ray_hits.hit_mask))
        exit()

        buffer[~ray_hits.hit_mask, :] = np.where(
            rays_y > 0,
            (1 - rays_y)*self.params.hori_col + rays_y*self.params.sky_col,
            self.params.grnd_col
        )
  

class PhongTracerParams(RayTracerParams):
    def __init__(
        self,
        light_dir   = float([[-1, 0, 0]]),
        light_col   = np.array([[100, 220, 120]], dtype = np.uint8),
        ambient_col = np.array([[50, 80, 120]], dtype = np.uint8),
        *args,
        **kwargs
    ):
        super().__init__(args, kwargs)
        self.light_dir   = light_dir,
        self.light_col   = light_col,
        self.ambient_col = ambient_col

class PhongTracer(RayTracer):
    def __init__(self, scene, viewport, params = PhongTracerParams()):
        super().__init__(scene, viewport, params)

    def shade(self, rays):
        buffer = np.zeros(rays.shape, dtype = np.uint8)

        ray_hits = self.trace(rays)
        self._shadeNohits(rays, ray_hits, buffer)

        return(buffer)