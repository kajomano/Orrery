import numpy as np

from PIL import Image

from linalg import norm, cross

class ViewportParams():
    def __init__(
        self, 
        height       = 2.0,
        aspect_ratio = 4/3,
        focal_len    = 1.0, 
        eye_pos      = np.array([0.0, -5.0, 0.0], dtype = np.float32),
        view_target  = np.array([0.0, 0.0, 0.0], dtype = np.float32)
    ):
        self.width       = height * aspect_ratio
        self.height      = height
        self.focal       = focal_len
        self.eye_pos     = eye_pos
        self.view_target = view_target

class Viewport():
    def __init__(self, res, params = ViewportParams()):
        self.res = res

        # NOTE: This presumes up will always be up
        view_dir = norm(params.view_target - params.eye_pos)
        h_norm  = norm(cross(
            view_dir,
            np.array([0.0, 0.0, 1.0], dtype = np.float32)
        ))
        v_norm  = norm(cross(h_norm, view_dir))
        
        print(h_norm)
        print(v_norm)

        self.buffer = np.zeros(list(res) + [3], dtype = np.uint8)
        self.buffer[:20, :20, 0] = 255

    def render(self, res_render):
        img = Image.fromarray(self.buffer.transpose(1, 0, 2), mode = 'RGB')

        if res_render != self.res:
            img = img.resize(tuple(res_render), Image.BILINEAR)        

        return(img)
