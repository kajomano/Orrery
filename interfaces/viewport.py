import numpy as np

from copy            import deepcopy
from multiprocessing import Lock, shared_memory

class ViewportParams():
    def __init__(
        self, 
        height       = 2.0,
        aspect_ratio = 16/9,
        focal_len    = 2.0, 
        eye_pos      = np.array([0.0, -10.0, 2.0], dtype = np.single),
        view_target  = np.array([0.0, -2.0, 0.0],  dtype = np.single)
        # eye_pos      = np.array([0.0, -10.0, 0.0], dtype = np.single),
        # view_target  = np.array([0.0, -2.0, 0.0],  dtype = np.single)
    ):
        # TODO: params check!
        self.width       = height * aspect_ratio
        self.height      = height
        self.focal       = focal_len
        self.eye_pos     = eye_pos.reshape(1, 3)
        self.view_target = view_target.reshape(1, 3)

class Viewport():
    def __init__(self, res, params = ViewportParams()):
        self.res = res

        self.buffer_lock = Lock()

        buff_tmp = np.zeros([res.v, res.h, 3], dtype = np.uint8)
        self.buff_shm = shared_memory.SharedMemory(create = True, size = buff_tmp.nbytes)
        buff_ptr = np.ndarray(buff_tmp.shape, dtype = np.uint8, buffer = self.buff_shm.buf)
        buff_ptr[:] = buff_tmp[:]
        self.buff_ptr = None

        self.main = True

        self.setParams(params)

    def setParams(self, params):
        self.params  = deepcopy(params)

        self.eye_pos = params.eye_pos
        view_dir  = (params.view_target - params.eye_pos).reshape(-1)
        view_dir /= np.linalg.norm(view_dir, axis = 0)

        h_norm  = np.cross(view_dir, np.array([0.0, 0.0, 1.0], dtype = np.single)) # NOTE: This presumes up will always be up
        h_norm /= np.linalg.norm(h_norm, axis = 0)
        v_norm  = np.cross(view_dir, h_norm)
        v_norm /= np.linalg.norm(v_norm, axis = 0)

        left_top = (params.eye_pos + params.focal * view_dir) - (params.width / 2 * h_norm) - (params.height / 2 * v_norm)
        left_top = left_top.reshape(1, 1, 3)

        self.h_step = (params.width / (self.res.h - 1) * h_norm)
        self.v_step = (params.height / (self.res.v - 1) * v_norm)

        h_offset  = self.h_step.reshape(1, 1, 3) * np.arange(self.res.h, dtype = np.single).reshape(1, self.res.h, 1)
        v_offset  = self.v_step.reshape(1, 1, 3) * np.arange(self.res.v, dtype = np.single).reshape(self.res.v, 1, 1)            

        self.rays_orig = (left_top + h_offset + v_offset).reshape(-1, 3)

    def __len__(self):
        return(self.rays_orig.shape[0])

    def getBuffer(self):
        if self.buff_ptr is None:
            self.buff_ptr = np.ndarray([self.res.v, self.res.h, 3], dtype = np.uint8, buffer = self.buff_shm.buf)
            self.main = False
        return(self.buff_ptr)

    # def getBufferWithCrosshair(self, off_h = 0, off_v = 0):
    #     mid_h = self.res.h // 2 + off_h
    #     mid_v = self.res.v // 2 - off_v      

    #     self.buffer[(mid_v - 1):(mid_v + 2), (mid_h - 5):(mid_h + 6), :] = 255
    #     self.buffer[(mid_v - 5):(mid_v + 6), (mid_h - 1):(mid_h + 2), :] = 255

    #     self.buffer[mid_v, (mid_h - 4):(mid_h + 5), :] = 0
    #     self.buffer[(mid_v - 4):(mid_v + 5), mid_h, :] = 0

    #     ray_orig = self.rays.orig.view(self.res.v, self.res.h, 3)[mid_v, mid_h, :]
    #     ray_dir  = self.rays.dir.view(self.res.v, self.res.h, 3)[mid_v, mid_h, :]

    #     print(f'origin: {ray_orig[0]:.4f}, {ray_orig[1]:.4f}, {ray_orig[2]:.4f}')
    #     print(f'direc.: {ray_dir[0]:.4f}, {ray_dir[1]:.4f}, {ray_dir[2]:.4f}')

    #     return(self.buffer)

    # TODO: clean up shm
    # def __del__(self):
    #     self.buff_shm.close()

    #     if current_process().name == 'MainProcess':
    #         print("cleaned main")
    #         self.buff_shm.unlink()
    #     else:
    #         print("cleaned off")

