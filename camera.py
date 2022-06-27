import numpy as np

from PIL         import Image
from PIL.ImageQt import ImageQt

class Camera():
    def __init__(self, h_res, v_res):
        self.h_res = h_res
        self.v_res = v_res

        self.buffer = np.zeros([v_res, h_res, 3], dtype = np.uint8)

    def render(self):
        return(ImageQt(Image.fromarray(self.buffer, mode = 'RGB')))

