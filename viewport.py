from tkinter import image_types
import numpy as np

from PIL import Image

class Viewport():
    def __init__(self, res):
        self.res = res

        self.buffer = np.zeros(res.list() + [3], dtype = np.uint8)
        self.buffer[:20, :20, 0] = 255

    def render(self, res_render):
        img = Image.fromarray(self.buffer.transpose(1, 0, 2), mode = 'RGB')

        if res_render != self.res:
            img = img.resize(res_render.tuple(), Image.BILINEAR)        

        return(img)
