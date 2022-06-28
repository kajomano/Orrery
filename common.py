import time

class Resolution():
    def __init__(self, vertical_res, aspect_ratio = 16/9):
        self.h_res = int(round(vertical_res * aspect_ratio))
        self.v_res = vertical_res

    def __eq__(self, other):
        return((self.h_res == other.h_res) & (self.v_res == other.v_res))

    def __iter__(self):
        yield self.h_res
        yield self.v_res

class Timer:    
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.elapsed = self.end - self.start