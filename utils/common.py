import time

class Resolution():
    def __init__(self, vertical_res, aspect_ratio = 16/9):
        self.v = vertical_res
        self.h = int(round(vertical_res * aspect_ratio))       

    def __eq__(self, other):
        return((self.h == other.h) & (self.v == other.v))

    def __iter__(self):
        yield self.h
        yield self.v

class Timer:    
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end  = time.time()
        self.elap = self.end - self.start

    def __truediv__(self, other):
        self.elap /= other
        return(self)

    def __str__(self):
        if self.elap > 1:
            return(f'{self.elap:.3f} seconds')
        elif self.elap > 0.001:
            return(f'{self.elap * 1000:.3f} milliseconds')
        else:
            return(f'{self.elap * 1000000:.0f} microseconds')