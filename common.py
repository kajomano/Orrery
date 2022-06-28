import time

class Resolution():
    def __init__(self, horizontal, vertical):
        self.h = horizontal
        self.v = vertical

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
        self.end = time.time()
        self.elapsed = self.end - self.start