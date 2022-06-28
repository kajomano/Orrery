class Resolution():
    def __init__(self, horizontal, vertical):
        self.h = horizontal
        self.v = vertical

    def tuple(self):
        return((self.h, self.v))

    def list(self):
        return([self.h, self.v])