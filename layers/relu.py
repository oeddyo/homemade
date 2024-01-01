class Relu:
    def __init__(self):
        self.masks = None

    def forward(self, x):
        self.masks = x < 0

        out = x.copy()
        out[self.masks] = 0
        return out

    def backward(self, dout):
        dout[self.masks] = 0
        dx = dout
        return dx

    def update(self, lr):
        # do nothing
        pass
