import numpy as np


class AffineLayer:
    def __init__(self, nin=10, nout=10):
        self.w = np.random.rand(nout, nin)
        self.b = np.random.rand(nout, 1)
        self.x = None

    # x has to be (nin, 1)
    def forward(self, x):
        self.x = x
        return np.dot(self.w, x) + self.b

    # dy's shape is (nout, 1)
    def backward(self, dy):
        dw = np.dot(dy, self.x.T)
        dx =


        self.db = dy




layer = AffineLayer(nin=3, nout=5)

print(layer.forward(np.random.rand(3, 1)))
