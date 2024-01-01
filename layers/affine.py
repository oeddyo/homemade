import numpy as np


class Affine:
    def __init__(self, nout=10):
        self.W = None
        self.B = None
        self.X = None
        self.nout = nout

    def forward(self, x):

        self.X = x

        # lazy initilization
        if self.W is None:
            self.W = np.random.rand(x.shape[1], self.nout)
            self.B = np.random.rand(self.nout)

        return np.dot(x, self.W) + self.B


layer = Affine(5)

print(layer.forward(np.random.rand(3, 2)))

