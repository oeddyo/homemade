import numpy as np


class Affine:
    def __init__(self, nout=10):
        self.W = None
        self.B = None
        self.X = None

        self.dW = None
        self.dB = None
        self.dX = None

        self.nout = nout

    def forward(self, x):
        self.X = x

        # lazy initialization
        if self.W is None:
            self.W = np.random.rand(x.shape[1], self.nout)
            self.B = np.random.rand(self.nout)

        return np.dot(x, self.W) + self.B

    def backward(self, dz):

        self.dB = np.sum(dz, axis=0)

        # compute dW
        self.dW = np.dot(self.X.T, dz)
        self.dX = np.dot(dz, self.W.T)

        return self.dX

    def update(self, lr=0.01):
        self.B -= lr * self.dB
        self.W -= lr * self.dW
