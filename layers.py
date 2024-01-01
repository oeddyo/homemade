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
            nin = x.shape[1]
            # Xavier initialization
            std = np.sqrt(2.0 / (nin + self.nout))
            self.W = np.random.randn(nin, self.nout) * std
            # Zero initialization for biases
            self.B = np.zeros(self.nout)

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


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = sigmoid(x)
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out

        return dx

    def update(self, lr):
        pass