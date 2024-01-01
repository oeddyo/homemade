import unittest
import numpy as np
from affine import Affine


def get_loss_gradient(r, target):
    return 2 * (r - target)


class TestAffine(unittest.TestCase):
    def test_forward_shape(self):
        layer = Affine(5)

        x = np.random.rand(3, 2)
        res = layer.forward(x)

        self.assertEqual((3, 5), res.shape)
        self.assertEqual((5,), layer.B.shape)

    def test_gradient_check(self):
        layer = Affine(1)
        x = np.random.rand(3, 2)
        forward_res = layer.forward(x)

        # y is ground truth
        y = np.array([0, 1, 2]).reshape((3, 1))
        loss_gradient = get_loss_gradient(forward_res, y)
        layer.backward(loss_gradient)

        # gradient check
        h = 1e-4

        for i in range(layer.W.shape[0]):
            for j in range(layer.W.shape[1]):
                original_w = layer.W[i, j]
                layer.W[i, j] += h
                loss_h = np.sum((layer.forward(x) - y) ** 2)
                layer.W[i, j] = original_w
                loss = np.sum((layer.forward(x) - y) ** 2)
                numeric_gradient = (loss_h - loss) / h
                self.assertAlmostEqual(numeric_gradient, layer.dW[i][j], places=3)

        # now make sure the gradient for B are correct

        for i in range(layer.B.shape[0]):
            original_b = layer.B[0]
            layer.B[i] += h
            loss_h = np.sum((layer.forward(x) - y) ** 2)
            layer.B[i] = original_b
            loss = np.sum((layer.forward(x) - y) ** 2)

            numeric_gradient = (loss_h - loss) / h
            self.assertAlmostEqual(numeric_gradient, layer.dB[i], places=3)
