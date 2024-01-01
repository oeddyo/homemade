import unittest
import numpy as np
from affine import Affine


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

    def test_update_function(self):
        layer = Affine(1)
        x = np.random.rand(3, 2)
        y = np.random.rand(3, 1)

        # Initial forward pass
        forward_res = layer.forward(x)

        # Compute the gradient of the loss
        loss_gradient = 2 * (forward_res - y)
        layer.backward(loss_gradient)

        # Store initial weights and biases
        initial_W = np.copy(layer.W)
        initial_B = np.copy(layer.B)

        # Update weights and biases
        lr = 0.01
        layer.update(lr)

        # Calculate expected new weights and biases
        expected_new_W = initial_W - lr * layer.dW
        expected_new_B = initial_B - lr * layer.dB

        # Check if the weights and biases have been updated as expected
        np.testing.assert_array_almost_equal(layer.W, expected_new_W, decimal=5)
        np.testing.assert_array_almost_equal(layer.B, expected_new_B, decimal=5)
