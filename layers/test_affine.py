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
