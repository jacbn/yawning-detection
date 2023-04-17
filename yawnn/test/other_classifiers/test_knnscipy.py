import unittest
import numpy as np
from yawnnlib.alternatives.manual import knn_scipy

class TestKnnScipy(unittest.TestCase):

    def test_knn(self):
        data = np.array([(-5, -5, -5, -5, -5, -5), (1, 1, 1, 1, 1, 1), (1, 1, 2, 3, 1, 1), (2, 1, 3, 1, 2, 1), (2, 3, 3, 3, 1, 2), (3, 2, 1, 1, 2, 3) , (-5, -3, -2, -5, -4, -1), (-5, -5, -2, -4, -3, -4)])
        classifications = [1, 1, 1, 1, 1, 0, 0, 0]
        self.assertEqual(knn_scipy.classify([1, 0, 2, 1, 3, 1], data, classifications), 1)
        self.assertEqual(knn_scipy.classify([0, 0, 5, 3, 2, 1], data, classifications), 1)
        self.assertEqual(knn_scipy.classify([10, 10, 5, 2, 3, 1], data, classifications), 1)
        self.assertEqual(knn_scipy.classify([1, -5, -5, -5, -4, -3], data, classifications), 0)

if __name__ == "__main__":
    unittest.main()