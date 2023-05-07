import unittest
from yawnnlib.evaluation import metrics

import numpy as np

ground1 = np.array([1, 0, 0, 1, 1, 0, 1, 0, 0, 1])
pred1   = np.array([1, 0, 1, 1, 1, 0, 0, 0, 0, 1])

ground2 = np.array([1, 0, 1, 1, 1, 0, 0, 0, 0, 0])
pred2   = np.array([0, 0, 0, 0, 1, 0, 0, 0, 1, 0])

ground3 = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
pred3   = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

ground4 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
pred4   = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

ground5 = np.array([[False], [False], [False], [True], [False], [False], [False], [False], [True], [False]])
pred5   = np.array([[False], [False], [True], [True], [True], [False], [True], [False], [True], [False]])

class TestMetrics(unittest.TestCase):   
    def test_calcAccuracy(self):
        self.assertAlmostEqual(0.8, metrics.calcAccuracy(ground1, pred1))
        self.assertAlmostEqual(0.6, metrics.calcAccuracy(ground2, pred2))
        self.assertAlmostEqual(1.0, metrics.calcAccuracy(ground3, pred3))
        self.assertAlmostEqual(1.0, metrics.calcAccuracy(ground4, pred4))
        self.assertAlmostEqual(0.7, metrics.calcAccuracy(ground5, pred5))
        
    def test_calcPrecision(self):
        self.assertAlmostEqual(0.8, metrics.calcPrecision(ground1, pred1))
        self.assertAlmostEqual(0.5, metrics.calcPrecision(ground2, pred2))
        self.assertAlmostEqual(1.0, metrics.calcPrecision(ground3, pred3))
        self.assertAlmostEqual(0.0, metrics.calcPrecision(ground4, pred4))
        self.assertAlmostEqual(0.4, metrics.calcPrecision(ground5, pred5))
        
    def test_calcRecall(self):
        self.assertAlmostEqual(0.8, metrics.calcRecall(ground1, pred1))
        self.assertAlmostEqual(1/4, metrics.calcRecall(ground2, pred2))
        self.assertAlmostEqual(1.0, metrics.calcRecall(ground3, pred3))
        self.assertAlmostEqual(0.0, metrics.calcRecall(ground4, pred4))
        self.assertAlmostEqual(1.0, metrics.calcRecall(ground5, pred5))
        
    def test_calcF1(self):
        self.assertAlmostEqual(0.8, metrics.calcF1(0.8, 0.8))
        self.assertAlmostEqual(1/3, metrics.calcF1(0.5, 0.25))
        self.assertAlmostEqual(1.0, metrics.calcF1(1.0, 1.0))
        self.assertAlmostEqual(0.0, metrics.calcF1(0.0, 0.0))
        self.assertAlmostEqual(4/7, metrics.calcF1(0.4, 1.0))
        
    
if __name__ == "__main__": # pragma: no cover
    unittest.main()