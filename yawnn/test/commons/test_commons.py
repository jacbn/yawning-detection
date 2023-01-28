import unittest
from yawnnlib.commons import commons
from yawnnlib.lstm import eimuLSTM

class TestCommons(unittest.TestCase):
    def test_directoryToModelData(self):
        (x1, y1), (x2, y2) = commons.directoryToModelData("./yawnn/test/test_data/directory_test", eimuLSTM.EimuLSTMInput())
        self.assertEqual(x1.shape, (1, 32 * commons.YAWN_TIME, 6))
        self.assertEqual(y1.shape, (1, 1))
        self.assertEqual(x2.shape, (1, 32 * commons.YAWN_TIME, 6))
        self.assertEqual(y2.shape, (1, 1))
