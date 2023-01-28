import unittest
from yawnnlib.lstm import eimuLSTM
from yawnnlib.commons import commons, filters
import numpy as np

class TestEimuLSTM(unittest.TestCase):
    def test_fromPathOnRegularData(self):
        data, annotations = eimuLSTM.EimuLSTMInput(sessionGap=1).fromPath("./yawnn/test/test_data/basic1.eimu")
        self.assertEqual(data.shape, (10, commons.YAWN_TIME * 32, 6))
        # 32Hz * yawn time = X-sample splits
        # the data has room for 10 sets of X, each with 6 axes -> (1, X, 6)
        self.assertEqual(annotations.shape, (10, 1))
        # 10 sets of annotations, each with 1 value -> (10, 1)

    def test_fromPathOnShortData(self):
        data, annotations = eimuLSTM.EimuLSTMInput(sessionGap=1).fromPath("./yawnn/test/test_data/short1.eimu")
        self.assertEqual(data.shape, (1, commons.YAWN_TIME * 32, 6))
        # the data should be padded to fit commons.YAWN_TIME * 32
        self.assertEqual(annotations.shape, (1, 1))

    def test_fromPathOnRegularDataWithFilter(self):
        data, annotations = eimuLSTM.EimuLSTMInput(dataFilter=filters.MovingAverageFilter(5)).fromPath("./yawnn/test/test_data/basic2.eimu")
        self.assertEqual(data[0][8].tolist(), [0.2, 0.2, 0.2, 0.2, 0.2, 0.2])
        self.assertEqual(data[0][9].tolist(), [0.4, 0.4, 0.4, 0.4, 0.4, 0.4])
        self.assertEqual(data[0][10].tolist(), [0.6, 0.6, 0.6, 0.6, 0.6, 0.6])

# with np.printoptions(threshold=np.inf):
#     data, annotations = eimuLSTM.EimuLSTMInput(dataFilter=filters.MovingAverageFilter(5)).fromPath("./yawnn/test/test_data/basic2.eimu")
#     print(data)
#     print(annotations)