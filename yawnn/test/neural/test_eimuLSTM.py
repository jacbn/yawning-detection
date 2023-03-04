import unittest
from yawnnlib.neural import eimuLSTM
from yawnnlib.utils import commons, filters, config
import numpy as np

PROJECT_ROOT = config.get("PROJECT_ROOT")

class TestEimuLSTM(unittest.TestCase):
    def test_fromPathOnRegularData(self):
        commons.YAWN_TIME = 2
        data, annotations = eimuLSTM.EimuLSTMInput(sessionWidth=commons.YAWN_TIME, sessionGap=1/32).fromPath(f"{PROJECT_ROOT}/test/test_data/basic1.eimu")
        self.assertEqual(data.shape, (10, commons.YAWN_TIME * 32, 6))
        # 32Hz * yawn time = X-sample splits
        # the data has room for 10 sets of X, each with 6 axes -> (1, X, 6)
        self.assertEqual(annotations.shape, (10, 1))
        # 10 sets of annotations, each with 1 value -> (10, 1)

    def test_fromPathOnShortData(self):
        commons.YAWN_TIME = 2
        data, annotations = eimuLSTM.EimuLSTMInput(sessionWidth=commons.YAWN_TIME, sessionGap=1/32).fromPath(f"{PROJECT_ROOT}/test/test_data/short1.eimu")
        self.assertEqual(data.shape, (1, commons.YAWN_TIME * 32, 6))
        # the data should be padded to fit commons.YAWN_TIME * 32
        self.assertEqual(annotations.shape, (1, 1))

    def test_fromPathOnRegularDataWithFilter(self):
        commons.YAWN_TIME = 2
        # check that none of the following throw an exception
        eimuLSTM.EimuLSTMInput(sessionWidth=commons.YAWN_TIME, sessionGap=3/32, dataFilter=filters.MovingAverageFilter(5)).fromPath(f"{PROJECT_ROOT}/test/test_data/basic2.eimu")
        # todo
        
if __name__ == "__main__":
    unittest.main()
    