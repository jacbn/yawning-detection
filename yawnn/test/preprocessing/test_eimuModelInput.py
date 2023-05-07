import unittest
from yawnnlib.preprocessing import eimuModelInput
from yawnnlib.utils import filters, config

PROJECT_ROOT = config.PROJECT_ROOT
config.set("YAWN_TIME", 2)

class TestEimuModelInput(unittest.TestCase):
    def test_fromPathOnRegularData(self):
        YAWN_TIME = config.get("YAWN_TIME")
        data, annotations = eimuModelInput.EimuModelInput(windowSize=YAWN_TIME, windowSep=1/32).fromPath(f"{PROJECT_ROOT}/test/test_data/basic1.eimu")
        self.assertEqual(data.shape, (10, YAWN_TIME * 32, 6))
        # 32Hz * yawn time = X-sample splits
        # the data has room for 10 sets of X, each with 6 axes -> (1, X, 6)
        self.assertEqual(annotations.shape, (10, 1))
        # 10 sets of annotations, each with 1 value -> (10, 1)

    def test_fromPathOnShortData(self):
        YAWN_TIME = config.get("YAWN_TIME")
        data, annotations = eimuModelInput.EimuModelInput(windowSize=YAWN_TIME, windowSep=1/32).fromPath(f"{PROJECT_ROOT}/test/test_data/short1.eimu")
        self.assertEqual(data.shape, (1, YAWN_TIME * 32, 6))
        # the data should be padded to fit YAWN_TIME * 32
        self.assertEqual(annotations.shape, (1, 1))

    def test_fromPathOnRegularDataWithFilter(self):
        YAWN_TIME = config.get("YAWN_TIME")
        # check that none of the following throw an exception
        eimuModelInput.EimuModelInput(windowSize=YAWN_TIME, windowSep=3/32, dataFilter=filters.MovingAverageFilter(5)).fromPath(f"{PROJECT_ROOT}/test/test_data/basic2.eimu")
        eimuModelInput.EimuModelInput(windowSize=YAWN_TIME, windowSep=3/32, dataFilter=filters.SmoothFilter(keepData=0.7)).fromPath(f"{PROJECT_ROOT}/test/test_data/basic2.eimu")
        eimuModelInput.EimuModelInput(windowSize=YAWN_TIME, windowSep=3/32, dataFilter=filters.NormalisationFilter()).fromPath(f"{PROJECT_ROOT}/test/test_data/basic2.eimu")
        
if __name__ == "__main__":
    unittest.main()
    