import unittest
from yawnnlib.preprocessing import fftModelInput
from yawnnlib.utils import filters, config

PROJECT_ROOT = config.PROJECT_ROOT
config.set("YAWN_TIME", 2)

class TestEimuModelInput(unittest.TestCase):
    def test_fromPathOnRegularData(self):
        YAWN_TIME = config.get("YAWN_TIME")
        data, annotations = fftModelInput.FFTModelInput(dataFilter=filters.NormalisationFilter(), windowSize=YAWN_TIME, windowSep=1/32, nPerSeg=128, nOverlap=96).applyModelTransformOnPath(f"{PROJECT_ROOT}/test/test_data/high_freq.eimu")
        self.assertEqual(data.shape[1:], (97, 6))
        self.assertEqual(annotations.shape, (data.shape[0], 1))

    def test_fromPathOnShortData(self):
        YAWN_TIME = config.get("YAWN_TIME")
        with self.assertRaises(ValueError):
            _ = fftModelInput.FFTModelInput(dataFilter=filters.NormalisationFilter(), windowSize=YAWN_TIME, windowSep=1/32).applyModelTransformOnPath(f"{PROJECT_ROOT}/test/test_data/short1.eimu")

    def test_fromPathOnRegularDataWithFilter(self):
        YAWN_TIME = config.get("YAWN_TIME")
        # check that none of the following throw an exception
        fftModelInput.FFTModelInput(windowSize=YAWN_TIME, windowSep=3/32, dataFilter=filters.FilterCollection([filters.NormalisationFilter(), filters.MovingAverageFilter(5)])).applyModelTransformOnPath(f"{PROJECT_ROOT}/test/test_data/high_freq.eimu")
        fftModelInput.FFTModelInput(windowSize=YAWN_TIME, windowSep=3/32, dataFilter=filters.FilterCollection([filters.NormalisationFilter(), filters.SmoothFilter(keepData=0.7)])).applyModelTransformOnPath(f"{PROJECT_ROOT}/test/test_data/high_freq.eimu")
        fftModelInput.FFTModelInput(windowSize=YAWN_TIME, windowSep=3/32, dataFilter=filters.FilterCollection([filters.NormalisationFilter(), filters.LowPassFilter(32, 10)])).applyModelTransformOnPath(f"{PROJECT_ROOT}/test/test_data/high_freq.eimu")
        
if __name__ == "__main__": # pragma: no cover
    unittest.main()
    