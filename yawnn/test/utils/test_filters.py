import unittest
from yawnnlib.structure.fourierData import FourierData
from yawnnlib.preprocessing import eimuModelInput
from yawnnlib.utils import commons, filters
import numpy as np

class TestFilters(unittest.TestCase):
    def test_LowPassFilter(self):
        # this session is composed of a 4Hz frequency and a 12Hz frequency
        session = FourierData.fromPath(f"{commons.PROJECT_ROOT}/test/test_data/high_freq.eimu")
        
        # filter out the 12Hz frequency
        filteredSession = FourierData.applyFilter(
            session, 
            filters.LowPassFilter(sampleRate=32, cutoff=8),
            filters.ApplyType.SESSION
        )
        assert isinstance(filteredSession, FourierData) # (ensure type checking)
        
        xf, fourierData = session._getFFTMagnitudes(session._getDataByAxis(0))
        xf2, fourierData2 = filteredSession._getFFTMagnitudes(filteredSession._getDataByAxis(0))
        
        self.assertEqual(len(fourierData[fourierData > 20]), 2) # other minor frequencies exist, but there are 2 major frequencies
        self.assertEqual(len(fourierData2[fourierData2 > 1000]), 1) # the butterworth filter means there is only 1 major frequency, but it increases the magnitude of others around it
        # (for reference, the magnitude of the 4Hz frequency is 930,000 and the magnitude of the 12Hz frequency is 445,000)
        
        if __name__ == "__main__":
            # run this file for a plot of this test
            from matplotlib import pyplot as plt
            plt.title("Frequency Domain without Low-Pass Filter")
            plt.stem(xf, fourierData)
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Magnitude")
            plt.figure(2)
            plt.title("Frequency Domain with Low-Pass Filter")
            plt.stem(xf2, fourierData2)
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Magnitude")
            plt.show()
        
    
    def test_MovingAverageFilter(self):
        model = eimuModelInput.EimuModelInput(windowSize=commons.YAWN_TIME*32, windowSep=3/32, dataFilter=filters.MovingAverageFilter(5))
        data, annotations = model.fromPath(f"{commons.PROJECT_ROOT}/test/test_data/basic2.eimu")
        self.assertEqual(data[0][8].tolist(), [0.2, 0.2, 0.2, 0.2, 0.2, 0.2])
        self.assertEqual(data[0][9].tolist(), [0.4, 0.4, 0.4, 0.4, 0.4, 0.4])
        self.assertEqual(data[0][10].tolist(), [0.6, 0.6, 0.6, 0.6, 0.6, 0.6])
        
    def test_SmoothFilter(self):
        # todo
        pass
    
if __name__ == "__main__":
    unittest.main()
