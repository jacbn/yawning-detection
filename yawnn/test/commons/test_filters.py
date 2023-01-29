import unittest
from yawnnlib.structure.fourierData import FourierData
from yawnnlib.lstm import eimuLSTM, fourierLSTM
from yawnnlib.commons import commons, filters
from scipy.signal import argrelextrema
import numpy as np

class TestFilters(unittest.TestCase):
    def test_LowPassFilter(self):
        # this session is composed of a 4Hz frequency and a 12Hz frequency
        session = FourierData.fromPath("./yawnn/test/test_data/high_freq.eimu")
        freqs, _ = session.getFourierData(chunkSize=commons.YAWN_TIME, chunkSeparation=commons.YAWN_TIME/4)
        
        # filter out the 12Hz frequency
        filteredSession = FourierData.applyFilter(session, filters.LowPassFilter(sampleRate=32, cutoff=8))
        assert isinstance(filteredSession, FourierData) # (ensure type checking)
        filteredFreqs, _ = filteredSession.getFourierData(chunkSize=commons.YAWN_TIME, chunkSeparation=commons.YAWN_TIME/4)
        
        xf, fourierData = session._getFFTMagnitudes(session._getDataByAxis(0))
        xf2, fourierData2 = filteredSession._getFFTMagnitudes(filteredSession._getDataByAxis(0))
        
        self.assertEqual(len(fourierData[fourierData > 20]), 2) # other minor frequencies exist, but there are 2 major frequencies
        self.assertEqual(len(fourierData2[fourierData2 > 1000]), 1) # the butterworth filter means there is only 1 major frequency, but it increases the magnitude of others around it
        # (for reference, the magnitude of the 4Hz frequency is 930,000 and the magnitude of the 12Hz frequency is 445,000)
        
        if __name__ == "__main__":
            # run this file for a plot of this test
            from matplotlib import pyplot as plt
            plt.stem(xf, fourierData)
            plt.figure(2)
            plt.stem(xf2, fourierData2)
            plt.show()
        
    
    def test_MovingAverageFilter(self):
        model = eimuLSTM.EimuLSTMInput(dataFilter=filters.MovingAverageFilter(5))
        data, annotations = model.fromPath("./yawnn/test/test_data/basic2.eimu")
        self.assertEqual(data[0][8].tolist(), [0.2, 0.2, 0.2, 0.2, 0.2, 0.2])
        self.assertEqual(data[0][9].tolist(), [0.4, 0.4, 0.4, 0.4, 0.4, 0.4])
        self.assertEqual(data[0][10].tolist(), [0.6, 0.6, 0.6, 0.6, 0.6, 0.6])
        
    def test_SmoothFilter(self):
        # todo
        pass
    
if __name__ == "__main__":
    unittest.main()
