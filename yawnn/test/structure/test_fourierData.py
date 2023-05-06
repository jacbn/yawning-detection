import unittest
from yawnnlib.structure import fourierData
from yawnnlib.utils import commons, config

config.set("YAWN_TIME", 2)

class TestFourierData(unittest.TestCase):
    def test_getFourierData(self):
        YAWN_TIME = config.get("YAWN_TIME")
        session = fourierData.FourierData.fromPath(f"{commons.PROJECT_ROOT}/test/test_data/high_freq.eimu", nPerSeg=256, nOverlap=255)
        freqs, timestamps = session.getSpectrogramData(windowSize=YAWN_TIME, windowSep=YAWN_TIME/4)
        # data format is (axes, windows, frequencies, samples per window).
        self.assertTupleEqual(freqs.shape, (66, 65, 129, 6))
        # the number of frequencies and samples per window are functions of N_PER_SEG, N_OVERLAP, windowSize and windowSep.
        # the number of windows is a function of the length of the data

if __name__ == "__main__":
    unittest.main()
    