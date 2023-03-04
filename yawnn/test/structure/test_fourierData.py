import unittest
from yawnnlib.structure import fourierData
from yawnnlib.utils import commons

class TestFourierData(unittest.TestCase):
    def test_getFourierData(self):
        commons.YAWN_TIME = 2
        session = fourierData.FourierData.fromPath(f"{commons.PROJECT_ROOT}/test/test_data/high_freq.eimu", nPerSeg=256, nOverlap=255)
        freqs, timestamps = session.getSpectrogramData(chunkSize=commons.YAWN_TIME, chunkSeparation=commons.YAWN_TIME/4)
        # data format is (axes, chunks, frequencies, samples per chunk).
        self.assertTupleEqual(freqs.shape, (66, 65, 129, 6))
        # the number of frequencies and samples per chunk are functions of N_PER_SEG, N_OVERLAP, chunkSize and chunkSeparation.
        # the number of chunks is a function of the length of the data

if __name__ == "__main__":
    unittest.main()
    