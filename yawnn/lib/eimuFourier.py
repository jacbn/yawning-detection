import commons
from eimuReader import SessionData, SensorReading, Timestamp

import numpy as np
from scipy.fft import fft, fftfreq
from matplotlib import pyplot as plt

SIGNIFICANT_FREQ = 0.1
SAMPLE_RATE = 32

class FourierData(SessionData):
    def __init__(self, dataset : list[SensorReading], timestamps : list[Timestamp], sampleRate : int, version : int):
        super().__init__(dataset, timestamps, sampleRate, version)
        self._initFrequencies()
        
    # @classmethod
    # def fromSessionData(cls, session : SessionData):
    #     return cls(session.rawDataset, session.timestamps, session.sampleRate, session.version)
    
    def _initFrequencies(self):    
        sessionData, sessionTimestamps = self.toRaw()
        self.sumFrequencies = [[] for _ in range(6)]
        # self.frequencies = [[None for _ in range(6)] for _ in range(len(sessionData))]
        for i in range(len(sessionData)):
            for axis in range(6):
                fftVal = fft(sessionData[i][:,axis])[:sessionData.shape[1]//2] # type: ignore
                fftVal = 2/len(sessionData) * np.abs(fftVal) # type: ignore
                # self.frequencies[i][axis] = fftVal # type: ignore
                
                if len(self.sumFrequencies[axis]) == 0:
                    self.sumFrequencies[axis] = fftVal
                else:
                    self.sumFrequencies[axis] += fftVal
    
    
    def plotFrequencies(self, start=0, end=-1, figure : int = 2):
        plt.figure(figure)
        
        N = len(self.sumFrequencies[0])
        T = 1 / SAMPLE_RATE
        xf = fftfreq(N*2, T)[:N]
        
        for i in range(6):
            yf = self.sumFrequencies[i]
            
            # significant = list(map(lambda y: xf[y[0]], filter(lambda x: x[1] > SIGNIFICANT_FREQ, enumerate(posF))))
            # print(significant)
        
            plt.plot(xf, yf) 
    
            
        plt.grid()
        plt.legend(["ax", "ay", "az", "gx", "gy", "gz"], loc="upper right")
        plt.show()
    

# def directoryToFourierInput(path : str) -> tuple[tuple, tuple]:
#     inputs = commons.mapToDirectory(eimuToFourier, path)
#     # combine all the inputs. each is a tuple of (trainX, trainY), (testX, testY),
#     # and the result is a combination of all the trainX, trainY, testX, testY individually
#     return ((),())
    
# if __name__ == "__main__":
#     # directoryToFourierInput("./yawnn/data")
#     eimuToFourier("./yawnn/data/long3.eimu")
    
s = FourierData.fromPath("./yawnn/data/long1.eimu")

s.plot(show=False)
s.plotFrequencies()