import commons
from eimuReader import SessionData, SensorReading, Timestamp

import numpy as np
from scipy.fft import rfft, rfftfreq, ifft
from matplotlib import pyplot as plt

SIGNIFICANT_FREQ = 1
SAMPLE_RATE = 32

class FourierData(SessionData):
    def __init__(self, dataset : list[SensorReading], timestamps : list[Timestamp], sampleRate : int, version : int):
        super().__init__(dataset, timestamps, sampleRate, version)
        
        f = rfft(list(map(lambda x: x[2], self.accel)))
        N = len(f)
        xf = rfftfreq(N*2-1, 1/SAMPLE_RATE)
        
        # we use abs below when we only care about the magnitude, not whether it is negative
        
        plt.stem(xf, np.abs(f), 'r', markerfmt=' ') # plots the magnitude of all frequencies in red
        f = np.where(np.abs(f) > SIGNIFICANT_FREQ, f, 0)  
        
        plt.stem(xf, np.abs(f)) # plots the magnitude of all frequencies greater than SIGNIFICANT_FREQ in blue
        plt.figure(4)
        
        yinv = ifft(f)
        plt.plot(np.arange(len(yinv)), yinv) # plots the reconstruction (inverse fft) of the frequencies greater than SIGNIFICANT_FREQ
        plt.figure(3)
        
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
                fftVal = rfft(sessionData[i][:,axis])#[:sessionData.shape[1]//2] # type: ignore
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
        xf = rfftfreq(N*2 - (1 if N%2==1 else 0), T)
        # xf = np.arange(0, 1/SAMPLE_RATE, 1/(SAMPLE_RATE*N))
        
        for i in range(6):
            yf = self.sumFrequencies[i]
            
            if i == 2:
                print("--------------------------------------")
                print(yf)
                
            
            # significant = list(map(lambda y: xf[y[0]], filter(lambda x: x[1] > SIGNIFICANT_FREQ, enumerate(posF))))
            # print(significant)
        
            # plt.plot(xf, yf) 
            plt.stem(xf, yf) #[:N//2]
    
            
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
    
s = FourierData.fromPath("./yawnn/data/long2.eimu")

s.plot(show=False)
s.plotFrequencies()