import commons

from eimuReader import SessionData, SensorReading, Timestamp

import numpy as np
from scipy.fft import rfft, rfftfreq, ifft
from scipy import signal
from matplotlib import pyplot as plt

SIGNIFICANT_FREQ = 0

class FourierData(SessionData):
    def __init__(self, dataset : list[SensorReading], timestamps : list[Timestamp], sampleRate : int, version : int):
        super().__init__(dataset, timestamps, sampleRate, version)
        self.sumFrequencies = []
        self.plotSessionData(show=False, dataFilter=commons.LowPassFilter(2))
        #self.plotFrequencies()
        
    def plotSessionData(self, show : bool = False, figure : int = 2, dataFilter : commons.DataFilter = commons.NoneFilter()):
        for axis in range(6):
            data = np.array(list(map(lambda x: x[axis%2][axis//2], zip(self.accel, self.gyro))))
            dataFiltered = dataFilter.apply(data, self.sampleRate)

            # self.plotFFTMagnitudes(data, axis, figure+5, False)
            # self.plotIFFTReconstruction(data, axis, figure+6, False)
            # self.plotSpectrograms(data, axis, figure+7, False, fmin=0, fmax=6) 
            
            self.plotFFTMagnitudes(dataFiltered, axis, figure, False)
            self.plotIFFTReconstruction(dataFiltered, axis, figure+1, False)
            self.plotSpectrograms(dataFiltered, axis, figure+2, False, fmin=0, fmax=6)           
        
        if show:
            plt.show()
    
    def plotFFTMagnitudes(self, data : np.ndarray, axis : int, figure : int = 2, show : bool = False):
        plt.figure(figure)
        plt.suptitle("FFT Magnitudes")
        ax = plt.subplot(3,2,axis+1)
        ax.set_title(commons.AXIS_NAMES[axis%2][axis//2])
        
        fourierData = np.abs(rfft(data)) # type: ignore # we use abs for everything here as we only care about the magnitude
        N = len(fourierData)
        xf = rfftfreq(N*2-1, 1/self.sampleRate)
        
        ax.stem(xf, fourierData, 'r', markerfmt=' ') # plots the magnitude of all frequencies in red
        fourierData = np.where(fourierData > SIGNIFICANT_FREQ, fourierData, 0)  
        ax.stem(xf, fourierData, commons.AXIS_COLOURS[axis//2], markerfmt=' ') # plots the magnitude of all frequencies greater than SIGNIFICANT_FREQ in blue

        if (show):
            plt.show()
            
    def plotIFFTReconstruction(self, data : np.ndarray, axis : int, figure : int = 3, show : bool = False):
        plt.figure(figure)
        plt.suptitle("Inverse FFT Reconstructions")
        ax = plt.subplot(3,2,axis+1)
        ax.set_title(commons.AXIS_NAMES[axis%2][axis//2], loc='left')
        
        fourierData = rfft(data)
        reconstructedData = ifft(fourierData) # here we do care about the sign, so we don't use abs
        
        # plots the reconstruction of the frequencies greater than SIGNIFICANT_FREQ
        ax.plot(np.arange(len(reconstructedData))*2, reconstructedData, color=commons.AXIS_COLOURS[axis//2]) 
        
        ax.set_title(commons.AXIS_NAMES[axis%2][axis//2], loc='left')
        ax.set_ylabel("Acceleration (m/s^2)" if axis//2 == 0 else "Angular Velocity (deg/s)")
        ax.set_xlabel(f"Samples ({self.sampleRate} = 1 sec)")
        
        for timestamp in self.timestamps:
            ax.axvline(timestamp.time, color='black', alpha=0.5)
            
        if (show):
            plt.show()
            
    def plotSpectrograms(self, data : np.ndarray, axis : int, figure : int = 5, show : bool = False, fmin : int = 0, fmax : int = 6):
        plt.figure(figure)
        plt.suptitle("Axis Spectrograms")
        ax = plt.subplot(3,2,axis+1)
        f, t, Sxx = signal.spectrogram(data, self.sampleRate, nperseg=256, noverlap=128)
        
        freq_slice = np.where((f >= fmin) & (f <= fmax))
        f = f[freq_slice]
        Sxx = Sxx[freq_slice,:][0] # type: ignore
        
        pc = ax.pcolormesh(t, f, Sxx, shading='gouraud')
        plt.colorbar(pc)
        
        for timestamp in self.timestamps:
            ax.axvline(timestamp.time/self.sampleRate, color='black', alpha=0.5)
            
        ax.set_title(commons.AXIS_NAMES[axis%2][axis//2], loc='left')
        ax.set_ylabel('Frequency [Hz]')
        ax.set_xlabel('Time [sec]')
        
        if (show):
            plt.show()
    
    def _initFrequencies(self):    
        sessionData, sessionTimestamps = self.toRaw()
        self.sumFrequencies = [[] for _ in range(6)]
        for i in range(len(sessionData)):
            for axis in range(6):
                fftVal = rfft(sessionData[i][:,axis])#[:sessionData.shape[1]//2] # type: ignore
                fftVal = 2/len(sessionData) * np.abs(fftVal) # type: ignore
                
                if len(self.sumFrequencies[axis]) == 0:
                    self.sumFrequencies[axis] = fftVal
                else:
                    self.sumFrequencies[axis] += fftVal
    
    def plotFrequencies(self, start=0, end=-1, figure : int = 4):
        if self.sumFrequencies == []:
            self._initFrequencies()
        plt.figure(figure)
        
        N = len(self.sumFrequencies[0])
        T = 1 / self.sampleRate
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
    
s = FourierData.fromPath("./yawnn/data/96hz-long2.eimu")

s.plot(show=True)
# s.plotFrequencies()