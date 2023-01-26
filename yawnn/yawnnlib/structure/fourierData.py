from yawnnlib.commons import commons, filters
from yawnnlib.structure.sessionData import SessionData
from yawnnlib.structure.sensorReading import SensorReading
from yawnnlib.structure.timestamp import Timestamp

import numpy as np
from scipy.fft import rfft, rfftfreq, ifft
from scipy import signal
from matplotlib import pyplot as plt

SIGNIFICANT_FREQ = 0
N_PER_SEG = 256
N_OVERLAP = N_PER_SEG-1 # greater n_overlap generally preferable. will miss the start/end of recording but much higher time resolution

class FourierData(SessionData):
    def __init__(self, dataset : list[SensorReading], timestamps : list[Timestamp], sampleRate : int, version : int, sessionID : int = -1, totalSessions : int = -1):
        super().__init__(dataset, timestamps, sampleRate, version, sessionID, totalSessions)
        self.sumFrequencies = []
        # _filter = commons.FilterCollection([commons.LowPassFilter(self.sampleRate, 2), commons.MovingAverageFilter(5)])
        # _filter = filters.NoneFilter()
        # self.plotSessionData(show=False, dataFilter=_filter)
        # self.getFourierData(dataFilter=_filter)
        # self.plotFrequencies()
    
    
    def getFourierData(self, dataFilter : filters.DataFilter = filters.NoneFilter(), chunkSize : float = commons.YAWN_TIME, chunkSeparation : float = commons.YAWN_TIME/4):
        '''Returns spectrogram data for the given input data, split into chunks of chunkSize seconds, with a separation between chunks of chunkSeparation seconds.'''
        frequencies = []
        timestamps = []
        
        trueChunkSize = int(chunkSize * self.sampleRate)
        trueChunkSeparation = int(chunkSeparation * self.sampleRate)
        boundary = N_PER_SEG//2 #todo: factor in overlap 
          
        pString = f"Calculating Fourier frequencies ({self.fileNum}/{self.totalFiles}): "
        print(pString + "......", end='')
            
        for axis in range(6):
            # obtain and filter the data
            data = np.array(list(map(lambda x: x[axis%2][axis//2], zip(self.accel, self.gyro))))
            dataFiltered = dataFilter.apply(data)
            
            # there won't be any spectrogram data outside of dataFiltered[boundary:-boundary] as this is the boundary required to calculate the fft
            if trueChunkSize > len(dataFiltered[boundary:-boundary]):
                raise ValueError(f"Not enough data to split into chunks of {chunkSize} seconds. Are you using the right file?")
            
            # split the data into chunks
            SxxList = []            
            chunkStart = boundary

            while chunkStart + trueChunkSize < len(dataFiltered) - boundary:
                chunk = dataFiltered[chunkStart-boundary : chunkStart+trueChunkSize+boundary]
                f, t, Sxx = signal.spectrogram(chunk, self.sampleRate, nperseg=N_PER_SEG, noverlap=N_OVERLAP)
                SxxList.append(Sxx)
                # timestamps.append(int(1 in self.getYawnIndices()[range(chunkStart, chunkStart+trueChunkSize)]))
                if axis == 0:
                    timestamps.append(self.getYawnIndices()[chunkStart+trueChunkSize//2])
                chunkStart += trueChunkSeparation
            
            SxxList = np.array(SxxList)
            frequencies.append(SxxList)
            
            print('\r' + pString + '#' * (axis+1) + '.' * (5-axis), end='' if axis < 5 else '\n')
            
            # f, t, Sxx = signal.spectrogram(dataFiltered, self.sampleRate, nperseg=N_PER_SEG, noverlap=N_OVERLAP)
            # frequencies.append(Sxx)
            # print(len(f), len(t), Sxx.shape)
            # print((t[-1]+t[0]) * self.sampleRate, len(dataFiltered))
        
        # print(frequencies)
        return np.array(frequencies), timestamps
    
        
    def plotSessionData(self, show : bool = False, figure : int = 2, dataFilter : filters.DataFilter = filters.NoneFilter()):
        for axis in range(6):
            data = np.array(list(map(lambda x: x[axis%2][axis//2], zip(self.accel, self.gyro))))
            dataFiltered = dataFilter.apply(data)
            
            self.plotFFTMagnitudes(dataFiltered, axis, figure, False)
            self.plotIFFTReconstruction(dataFiltered, axis, figure+1, False)
            self.plotSpectrograms(dataFiltered, axis, figure+2, False, fmin=0, fmax=6, maxAmp=-1)           
        
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
            
    def plotSpectrograms(self, data : np.ndarray, axis : int, figure : int = 5, show : bool = False, fmin : int = 0, fmax : int = 6, maxAmp : int = -1):
        plt.figure(figure)
        plt.suptitle("Axis Spectrograms")
        ax = plt.subplot(3,2,axis+1)
        f, t, Sxx = signal.spectrogram(data, self.sampleRate, nperseg=N_PER_SEG, noverlap=N_OVERLAP)
        
        freq_slice = np.where((f >= fmin) & (f <= fmax))
        f = f[freq_slice]
        Sxx = Sxx[freq_slice,:][0] # type: ignore
        
        if maxAmp > 0:
            Sxx[Sxx > maxAmp] = np.nan
        
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
        sessionData, sessionTimestamps = self.getEimuData()
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
    
    
if __name__ == "__main__":
    s = FourierData.fromPath("./yawnn/data/tests/96hz/96hz-yawns1.eimu")
    s.plot(show=True)
    # s.plotFrequencies()
    