from abc import ABC, abstractmethod
from scipy import signal
from scipy.stats import norm
from enum import Enum
from yawnnlib.structure.timestamp import Timestamp
import numpy as np

class ApplyType(Enum):
    # most filters apply to the entire signal,
    # but some apply to the chunks of data used to train with
    SESSION = 1
    SPLIT = 2

class DataFilter(ABC):
    @abstractmethod
    def apply(self, data : np.ndarray) -> np.ndarray:
        pass
    
    @abstractmethod
    def getApplyType(self) -> ApplyType:
        pass
    
class TimestampedDataFilter(DataFilter):
    @abstractmethod
    def manageTimestamps(self, data : np.ndarray, timestamps : list[Timestamp]) -> list[Timestamp]:
        pass
    
class FilterCollection(DataFilter):
    def __init__(self, filters : list[DataFilter]) -> None:
        super().__init__()
        self.filters = filters
        
    def apply(self, data : np.ndarray) -> np.ndarray:
        for f in self.filters:
            data = f.apply(data)
        return data

# remove high frequency noise. returns a new signal (same length as original) with the noise removed
class LowPassFilter(DataFilter):
    def __init__(self, sampleRate : int, cutoff : int) -> None:
        super().__init__()
        self.sampleRate = sampleRate
        self.cutoff = cutoff
    
    def apply(self, data : np.ndarray) -> np.ndarray:
        b, a = signal.butter(10, self.cutoff/(self.sampleRate/2), 'low', analog=False)
        return signal.filtfilt(b, a, data, axis=0)
    
    def getApplyType(self) -> ApplyType:
        return ApplyType.SESSION

class NoneFilter(DataFilter):
    def apply(self, data : np.ndarray) -> np.ndarray:
        return data
    
    def getApplyType(self) -> ApplyType:
        return ApplyType.SESSION
    
class MovingAverageFilter(DataFilter):
    def __init__(self, windowSize : int = 5) -> None:
        super().__init__()
        self.windowSize = windowSize
    
    def apply(self, data : np.ndarray) -> np.ndarray:
        return np.apply_along_axis(lambda d: self._moving_average(d, self.windowSize), 0, data)
        # return np.cumsum(data, axis=0) / self.windowSize
        # return np.convolve(data, np.ones(self.windowSize)/self.windowSize, mode='same')
    
    def getApplyType(self) -> ApplyType:
        return ApplyType.SESSION
    
    @staticmethod
    def _moving_average(data : np.ndarray, window : int) -> np.ndarray:
        """ Applies a moving average to a 1D array. """
        d = np.copy(data)
        for i in range(len(data)):
            d[i] = np.mean(data[max(0, i - window//2):min(len(data), int(np.ceil(i + window/2)))])
        return d
    
class SmoothFilter(TimestampedDataFilter):
    """ Smooths the ends of the data using a gaussian. Used to prevent cut-off yawns from registering. """
    def __init__(self, keepData : float = 0.5) -> None:
        """ Create a new SmoothFilter
        
        Attributes
        ----------
        
        keepData : float = 0.5
            fraction of the data to keep exactly as is 
        
        """
        super().__init__()
        self.keepData = keepData
    
    def apply(self, data : np.ndarray) -> np.ndarray:
        # ---------------------------------------------------- : let this represent all the data
        #
        #           --------------------------------           : this data ((keepData * 100%) of the total) is kept exactly as is
        # ----------                                ---------- : this data is smoothed according to the smoothCurve function
        #
        #      ------------------------------------------      : this data (halfway into the smoothed data) keeps any timestamps
        # -----                                          ----- : this data does not keep any timestamps
        return self.smoothCurve(data)
    
    def manageTimestamps(self, data : np.ndarray, timestamps : list[Timestamp]) -> list[Timestamp]:
        lBound = int(len(data) * (1 - self.keepData)/4)
        return [t for t in timestamps if lBound <= t.time <= len(data) - lBound]
    
    def getApplyType(self) -> ApplyType:
        return ApplyType.SPLIT

    def smoothCurve(self, data : np.ndarray, plot : bool = False) -> np.ndarray:
        # run this file to view a plot of this distribution
        x = np.arange(0, 1, step = 1/len(data))
        # old method: cut-off gaussian. had sharp edges at cut-off boundary
        # new method: double smoothstep. less choice over smoothness but default is good
        f = lambda x: 3 * x**2 - 2 * x**3
        curveLength = int(len(data) * (1 - self.keepData)/2)
        y = np.ones(len(data))
        y[:curveLength] = f(np.linspace(0, 1, curveLength))
        y[-curveLength:] = f(np.linspace(1, 0, curveLength))
        
        smoothed = np.apply_along_axis(lambda d: d*y, 0, data)
        
        if plot:
            plt.plot(x, y)
            plt.figure(2)
            plt.plot(x, smoothed)
            plt.show()
            
        return smoothed

if __name__ == "__main__":
    from matplotlib import pyplot as plt
    data_length = 200
    
    x = np.arange(0, data_length, step = data_length/1000)
    y = np.sin(x) # sample data
    f = SmoothFilter()
    f.smoothCurve(y, plot=True)
        