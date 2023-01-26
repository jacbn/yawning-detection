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
    def applyType(self) -> ApplyType:
        pass
    
class TimestampedDataFilter(DataFilter):
    @abstractmethod
    def manageTimestamps(self, timestamps : list[Timestamp]) -> list[Timestamp]:
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
    def __init__(self, sampleRate, cutoff=5) -> None:
        super().__init__()
        self.sampleRate = sampleRate
        self.cutoff = cutoff
    
    def apply(self, data : np.ndarray) -> np.ndarray:
        b, a = signal.butter(4, self.cutoff/(self.sampleRate/2), 'low', analog=False)
        return signal.filtfilt(b, a, data)
    
    def applyType(self) -> ApplyType:
        return ApplyType.SESSION

class NoneFilter(DataFilter):
    def apply(self, data : np.ndarray) -> np.ndarray:
        return data
    
    def applyType(self) -> ApplyType:
        return ApplyType.SESSION
    
class MovingAverageFilter(DataFilter):
    def __init__(self, windowSize : int = 5) -> None:
        super().__init__()
        self.windowSize = windowSize
    
    def apply(self, data : np.ndarray) -> np.ndarray:
        return np.convolve(data, np.ones(self.windowSize)/self.windowSize, mode='same')
    
    def applyType(self) -> ApplyType:
        return ApplyType.SESSION
    
class SmoothFilter(TimestampedDataFilter):
    """ Smooths the ends of the data using a gaussian. Used to prevent cut-off yawns from registering. """
    def __init__(self, keepData : float = 0.5, edgeFlatness : float = 6.0) -> None:
        """ Create a new SmoothFilter
        
        Attributes
        ----------
        
        keepData : float = 0.5
            fraction of the data to keep exactly as is 
        edgeFlatness : float = 6.0
            higher value gives flatter edges but sharper cutoff
        
        """
        super().__init__()
        self.keepData = keepData
        self.edgeFlatness = edgeFlatness
    
    def apply(self, data : np.ndarray) -> np.ndarray:
        return self.smoothCurve(data)
    
    def manageTimestamps(self, timestamps : list[Timestamp]) -> list[Timestamp]:
        # TODO
        pass
    
    def applyType(self) -> ApplyType:
        return ApplyType.SPLIT

    def smoothCurve(self, data : np.ndarray, plot : bool = False) -> np.ndarray:
        x = np.arange(0, len(data), step = len(data)/1000)
        y = norm.pdf(x, len(data)/2, len(data) / self.edgeFlatness)
        y /= y[int(len(y) * (1 - self.keepData)/2)] # scale so that keep_data is >=1
        y[y > 1] = 1                                # but cap at 1
        # run this file to view a plot of this normal distribution
        
        smoothed = data * y
        
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
        