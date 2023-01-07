from abc import ABC, abstractmethod
from scipy import signal
import numpy as np

class DataFilter(ABC):
    @abstractmethod
    def apply(self, data : np.ndarray) -> np.ndarray:
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

class NoneFilter(DataFilter):
    def apply(self, data : np.ndarray) -> np.ndarray:
        return data
    
class MovingAverageFilter(DataFilter):
    def __init__(self, windowSize=5) -> None:
        super().__init__()
        self.windowSize = windowSize
    
    def apply(self, data : np.ndarray) -> np.ndarray:
        return np.convolve(data, np.ones(self.windowSize)/self.windowSize, mode='same')