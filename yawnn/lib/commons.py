from typing import TypeVar, Callable
from abc import ABC, abstractmethod
from os import listdir
from os.path import isfile, join
from matplotlib import pyplot as plt
import numpy as np
from scipy import signal

AXIS_NAMES = [['Accel X', 'Accel Y', 'Accel Z'], ['Gyro X', 'Gyro Y', 'Gyro Z']]
AXIS_COLOURS = plt.rcParams['axes.prop_cycle'].by_key()['color']

YAWN_TIME = 2 # time, in seconds, an individual yawn lasts for
TRAIN_PERCENT = 0.8 # percentage of data to use for training

T = TypeVar('T')

def mapToDirectory(f : Callable[[str], T], path : str) -> list[T]:
    return [f(join(path,file)) for file in listdir(path) if isfile(join(path, file))]

def shuffleTrainingData(combined : tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    dataLength = len(combined[0][0])
    train = np.concatenate((combined[0][0], combined[1][0]))
    test = np.concatenate((combined[0][1], combined[1][1]))
    indices = np.arange(dataLength)
    
    trainLength = int(dataLength * TRAIN_PERCENT)
    return (train[indices][:trainLength], test[indices][:trainLength]), (train[indices][trainLength:], test[indices][trainLength:])

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
    