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

T = TypeVar('T')

def mapToDirectory(f : Callable[[str], T], path : str) -> list[T]:
    return [f(join(path,file)) for file in listdir(path) if isfile(join(path, file))]

class DataFilter(ABC):
    @abstractmethod
    def apply(self, data : np.ndarray, sampleRate : int) -> np.ndarray:
        pass
    
class LowPassFilter(DataFilter):
    def __init__(self, cutoff=5) -> None:
        super().__init__()
        self.cutoff = cutoff
    
    def apply(self, data : np.ndarray, sampleRate : int) -> np.ndarray:
        b, a = signal.butter(4, self.cutoff/(sampleRate/2), 'low', analog=False)
        return signal.filtfilt(b, a, data)

class NoneFilter(DataFilter):
    def apply(self, data : np.ndarray, sampleRate : int) -> np.ndarray:
        return data
    