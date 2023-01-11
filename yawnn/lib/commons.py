from typing import TypeVar, Callable
from abc import ABC, abstractmethod
from os import listdir
from os.path import isfile, join
from matplotlib import pyplot as plt
import numpy as np

AXIS_NAMES = [['Accel X', 'Accel Y', 'Accel Z'], ['Gyro X', 'Gyro Y', 'Gyro Z']]
AXIS_COLOURS = plt.rcParams['axes.prop_cycle'].by_key()['color']

YAWN_TIME = 2 # time, in seconds, an individual yawn lasts for
TRAIN_PERCENT = 0.8 # percentage of data to use for training

T = TypeVar('T')
ModelData = tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]

class ModelType(ABC):
    """ An abstract class that represents an input to an LSTM model. See eimuLSTM.EimuLSTMInput for an example implementation. """
    @abstractmethod
    def fromPath(self, path : str, fileNum : int = -1, totalFiles : int = -1) -> ModelData:
        pass
    
    @abstractmethod
    def getType(self) -> str:
        pass
    

def directoryToModelData(directoryPath : str, modelType : ModelType, shuffle : bool = True, equalPositiveAndNegative : bool = True) -> ModelData:
    """ Convert a directory of .eimu files to a tuple of (trainX, trainY), (testX, testY) using a given LSTMInput. """
    inputs = mapToDirectory(modelType.fromPath, directoryPath)
    # combine all the inputs. each is a tuple of (trainX, trainY), (testX, testY),
    # and the result is a combination of all the trainX, trainY, testX, testY individually
    combined = (np.concatenate(list(map(lambda x: x[0][0], inputs))), np.concatenate(list(map(lambda x: x[0][1], inputs)))), (np.concatenate(list(map(lambda x: x[1][0], inputs))), np.concatenate(list(map(lambda x: x[1][1], inputs))))
    
    if equalPositiveAndNegative:
        combined = equalizePositiveAndNegative(combined, shuffle)
    if shuffle:
        combined = shuffleAllData(combined)
    
    return combined


def mapToDirectory(f : Callable[[str, int, int], T], path : str) -> list[T]:
    """ Apply a function to all files in a directory. The function should take input parameters [fileName : str, fileNum : int, totalFiles : int]. """
    files = [join(path,file) for file in listdir(path) if isfile(join(path, file))]
    return [f(file, i+1, len(files)) for i, file in enumerate(files)]


def shuffleAllData(combined : ModelData) -> ModelData:
    """ Shuffles all the data, across both the training and test sets. """
    dataLength = len(combined[0][0])
    data = np.concatenate((combined[0][0], combined[1][0]))
    annotations = np.concatenate((combined[0][1], combined[1][1]))
    indices = np.arange(dataLength)
    np.random.shuffle(indices)
    
    trainLength = int(dataLength * TRAIN_PERCENT)
    return (data[indices][:trainLength], annotations[indices][:trainLength]), (data[indices][trainLength:], annotations[indices][trainLength:])


def equalizePositiveAndNegative(combined : ModelData, shuffle : bool) -> ModelData:
    """ Equalizes the number of positive and negative examples in the training data. """
    trainX, trainY = combined[0]
    positiveIndices = np.where(trainY == 1)[0]
    negativeIndices = np.where(trainY == 0)[0]
    
    np.random.shuffle(positiveIndices) # shuffle the indices so we don't always remove the last ones
    np.random.shuffle(negativeIndices)
    
    if len(positiveIndices) > len(negativeIndices):
        positiveIndices = positiveIndices[:len(negativeIndices)]
    elif len(negativeIndices) > len(positiveIndices):
        negativeIndices = negativeIndices[:len(positiveIndices)]
    
    indices = np.concatenate((positiveIndices, negativeIndices))
    if not shuffle:
        # if we're not going to shuffle later, need to sort the indices back into the original order
        indices = np.sort(indices)
        
    return (trainX[indices], trainY[indices]), combined[1]

