from typing import TypeVar, Callable
from abc import ABC, abstractmethod
from os import listdir
from os.path import isfile, join
from matplotlib import pyplot as plt
import numpy as np

AXIS_NAMES = [['Accel X', 'Accel Y', 'Accel Z'], ['Gyro X', 'Gyro Y', 'Gyro Z']]
AXIS_COLOURS = plt.rcParams['axes.prop_cycle'].by_key()['color']

YAWN_TIME = 2 # time, in seconds, an individual yawn lasts for
TRAIN_SPLIT = 0.8 # default fraction of data to use for training

T = TypeVar('T')
AnnotatedData = tuple[np.ndarray, np.ndarray] # (data, annotations)
ModelData = tuple[AnnotatedData, AnnotatedData] # one for training, one for testing

class ModelType(ABC):
    """ An abstract class that represents an input to an LSTM model. See eimuLSTM.EimuLSTMInput for an example implementation. """
    @abstractmethod
    def fromPath(self, path : str, fileNum : int = -1, totalFiles : int = -1) -> ModelData:
        pass
    
    @abstractmethod
    def getType(self) -> str:
        pass
    

def directoryToModelData(directoryPath : str, modelType : ModelType, shuffle : bool = True, equalPositiveAndNegative : bool = True, trainSplit : float = TRAIN_SPLIT) -> ModelData:
    """ Convert a directory of .eimu files to a tuple of (trainX, trainY), (testX, testY) using a given LSTMInput. """
    # build a list of tuples of (data, annotations), one tuple for each file in directoryPath
    inputs = mapToDirectory(modelType.fromPath, directoryPath) 

    try:
        # combine all the inputs into a single tuple of (data, annotations)
        combinedInputs = np.concatenate(list(map(lambda x: x[0], inputs))), np.concatenate(list(map(lambda x: x[1], inputs)))
        # split the data into training and test sets (the model data); (trainSplit * 100%) of the data is used for training
        trainLength = int(len(combinedInputs[0]) * trainSplit)
        modelData = (combinedInputs[0][:trainLength], combinedInputs[1][:trainLength]), (combinedInputs[0][trainLength:], combinedInputs[1][trainLength:])
    except ValueError:
        raise ValueError(f"Data from directory {directoryPath} could not be combined. Ensure all files use the same sampling rate.")
    
    if equalPositiveAndNegative:
        modelData = equalisePositiveAndNegative(modelData, shuffle)
    if shuffle:
        modelData = shuffleAllData(modelData)
    
    
    return modelData


def mapToDirectory(f : Callable[[str, int, int], T], path : str) -> list[T]:
    if path[-1] != '/':
        path += '/'
    """ Apply a function to all files in a directory. The function should take input parameters [fileName : str, fileNum : int, totalFiles : int]. """
    files = [join(path, file) for file in listdir(path) if isfile(join(path, file))]
    return [f(file, i+1, len(files)) for i, file in enumerate(files)]


def shuffleAllData(combined : ModelData) -> ModelData:
    """ Shuffles all the data, across both the training and test sets. """
    data = np.concatenate((combined[0][0], combined[1][0]))
    annotations = np.concatenate((combined[0][1], combined[1][1]))
    indices = np.arange(len(data))
    np.random.shuffle(indices)
    
    trainLength = int(len(data) * TRAIN_SPLIT)
    return (data[indices][:trainLength], annotations[indices][:trainLength]), (data[indices][trainLength:], annotations[indices][trainLength:])


def equalisePositiveAndNegative(combined : ModelData, shuffle : bool) -> ModelData:
    """ Equalises the number of positive and negative examples in both the training and test sets (individually). """
    train, test = combined
    return _equalisePNForSingleSet(train, shuffle), _equalisePNForSingleSet(test, shuffle)


def _equalisePNForSingleSet(annotatedData : AnnotatedData, shuffle : bool) -> AnnotatedData:
    data, annotations = annotatedData
    positiveIndices = np.where(annotations == 1)[0]
    negativeIndices = np.where(annotations == 0)[0]
    
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
    
    return data[indices], annotations[indices]