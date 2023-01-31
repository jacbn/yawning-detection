from typing import TypeVar, Callable
from abc import ABC, abstractmethod
from os import listdir
from os.path import isfile, join, abspath
from matplotlib import pyplot as plt
import numpy as np

# todo: move user settings to a config file

AXIS_NAMES = [['Accel X', 'Accel Y', 'Accel Z'], ['Gyro X', 'Gyro Y', 'Gyro Z']]
AXIS_COLOURS = plt.rcParams['axes.prop_cycle'].by_key()['color']

PROJECT_ROOT = abspath(join(__file__, '../../..')) # yawning-detection/yawnn/
CACHE_DIRECTORY = f'{PROJECT_ROOT}/data/.preprocessing_cache'
ENABLE_CACHING = True

YAWN_TIME = 2 # time, in seconds, an individual yawn lasts for
TRAIN_SPLIT = 0.8 # default fraction of data to use for training

T = TypeVar('T')
AnnotatedData = tuple[np.ndarray, np.ndarray] # (data, annotations)
ModelData = tuple[AnnotatedData, AnnotatedData] # one for training, one for testing


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

def getCacheDirectory() -> str:
    """ Returns the directory to use for caching. """
    listdir(CACHE_DIRECTORY)
    return CACHE_DIRECTORY

getCacheDirectory()