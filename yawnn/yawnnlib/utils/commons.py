from yawnnlib.utils import config

from typing import TypeVar, Callable
from abc import ABC, abstractmethod
from os import listdir
from os.path import isfile, join, abspath
from matplotlib import pyplot as plt
import numpy as np

AXIS_NAMES = [['Accel X', 'Accel Y', 'Accel Z'], ['Gyro X', 'Gyro Y', 'Gyro Z']]
AXIS_COLOURS = plt.rcParams['axes.prop_cycle'].by_key()['color']

# while the user-defined values for these are in the config, we need to be able to modify them for testing purposes,
# so uses in non-entry-point code will refer to these instead of the config values
ENABLE_CACHING = config.get("ENABLE_CACHING")
PROJECT_ROOT = config.PROJECT_ROOT
YAWN_TIME = config.get("YAWN_TIME")
YAWN_CORRECTION = config.get("YAWN_CORRECTION")


T = TypeVar('T')
AnnotatedData = tuple[np.ndarray, np.ndarray] # (data, annotations)
ModelData = tuple[AnnotatedData, AnnotatedData] # one for training, one for testing


def mapToDirectory(f : Callable[[str, int, int], T], path : str) -> list[T]:
    if path[-1] != '/':
        path += '/'
    """ Apply a function to all files in a directory. The function should take input parameters [fileName : str, fileNum : int, totalFiles : int]. """
    files = [join(path, file) for file in listdir(path) if isfile(join(path, file))]
    return [f(file, i+1, len(files)) for i, file in enumerate(files)]


def shuffleAllData(combined : ModelData, trainSplit : float) -> ModelData:
    """ Shuffles all the data, across both the training and test sets. """
    data = np.concatenate((combined[0][0], combined[1][0]))
    annotations = np.concatenate((combined[0][1], combined[1][1]))
    assert len(data) == len(annotations)
    indices = np.arange(len(data))
    np.random.shuffle(indices)
    
    trainLength = int(len(data) * trainSplit)
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

def timeDistributeData(data : ModelData) -> ModelData:
    ((trainX, trainY), (testX, testY)) = data
    return (timeDistributeAnnotatedData((trainX, trainY)), timeDistributeAnnotatedData((testX, testY)))

def timeDistributeAnnotatedData(annotatedData : AnnotatedData) -> AnnotatedData:
    data, annotations = annotatedData
    distribution = config.get("TIME_DISTRIBUTION")
    newData = np.zeros((data.shape[0] // distribution, distribution, *data.shape[1:]))
    newAnnotations = np.zeros((annotations.shape[0] // distribution))
    i = 0
    j = 0
    for sample in data:
        if j >= newData.shape[0]:
            print("WARN: Data was not divisible by distribution factor. Some data was lost.")
            continue
        newData[j][i] = sample
        newAnnotations[j] = max(newAnnotations[j], annotations[j * distribution + i])
        i += 1
        if i >= distribution:
            i = 0
            j += 1
    return (newData, newAnnotations)
            