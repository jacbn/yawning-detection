from yawnnlib.utils import config

from typing import TypeVar, Callable, Union
from abc import ABC, abstractmethod
from os import listdir
from os.path import isfile, join, abspath
from matplotlib import pyplot as plt
import numpy as np

AXIS_NAMES = [['Accel X', 'Accel Y', 'Accel Z'], ['Gyro X', 'Gyro Y', 'Gyro Z']]
AXIS_COLOURS = plt.rcParams['axes.prop_cycle'].by_key()['color']

PROJECT_ROOT = config.PROJECT_ROOT
YAWN_TIME = config.get("YAWN_TIME")
YAWN_CORRECTION = config.get("YAWN_CORRECTION")
TRAIN_SPLIT = config.get("TRAIN_SPLIT")
VALIDATION_SPLIT = config.get("VALIDATION_SPLIT")

T = TypeVar('T')
AnnotatedData = tuple[np.ndarray, np.ndarray] # (data, annotations)
TrainTestData = tuple[AnnotatedData, AnnotatedData] # (train, test)
ValidatedModelData = tuple[AnnotatedData, AnnotatedData, AnnotatedData] # (train, val, test)
WeightedAnnotatedData = tuple[AnnotatedData, np.ndarray] # (data, weights)
SampleWeights = Union[np.ndarray, None]

def mapToDirectory(f : Callable[[str, int, int], T], path : str) -> list[T]:
    if path[-1] != '/':
        path += '/'
    """ Apply a function to all files in a directory. The function should take input parameters [fileName : str, fileNum : int, totalFiles : int]. """
    files = [join(path, file) for file in listdir(path) if isfile(join(path, file))]
    return [f(file, i+1, len(files)) for i, file in enumerate(files)]

def timeDistributeData(data : Union[TrainTestData, ValidatedModelData]):
    return tuple([timeDistributeAnnotatedData(x) for x in data])

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

def splitTrainingData(data : AnnotatedData, modelNum : int, totalModels : int):
    """ Splits the training data into training and validation sets, based on the model number and total number of models.

    Parameters
    ----------
    data : AnnotatedData
        The training data to split.
    modelNum : int
        The model index (starting from 0).
    totalModels : int
        The total number of models to train.

    Returns
    -------
    (train, val), trainIndices : tuple[tuple[AnnotatedData, AnnotatedData], np.ndarray]
        The training and validation sets, and the indices of the training set.
        
    """
    if (totalModels == 1):
        valStart = 0 
    else:
        valStart = int(len(data[0]) * (TRAIN_SPLIT - VALIDATION_SPLIT) * modelNum / (totalModels - 1))
    valEnd = valStart + int(len(data[0]) * VALIDATION_SPLIT)
    
    # train = (np.concatenate((data[0][:valStart], data[0][valEnd:])), np.concatenate((data[1][:valStart], data[1][valEnd:])))
    trainIndices = np.concatenate((np.arange(0, valStart), np.arange(valEnd, len(data[0]))))
    train = (data[0][trainIndices], data[1][trainIndices])
    val = (data[0][valStart:valEnd], data[1][valStart:valEnd])
    
    return (train, val), trainIndices
