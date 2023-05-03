from yawnnlib.utils import config

import numpy as np

TRAIN_SPLIT = config.get("TRAIN_SPLIT")
VALIDATION_SPLIT = config.get("VALIDATION_SPLIT")

def splitTrainingData(data : tuple[np.ndarray, np.ndarray], modelNum : int, totalModels : int):
    if (totalModels == 1):
        valStart = 0 
    else:
        valStart = int(len(data[0]) * (TRAIN_SPLIT - VALIDATION_SPLIT) * modelNum / (totalModels - 1))
    valEnd = valStart + int(len(data[0]) * VALIDATION_SPLIT)
    
    train = (np.concatenate((data[0][:valStart], data[0][valEnd:])), np.concatenate((data[1][:valStart], data[1][valEnd:])))
    val = (data[0][valStart:valEnd], data[1][valStart:valEnd])
    
    return (train, val)
