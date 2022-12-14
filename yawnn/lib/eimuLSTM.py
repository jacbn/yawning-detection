import commons
from eimuReader import SessionData

import numpy as np

TIMESTAMP_PREDICATE = lambda tList: sum(map(lambda t: t.type == 'yawn', tList))

# convert a single eimu file to a tuple of (trainX, trainY), (testX, testY)
def eimuToLSTMInput(eimuPath : str, fileNum : int = -1, totalFiles : int = -1) -> commons.ModelData:
    session = SessionData.fromPath(eimuPath, fileNum, totalFiles)
    data, timestamps = session.getEimuData()
    annotations = np.array(list(map(TIMESTAMP_PREDICATE, timestamps)))
    annotations.resize(annotations.shape[0], 1)

    trainLength = int(len(data) * commons.TRAIN_PERCENT)
    return (data[:trainLength], annotations[:trainLength]), (data[trainLength:], annotations[trainLength:])

class EimuLSTMInput(commons.ModelType):
    def __init__(self, trainOnTimeAxis=False):
        self.trainOnTimeAxis = trainOnTimeAxis
    
    def fromPath(self, path : str, fileNum : int = -1, totalFiles : int = -1) -> commons.ModelData:
        return eimuToLSTMInput(path, fileNum, totalFiles)
    
    def getType(self) -> str:
        return 'eimuLSTM'
