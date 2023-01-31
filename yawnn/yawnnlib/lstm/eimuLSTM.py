from yawnnlib.utils import commons, filters
from yawnnlib.structure.sessionData import SessionData

import numpy as np

TIMESTAMP_PREDICATE = lambda tList: sum(map(lambda t: t.type == 'yawn', tList))

def eimuToLSTMInput(eimuPath : str, dataFilter : filters.DataFilter, sessionGap : int = 3, fileNum : int = -1, totalFiles : int = -1) -> commons.AnnotatedData:
    """ Converts a single .eimu file to a tuple of (data, annotations)

    Parameters
    ----------
    eimuPath : str
        The path to the .eimu file
    dataFilter : filters.DataFilter
        The filter to apply to the data
    sessionGap : int, optional
        The gap (measured in number of samples) between each split, by default 3
    fileNum : int, optional
        The current file number, by default -1
    totalFiles : int, optional
        The total number of files, by default -1

    Returns
    -------
    commons.AnnotatedData
        A tuple of (data, annotations)
    """
    print(f"Converting to LSTM input: {fileNum}/{totalFiles}...")
    session = SessionData.fromPath(eimuPath, fileNum, totalFiles)
    data, timestamps = session.getEimuData(dataFilter=dataFilter, sessionGap=sessionGap)
    annotations = np.array(list(map(TIMESTAMP_PREDICATE, timestamps)))
    annotations.resize(annotations.shape[0], 1)
    
    return data, annotations

class EimuLSTMInput(commons.ModelType):
    def __init__(self, dataFilter : filters.DataFilter = filters.NoneFilter(), sessionGap : int = 3) -> None:
        self.dataFilter = dataFilter
        self.sessionGap = sessionGap
    
    def fromPath(self, path : str, fileNum : int = -1, totalFiles : int = -1) -> commons.AnnotatedData:
        return eimuToLSTMInput(path, self.dataFilter, self.sessionGap, fileNum, totalFiles)
    
    def getType(self) -> str:
        return 'eimuLSTM'
    
    def toCache(self) -> None:
        return super().toCache()
    
    def fromCache(self) -> None:
        return super().fromCache()
