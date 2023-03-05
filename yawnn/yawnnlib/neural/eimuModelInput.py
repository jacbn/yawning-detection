from yawnnlib.utils import commons, filters
from yawnnlib.neural.modelInput import ModelInput
from yawnnlib.structure.sessionData import SessionData

import numpy as np

TIMESTAMP_PREDICATE = lambda tList: sum(map(lambda t: t.type == 'yawn', tList))

def getEimuModelData(eimuPath : str, windowSize : float, windowSep : float, dataFilter : filters.DataFilter, fileNum : int = -1, totalFiles : int = -1) -> commons.AnnotatedData:
    """ Converts a single .eimu file to a tuple of (data, annotations)

    Parameters
    ----------
    eimuPath : str
        The path to the .eimu file
    dataFilter : filters.DataFilter
        The filter to apply to the data
    windowSize : float
        The width (in seconds) of each split
    windowSep : float
        The gap (in seconds) between each split
    fileNum : int, optional
        The current file number, by default -1
    totalFiles : int, optional
        The total number of files, by default -1

    Returns
    -------
    commons.AnnotatedData
        A tuple of (data, annotations)
    """
    session = SessionData.fromPath(eimuPath, fileNum, totalFiles)
    data, timestamps = session.getEimuData(windowSize=windowSize, windowSep=windowSep, dataFilter=dataFilter)
    annotations = np.array(list(map(TIMESTAMP_PREDICATE, timestamps)))
    annotations.resize(annotations.shape[0], 1)
    
    return data, annotations

class EimuModelInput(ModelInput):
    def __init__(self, windowSize : float, windowSep : float, dataFilter : filters.DataFilter = filters.NoneFilter(), name : str = "eimuNN") -> None:
        self.windowSize = windowSize
        self.windowSep = windowSep
        self.dataFilter = dataFilter
        self.name = name
    
    def fromPath(self, path : str, fileNum : int = -1, totalFiles : int = -1) -> commons.AnnotatedData:
        return getEimuModelData(path, self.windowSize, self.windowSep, self.dataFilter, fileNum, totalFiles)
    
    def getType(self) -> str:
        return self.name
