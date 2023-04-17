from yawnnlib.utils import commons, filters
from yawnnlib.neural.modelInput import ModelInput
from yawnnlib.structure.fourierData import FourierData

import numpy as np

TIMESTAMP_PREDICATE = lambda tList: sum(map(lambda t: t.type == 'yawn', tList))

def getSpectrogramModelData(eimuPath : str, dataFilter : filters.DataFilter, windowSize : float, windowSep : float, fileNum : int, totalFiles : int, nOverlap : int, nPerSeg : int) -> commons.AnnotatedData:
    """ Applies Fourier methods to a .eimu file to generate a tuple of (data, annotations).

    Parameters
    ----------
    eimuPath : str
        The path to the .eimu file.
    dataFilter : filters.DataFilter
        The filter to apply to the data.
    trainOnTimeAxis : bool, optional
        Whether to include the time axis as training data, by default False
    fileNum : int, optional
        The current file number, by default -1
    totalFiles : int, optional
        The total number of files, by default -1

    Returns
    -------
    commons.AnnotatedData
        A tuple of (data, annotations)
    """
    session = FourierData.fromPath(eimuPath, fileNum=fileNum, totalFiles=totalFiles, nOverlap=nOverlap, nPerSeg=nPerSeg)
    data, timestamps = session.getSpectrogramData(dataFilter=dataFilter, windowSize=windowSize, windowSep=windowSep)
    
    # the number of windows is variable based on input data, the others depend on constants.
    # we can either train on (times, frequencies, axes) tuples from the windows' spectrograms (this file), 
    # or (frequency, axes) tuples from the FFTs over each window (c.f. fftModelInput.py).
    
    # for this method, we keep the data in the format (windows, times, frequencies, axes); iterating through will give the tuples.
    
    annotations = np.array(timestamps)
    annotations = np.reshape(annotations, (data.shape[0], 1))

    return data, annotations

class SpectrogramModelInput(ModelInput):
    def __init__(self, dataFilter : filters.DataFilter = filters.NoneFilter(), windowSize : float = commons.YAWN_TIME*2, windowSep : float = commons.YAWN_TIME/2, nPerSeg : int = 128, nOverlap : int = 96, name : str = "spectrogramNN") -> None:
        self.dataFilter = dataFilter
        self.windowSize = windowSize
        self.windowSep = windowSep
        self.nOverlap = nOverlap
        self.nPerSeg = nPerSeg
        self.name = name
    
    def fromPath(self, path : str, fileNum : int = -1, totalFiles : int = -1) -> commons.AnnotatedData:
        return getSpectrogramModelData(path, self.dataFilter, self.windowSize, self.windowSep, fileNum, totalFiles, self.nOverlap, self.nPerSeg)
    
    def getType(self) -> str:
        return self.name
    