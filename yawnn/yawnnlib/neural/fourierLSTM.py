from yawnnlib.utils import commons, filters
from yawnnlib.neural.modelType import ModelType
from yawnnlib.structure.fourierData import FourierData

import numpy as np

TIMESTAMP_PREDICATE = lambda tList: sum(map(lambda t: t.type == 'yawn', tList))

def eimuToFourierLSTMInput(eimuPath : str, dataFilter : filters.DataFilter, chunkSize : float, chunkSeparation : float, fileNum : int = -1, totalFiles : int = -1) -> commons.AnnotatedData:
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
    session = FourierData.fromPath(eimuPath, fileNum=fileNum, totalFiles=totalFiles)
    data, timestamps = session.getFourierData(dataFilter=dataFilter, chunkSize=chunkSize, chunkSeparation=chunkSeparation)
    
    # data format is (chunks, times, frequencies, axes).
    ch, ts, fs, ax = data.shape
    
    # the number of chunks is variable based on input data, the others depend on constants.
    # we can either train on (times, frequencies, axes) tuples via a CNN, or (frequency, axes) tuples via an LSTM.
    # the former will need significantly more data.
    
    # for this LSTM method, we reshape the data to (chunks * times, frequencies, axes), effectively squashing the first two axes.
    
    data = np.reshape(data, (ch * ts, fs, ax))
    annotations = np.array([timestamps[chunk] for chunk in range(ch) for _ in range(ts)])
    annotations.resize(annotations.shape[0], 1)
    
    return data, annotations

class FourierLSTMInput(ModelType):
    def __init__(self, dataFilter : filters.DataFilter = filters.NoneFilter(), chunkSize : float = commons.YAWN_TIME*2, chunkSeparation : float = commons.YAWN_TIME/2) -> None:
        self.dataFilter = dataFilter
        self.chunkSize = chunkSize
        self.chunkSeparation = chunkSeparation
    
    def fromPath(self, path : str, fileNum : int = -1, totalFiles : int = -1) -> commons.AnnotatedData:
        return eimuToFourierLSTMInput(path, self.dataFilter, self.chunkSize, self.chunkSeparation, fileNum, totalFiles)
    
    def getType(self) -> str:
        return 'fourierLSTM'
    