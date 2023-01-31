from yawnnlib.utils import commons, filters
from yawnnlib.lstm.modelType import ModelType
from yawnnlib.structure.fourierData import FourierData

import numpy as np

TIMESTAMP_PREDICATE = lambda tList: sum(map(lambda t: t.type == 'yawn', tList))

def eimuToFourierLSTMInput(eimuPath : str, dataFilter : filters.DataFilter, trainOnTimeAxis=False, fileNum : int = -1, totalFiles : int = -1) -> commons.AnnotatedData:
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
    data, timestamps = session.getFourierData(dataFilter=dataFilter, chunkSize=commons.YAWN_TIME, chunkSeparation=commons.YAWN_TIME/4)
    
    # data format is (axes, chunks, frequencies, times (samples) per chunk).
    ax, ch, fs, ts = data.shape
    # the number of chunks is variable based on input data, the others depend on constants.
    # we can either train on (axes, frequency, time) tuples via a CNN, or (axes, frequency) tuples via an LSTM.
    # the former will will need significantly more data.
    
    # for first method, we reshape the data to (chunks, axes, frequencies, times), iterating through will give the tuples.
    # for the second, we reshape the data to (chunks, times, axes, frequencies), then squash the first two axes.
    
    if trainOnTimeAxis:
        data = np.reshape(data, (ch, ax, fs, ts))
        annotations = np.array(timestamps)
        # todo: resize
    else:
        data = np.reshape(data, (ch * ts, ax, fs))
        annotations = np.array([timestamps[chunk] for chunk in range(ch) for _ in range(ts)])
        annotations.resize(annotations.shape[0], 1)
    
    return data, annotations

class FourierLSTMInput(ModelType):
    def __init__(self, dataFilter : filters.DataFilter = filters.NoneFilter(), trainOnTimeAxis=False) -> None:
        self.dataFilter = dataFilter
        self.trainOnTimeAxis = trainOnTimeAxis
    
    def fromPath(self, path : str, fileNum : int = -1, totalFiles : int = -1) -> commons.AnnotatedData:
        return eimuToFourierLSTMInput(path, self.dataFilter, self.trainOnTimeAxis, fileNum, totalFiles)
    
    def getType(self) -> str:
        return 'fourierLSTM'

    def _toCache(self) -> None:
        return super()._toCache()
    
    def _fromCache(self) -> None:
        return super()._fromCache()