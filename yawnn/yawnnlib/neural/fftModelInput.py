from yawnnlib.utils import commons, filters
from yawnnlib.neural.modelInput import ModelInput
from yawnnlib.structure.fourierData import FourierData

import numpy as np

TIMESTAMP_PREDICATE = lambda tList: sum(map(lambda t: t.type == 'yawn', tList))

def getFFTModelData(eimuPath : str, dataFilter : filters.DataFilter, chunkSize : float, chunkSeparation : float, fileNum : int, totalFiles : int, nOverlap : int, nPerSeg : int) -> commons.AnnotatedData:
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
    data, timestamps = session.getFFTData(dataFilter=dataFilter, chunkSize=chunkSize, chunkSeparation=chunkSeparation)
    
    # the number of chunks is variable based on input data, the others depend on constants.
    # we can either train on (times, frequencies, axes) tuples from the chunks' spectrograms (c.f. spectrogramModelInput.py), 
    # or (frequency, axes) tuples from the FFTs over each chunk (this file).
    
    # data format is (chunks, frequencies, axes).
    ch, fs, ax = data.shape
    
    annotations = np.array([timestamps[chunk] for chunk in range(ch)])
    annotations = np.reshape(annotations, (-1, 1))
    
    assert len(annotations) == len(data)
    
    return data, annotations

class FFTModelInput(ModelInput):
    def __init__(self, dataFilter : filters.DataFilter = filters.NoneFilter(), chunkSize : float = commons.YAWN_TIME*2, chunkSeparation : float = commons.YAWN_TIME/2, nPerSeg : int = 128, nOverlap : int = 96, name : str = "fftNN") -> None:
        self.dataFilter = dataFilter
        self.chunkSize = chunkSize
        self.chunkSeparation = chunkSeparation
        self.nOverlap = nOverlap
        self.nPerSeg = nPerSeg
        self.name = name
    
    def fromPath(self, path : str, fileNum : int = -1, totalFiles : int = -1) -> commons.AnnotatedData:
        return getFFTModelData(path, self.dataFilter, self.chunkSize, self.chunkSeparation, fileNum, totalFiles, self.nOverlap, self.nPerSeg)
    
    def getType(self) -> str:
        return self.name

if __name__ == "__main__":
    from matplotlib import pyplot as plt
    chunkSize = commons.YAWN_TIME*2
    chunkSeparation = commons.YAWN_TIME/2
    nPerSeg = 128
    filt = filters.FilterCollection([filters.LowPassFilter(96, 5), filters.HighPassFilter(96, 0.1), filters.NormalisationFilter()])
    
    data, annotations = FFTModelInput(
        chunkSize=chunkSize,
        chunkSeparation=chunkSeparation,
        nPerSeg=nPerSeg,
        dataFilter=filt,
        # dataFilter=filters.FilterCollection([filters.LowPassFilter(96, 5)])
    ).fromPath("./yawnn/data/tests/96hz/96hz-yawns2.eimu")
    boundary = nPerSeg//2
    trueChunkSize = chunkSize * 96
    trueChunkSeparation = chunkSeparation * 96
    for i in range(10, 20):
        plt.figure(i+1)
        y = data[i,:,:]
        plt.plot(np.arange(len(y)), y)
        plt.title(f"{boundary + trueChunkSeparation * i}, {boundary + trueChunkSeparation * i + trueChunkSize}: {annotations[i]}")
        
    session = FourierData.fromPath("./yawnn/data/tests/96hz/96hz-yawns2.eimu", nPerSeg=nPerSeg)
    session = FourierData.applyFilter(session, filt, filters.ApplyType.SESSION)
    session.plot(show=False, unitConversion=False)
    assert isinstance(session, FourierData)
    session.plotSessionData(show=True)
    