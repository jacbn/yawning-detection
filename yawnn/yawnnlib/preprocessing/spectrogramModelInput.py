from yawnnlib.utils import commons, filters, config
from yawnnlib.preprocessing.modelInput import ModelInput
from yawnnlib.structure.fourierData import FourierData

import numpy as np

TIMESTAMP_PREDICATE = lambda tList: sum(map(lambda t: t.type == 'yawn', tList))

class SpectrogramModelInput(ModelInput):
    def __init__(self, dataFilter : filters.DataFilter = filters.NoneFilter(), windowSize : float = commons.YAWN_TIME*2, windowSep : float = commons.YAWN_TIME/2, nPerSeg : int = 128, nOverlap : int = 96, name : str = "spectrogramNN") -> None:
        self.dataFilter = dataFilter
        self.windowSize = windowSize
        self.windowSep = windowSep
        self.nOverlap = nOverlap
        self.nPerSeg = nPerSeg
        self.name = name
    
    def applyModelTransformOnPath(self, path : str, fileNum : int = -1, totalFiles : int = -1) -> commons.AnnotatedData:
        session = FourierData.fromPath(path, fileNum=fileNum, totalFiles=totalFiles, nOverlap=self.nOverlap, nPerSeg=self.nPerSeg)
        return self._applyModelTransform(session)
    
    def applyModelTransformOnWeightedAnnotatedData(self, hafarData : commons.WeightedAnnotatedData) -> commons.WeightedAnnotatedData:
        session = FourierData.fromWeightedAnnotatedData(hafarData, config.get("HAFAR_SAMPLE_RATE"), config.get("EIMU_VERSION"), nOverlap=self.nOverlap, nPerSeg=self.nPerSeg) 
        return self._applyModelTransform(session), np.array(session.weights)
    
    def _applyModelTransform(self, session : FourierData) -> commons.AnnotatedData:
        """ Applies Fourier methods to a session to generate a tuple of (data, annotations).

        Parameters
        ----------
        session : SessionData
            The session to obtain data from
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
        data, timestamps = session.getSpectrogramData(dataFilter=self.dataFilter, windowSize=self.windowSize, windowSep=self.windowSep)
        
        # the number of windows is variable based on input data, the others depend on constants.
        # we can either train on (times, frequencies, axes) tuples from the windows' spectrograms (this file), 
        # or (frequency, axes) tuples from the FFTs over each window (c.f. fftModelInput.py).
        
        # for this method, we keep the data in the format (windows, times, frequencies, axes); iterating through will give the tuples.
        
        print(f"============== {data.shape} ================ ")
        
        annotations = np.array(timestamps)
        annotations = np.reshape(annotations, (data.shape[0], 1))

        return data, annotations
    
    def getType(self) -> str:
        return self.name
    