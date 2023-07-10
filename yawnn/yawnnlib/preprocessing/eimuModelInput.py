from yawnnlib.utils import commons, filters, config
from yawnnlib.preprocessing.modelInput import ModelInput
from yawnnlib.structure.sessionData import SessionData

import numpy as np

TIMESTAMP_PREDICATE = lambda tList: sum(map(lambda t: t.type == 'yawn', tList))

class EimuModelInput(ModelInput):
    def __init__(self, windowSize : float, windowSep : float, dataFilter : filters.DataFilter = filters.NoneFilter(), name : str = "eimuNN") -> None:
        self.windowSize = windowSize
        self.windowSep = windowSep
        self.dataFilter = dataFilter
        self.name = name
    
    def applyModelTransformOnPath(self, path : str, fileNum : int = -1, totalFiles : int = -1) -> commons.AnnotatedData:
        session = SessionData.fromPath(path, fileNum, totalFiles)
        return self._applyModelTransform(session)
    
    def applyModelTransformOnWeightedAnnotatedData(self, hafarData : commons.WeightedAnnotatedData) -> commons.WeightedAnnotatedData:
        # the data is already windowed, so we need only apply the FFT to each
        (data, annotations), weights = hafarData
        transformedData = []
        for window in range(len(data)):
            waData = (data[window], np.array([annotations[window]])), np.array([weights[window]])
            session = SessionData.fromWeightedAnnotatedData(waData, config.get("HAFAR_SAMPLE_RATE"), config.get("EIMU_VERSION"))
            d, _ = self._applyModelTransform(session)
            # annotations ignored as we already have correctly labelled ones from the dataset.
            # there is definitely wasted work going on behind the scenes # todo
            d = np.squeeze(d)
            transformedData.append(d)
            if (window % 1000 == 0):
                print(f"Processing data: {window}/{len(data)}... Expected shape: ({len(data)}, {d.shape})", end='\r')
        fData = np.array(transformedData)
        print(f"\nFinal shape: {fData.shape}, {annotations.shape}\n")
        return (fData, annotations), weights
    
    def _applyModelTransform(self, session : SessionData) -> commons.AnnotatedData:
        """ Converts a session to a tuple of (data, annotations)

        Parameters
        ----------
        session : SessionData
            The session to obtain data from
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
        data, timestamps = session.getEimuData(windowSize=self.windowSize, windowSep=self.windowSep, dataFilter=self.dataFilter)
        annotations = np.array(list(map(TIMESTAMP_PREDICATE, timestamps)))
        annotations.resize(annotations.shape[0], 1)
        
        return data, annotations
    
    def getType(self) -> str:
        return self.name
