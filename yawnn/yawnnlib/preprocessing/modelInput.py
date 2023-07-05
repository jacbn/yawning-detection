from yawnnlib.utils import commons, config
from yawnnlib.preprocessing.modelData import ModelData
import tools.hafarToEimu as hafarToEimu

import numpy as np
from abc import ABC, abstractmethod
from os import mkdir
from os.path import exists, basename, normpath

class ModelInput(ABC):
    """ An abstract class that represents an input to a NN model. """
    # todo: make private
    @abstractmethod
    def fromPath(self, path : str, fileNum : int = -1, totalFiles : int = -1) -> commons.AnnotatedData:
        pass
    
    # todo: as above
    def fromPathOrCache(self, path : str, fileNum : int = -1, totalFiles : int = -1) -> commons.AnnotatedData:
        print(f"Processing data: {fileNum}/{totalFiles}...", end='\r')
        if (config.get("ENABLE_CACHING")):
            fileName = basename(normpath(path))
            
            directory = self._getCachePathForFile(fileName)
            
            if (exists(f"{directory}_data.npy") and exists(f"{directory}_anno.npy")):
                # check both in case something went wrong saving
                print(f"Processing data: {fileNum}/{totalFiles} (read from cache)")
                return self._fromCache(directory)
            else:
                print()
                data, annotations = self.fromPath(path, fileNum, totalFiles)
                self._toCache(directory, data, annotations)
                return data, annotations
        print()
        return self.fromPath(path, fileNum, totalFiles)
    
    @abstractmethod
    def getType(self) -> str:
        pass
    
    def fromEimuDirectory(self, directoryPath : str, trainSplit : float = config.get("TRAIN_SPLIT"), equalPositiveAndNegative=True, shuffle=True) -> ModelData:
        """ Pull all .eimu files from one directory into a list of pairs of (data, annotations).
        Pass result into fromAnnotatedDataList to get a ModelData object. """
        # todo: update descriptions
        # todo: merge this with fromHafarDirectory (move L2 of fromAnnotatedDataList to before use of this)
        combinedList = commons.mapToDirectory(self.fromPathOrCache, directoryPath)
        return ModelData.fromAnnotatedDataList(combinedList, None, 96, trainSplit=trainSplit, equalPositiveAndNegative=equalPositiveAndNegative, shuffle=shuffle)
    
    def fromHafarDirectory(self, directoryPath : str, trainSplit : float = config.get("TRAIN_SPLIT"), isTrain=True, equalPositiveAndNegative=True, shuffle=True) -> ModelData:
        """ Put all .csv files from the HAFAR dataset into a combined tuple of (data, annotations).
        Pass result into fromCombinedTuple to get a ModelData object. """
        annotatedData, weights = hafarToEimu.convert(
            directoryPath, 
            specificUsers=config.get("HAFAR_USERS"),
            poiUsers=config.get("HAFAR_POI_USERS"),
            poiTrainSplit=config.get("HAFAR_POI_TRAIN_SPLIT"),
            isTrain=isTrain
        )
        return ModelData.fromCombinedTuple(annotatedData, weights, config.get("HAFAR_SAMPLE_RATE"), trainSplit=trainSplit, equalPositiveAndNegative=equalPositiveAndNegative, shuffle=shuffle)
    
    def _getCachePathForFile(self, fileName : str) -> str:
        cacheDirectory = config.get("CACHE_DIRECTORY")
        parent = f"{cacheDirectory}/{self.getType()}"
        if not exists(parent):
            mkdir(parent)
        return f"{parent}/{fileName}"
    
    @staticmethod
    def _toCache(directory : str, data : np.ndarray, annotations : np.ndarray) -> None:
        np.save(f"{directory}_data.npy", data)
        np.save(f"{directory}_anno.npy", annotations)
    
    @staticmethod
    def _fromCache(directory : str) -> commons.AnnotatedData:
        data = np.load(f"{directory}_data.npy")
        annotations = np.load(f"{directory}_anno.npy")
        return data, annotations