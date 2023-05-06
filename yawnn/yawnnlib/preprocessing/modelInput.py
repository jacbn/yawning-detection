from yawnnlib.utils import commons, config

from abc import ABC, abstractmethod
import numpy as np
from os import mkdir
from os.path import exists, basename, normpath

ENABLE_CACHING = config.get("ENABLE_CACHING")

class ModelInput(ABC):
    """ An abstract class that represents an input to a NN model. """
    @abstractmethod
    def fromPath(self, path : str, fileNum : int = -1, totalFiles : int = -1) -> commons.AnnotatedData:
        pass
    
    def fromPathOrCache(self, path : str, fileNum : int = -1, totalFiles : int = -1) -> commons.AnnotatedData:
        print(f"Processing data: {fileNum}/{totalFiles}...", end='\r')
        if (ENABLE_CACHING):
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
    
    def fromDirectory(self, directoryPath : str) -> list[commons.AnnotatedData]:
        """ Pull all .eimu files from one directory into a list of pairs of (data, annotations).
        Pass result into fromAnnotatedDataList to get a ModelData object. """
        return commons.mapToDirectory(self.fromPathOrCache, directoryPath) 
        
    def fromAnnotatedDataList(self, annotatedDataList : list[commons.AnnotatedData], shuffle : bool = True, equalPositiveAndNegative : bool = True, trainSplit : float = config.get("TRAIN_SPLIT")) -> commons.ModelData:
        try:
            # combine all the inputs into a single tuple of (data, annotations)
            combinedInputs = np.concatenate(list(map(lambda x: x[0], annotatedDataList))), np.concatenate(list(map(lambda x: x[1], annotatedDataList)))

            # split the data into training and test sets (the model data); (trainSplit * 100%) of the data is used for training
            trainLength = int(len(combinedInputs[0]) * trainSplit)
            modelData = (combinedInputs[0][:trainLength], combinedInputs[1][:trainLength]), (combinedInputs[0][trainLength:], combinedInputs[1][trainLength:])
        except ValueError as e:
            raise ValueError(f"Data could not be combined. Ensure all files use the same sampling rate.", e)
        
        if equalPositiveAndNegative:
            modelData = commons.equalisePositiveAndNegative(modelData, shuffle)
        if shuffle:
            modelData = commons.shuffleAllData(modelData, trainSplit)
    
        return modelData
    
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