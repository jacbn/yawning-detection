from yawnnlib.utils import commons, config
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
    
    def fromEimuDirectory(self, directoryPath : str) -> list[commons.AnnotatedData]:
        """ Pull all .eimu files from one directory into a list of pairs of (data, annotations).
        Pass result into fromAnnotatedDataList to get a ModelData object. """
        # todo: merge this with fromHafarDirectory (move L2 of fromAnnotatedDataList to before use of this)
        return commons.mapToDirectory(self.fromPathOrCache, directoryPath) 
    
    def fromHafarDirectory(self, directoryPath : str) -> commons.AnnotatedData:
        """ Put all .csv files from the HAFAR dataset into a combined tuple of (data, annotations).
        Pass result into fromCombinedTuple to get a ModelData object. """
        return hafarToEimu.convert(directoryPath)
        
    def fromAnnotatedDataList(self, annotatedDataList : list[commons.AnnotatedData], shuffle : bool = True, equalPositiveAndNegative : bool = True, trainSplit : float = config.get("TRAIN_SPLIT")) -> commons.ModelData:
        try:
            # combine all the inputs into a single tuple of (data, annotations)
            combinedTuple = np.concatenate(list(map(lambda x: x[0], annotatedDataList))), np.concatenate(list(map(lambda x: x[1], annotatedDataList)))
            return self.fromCombinedTuple(combinedTuple, shuffle, equalPositiveAndNegative, trainSplit)
        except ValueError as e:
            raise ValueError(f"Data could not be combined. Ensure all files use the same sampling rate.", e)
    
    def fromCombinedTuple(self, combinedTuple : commons.AnnotatedData, shuffle : bool = True, equalPositiveAndNegative : bool = True, trainSplit : float = config.get("TRAIN_SPLIT")) -> commons.ModelData:
        """ Converts a tuple of (data, annotations) into a ModelData object. """
        # split the data into training and test sets (the model data); (trainSplit * 100%) of the data is used for training
        trainLength = int(len(combinedTuple[0]) * trainSplit)
        modelData = (combinedTuple[0][:trainLength], combinedTuple[1][:trainLength]), (combinedTuple[0][trainLength:], combinedTuple[1][trainLength:])
        
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