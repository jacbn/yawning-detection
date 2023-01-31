from abc import ABC, abstractmethod
import numpy as np
from os import mkdir
from os.path import exists, basename, normpath
from yawnnlib.utils import commons

class ModelType(ABC):
    """ An abstract class that represents an input to an LSTM model. See eimuLSTM.EimuLSTMInput for an example implementation. """
    @abstractmethod
    def fromPath(self, path : str, fileNum : int = -1, totalFiles : int = -1) -> commons.AnnotatedData:
        pass
    
    def fromPathOrCache(self, path : str, fileNum : int = -1, totalFiles : int = -1) -> commons.AnnotatedData:
        print(f"Converting to LSTM input: {fileNum}/{totalFiles}...", end='\r')
        if (commons.ENABLE_CACHING):
            fileName = basename(normpath(path))
            
            directory = self._getCachePathForFile(fileName)
            
            if (exists(f"{directory}_data.npy") and exists(f"{directory}_anno.npy")):
                # check both in case something went wrong saving
                print(f"Converting to LSTM input: {fileNum}/{totalFiles} (read from cache)")
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
    
    def fromDirectory(self, directoryPath : str, shuffle : bool = True, equalPositiveAndNegative : bool = True, trainSplit : float = commons.TRAIN_SPLIT) -> commons.ModelData:
        """ Convert a directory of .eimu files to a tuple of (trainX, trainY), (testX, testY) using this ModelType. """
        # build a list of tuples of (data, annotations), one tuple for each file in directoryPath
        inputs = commons.mapToDirectory(self.fromPathOrCache, directoryPath) 

        try:
            # combine all the inputs into a single tuple of (data, annotations)
            combinedInputs = np.concatenate(list(map(lambda x: x[0], inputs))), np.concatenate(list(map(lambda x: x[1], inputs)))
            # split the data into training and test sets (the model data); (trainSplit * 100%) of the data is used for training
            trainLength = int(len(combinedInputs[0]) * trainSplit)
            modelData = (combinedInputs[0][:trainLength], combinedInputs[1][:trainLength]), (combinedInputs[0][trainLength:], combinedInputs[1][trainLength:])
        except ValueError:
            raise ValueError(f"Data from directory {directoryPath} could not be combined. Ensure all files use the same sampling rate.")
        
        if equalPositiveAndNegative:
            modelData = commons.equalisePositiveAndNegative(modelData, shuffle)
        if shuffle:
            modelData = commons.shuffleAllData(modelData)
    
        return modelData
    
    def _getCachePathForFile(self, fileName : str) -> str:
        parent = f"{commons.CACHE_DIRECTORY}/{self.getType()}"
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