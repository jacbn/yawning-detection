from abc import ABC, abstractmethod
import numpy as np
from yawnnlib.utils import commons

class ModelType(ABC):
    """ An abstract class that represents an input to an LSTM model. See eimuLSTM.EimuLSTMInput for an example implementation. """
    @abstractmethod
    def fromPath(self, path : str, fileNum : int = -1, totalFiles : int = -1) -> commons.ModelData:
        pass
    
    @abstractmethod
    def getType(self) -> str:
        pass
    
    def fromDirectory(self, directoryPath : str, shuffle : bool = True, equalPositiveAndNegative : bool = True, trainSplit : float = commons.TRAIN_SPLIT) -> commons.ModelData:
        """ Convert a directory of .eimu files to a tuple of (trainX, trainY), (testX, testY) using this ModelType. """
        # build a list of tuples of (data, annotations), one tuple for each file in directoryPath
        inputs = commons.mapToDirectory(self.fromPath, directoryPath) 

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
    
    @abstractmethod
    def _toCache(self) -> None:
        pass 
    
    @abstractmethod
    def _fromCache(self) -> None:
        pass