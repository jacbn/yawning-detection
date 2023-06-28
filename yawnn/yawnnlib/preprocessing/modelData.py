from yawnnlib.utils import commons, config
import numpy as np

from typing import Union

currentTrainTestData = []
# todo: make config params
equalPositiveAndNegative = True
shuffle = True

class ModelData:
    def __init__(self, data : commons.TrainTestData, sampleWeights : commons.SampleWeights, sampleRate : int):
        """ A class that represents the data for a model. """
        self.train = data[0]
        self.test = data[1]
        self.sampleWeights = sampleWeights
        self.sampleRate = sampleRate
        
    @classmethod
    def fromWeightedAnnotatedData(cls, annotatedData : commons.AnnotatedData, sampleWeights : commons.SampleWeights, sampleRate : int, trainSplit : float = config.get("TRAIN_SPLIT"), equalPositiveAndNegative : bool = True, shuffle : bool = True):
        """ Creates a ModelData object from annotated data (i.e. windows and timestamps only) and sample weights. """
        return cls.fromCombinedTuple(annotatedData, sampleWeights, sampleRate)
    
    @classmethod
    def fromAnnotatedDataList(cls, annotatedDataList : list[commons.AnnotatedData], sampleWeights : commons.SampleWeights, sampleRate : int, trainSplit : float = config.get("TRAIN_SPLIT"), equalPositiveAndNegative : bool = True, shuffle : bool = True):
        try:
            # combine all the inputs into a single tuple of (data, annotations)
            combinedTuple = np.concatenate(list(map(lambda x: x[0], annotatedDataList))), np.concatenate(list(map(lambda x: x[1], annotatedDataList)))
            return cls.fromCombinedTuple(combinedTuple, sampleWeights, sampleRate) # type: ignore (union types don't type to themselves? :( )
        except ValueError as e:
            raise ValueError(f"Data could not be combined. Ensure all files use the same sampling rate.", e)
    
    @classmethod
    def fromCombinedTuple(cls, combinedTuple : commons.AnnotatedData, sampleWeights : commons.SampleWeights, sampleRate : int, trainSplit : float = config.get("TRAIN_SPLIT"), equalPositiveAndNegative : bool = True, shuffle : bool = True):
        """ Converts a tuple of (data, annotations) into a ModelData object. """
        # split the data into training and test sets (the model data); (trainSplit * 100%) of the data is used for training
        trainLength = int(len(combinedTuple[0]) * trainSplit)
        trainTestData = (combinedTuple[0][:trainLength], combinedTuple[1][:trainLength]), (combinedTuple[0][trainLength:], combinedTuple[1][trainLength:])
        
        modelData = cls(trainTestData, sampleWeights, sampleRate)
        
        if equalPositiveAndNegative:
            modelData.equalisePositiveAndNegative(shuffle)
        if shuffle:
            modelData.shuffleAllData(trainSplit)
    
        return modelData
    
    def splitValidationFromTrainTest(self, modelNum : int = 0, totalModels : int = 1) -> tuple[commons.ValidatedModelData, commons.SampleWeights]:
        """ Gets the training, validation and test data from combined annotated data (c.f. modelInput.fromAnnotatedDataList ).

        Parameters
        ----------
        modelType : commons.ModelType
            The type of model to train.
        annotatedData : list[commons.AnnotatedData]
            The annotated data to use.
        shuffle : bool
            Whether to shuffle the data before training.
        equalPositiveAndNegative : bool
            Whether to equalize the number of positive and negative samples before training.
        modelNum : int
            Used when training multiple models on the same data. The number of the current model.
        totalModels : int
            Used when training multiple models on the same data. The total number of models.

        Returns
        -------
        commons.ModelData
            The training, validation and test data, as a tuple of ((trainX, trainY), (valX, valY), (testX, testY)).
        np.ndarray
            The sample weights for the training data.
        """
        global currentTrainTestData # needed as this state must be preserved between calls
        
        if (modelNum == 0):
            currentTrainTestData = (self.train, self.test)
        
        assert currentTrainTestData != [], "Model number incorrect. Models must start from modelNum=0."
        
        (allTrainX, allTrainY), (testX, testY) = currentTrainTestData
        ((trainX, trainY), (valX, valY)), trainIndices = commons.splitTrainingData((allTrainX, allTrainY), modelNum, totalModels)
        
        weights = self.sampleWeights[trainIndices] if self.sampleWeights is not None else None
        
        return ((trainX, trainY), (valX, valY), (testX, testY)), weights
     
    def shuffleAllData(self, trainSplit : float) -> None:
        """ Shuffles all the data, across both the training and test sets. """
        data = np.concatenate((self.train[0], self.test[0]))
        annotations = np.concatenate((self.train[1], self.test[1]))
        assert len(data) == len(annotations)
        
        indices = np.arange(len(data))
        np.random.shuffle(indices)
        
        trainLength = int(len(data) * trainSplit)
        self.train = (data[indices][:trainLength], annotations[indices][:trainLength])
        self.test  = (data[indices][trainLength:], annotations[indices][trainLength:])
        if self.sampleWeights is not None:
            self.sampleWeights = self.sampleWeights[indices]

    def equalisePositiveAndNegative(self, shuffle : bool) -> None:
        """ Equalises the number of positive and negative examples in both the training and test sets (individually). """
        self.train, trainIndices = self._equalisePNForSingleSet(self.train, shuffle)
        self.test, testIndices = self._equalisePNForSingleSet(self.test, shuffle)
        
        if self.sampleWeights is not None:
            self.sampleWeights = self.sampleWeights[np.concatenate([trainIndices, testIndices])]

    @staticmethod
    def _equalisePNForSingleSet(annotatedData : commons.AnnotatedData, shuffle : bool):
        data, annotations = annotatedData
        positiveIndices = np.where(annotations == 1)[0]
        negativeIndices = np.where(annotations == 0)[0]
        
        np.random.shuffle(positiveIndices) # shuffle the indices so we don't always remove the last ones
        np.random.shuffle(negativeIndices)
        
        if len(positiveIndices) > len(negativeIndices):
            positiveIndices = positiveIndices[:len(negativeIndices)]
        elif len(negativeIndices) > len(positiveIndices):
            negativeIndices = negativeIndices[:len(positiveIndices)]
        
        indices = np.concatenate((positiveIndices, negativeIndices))
        if not shuffle:
            # if we're not going to shuffle later, need to sort the indices back into the original order
            indices = np.sort(indices)
        
        return (data[indices], annotations[indices]), indices
