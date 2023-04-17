print("Loading imports...")

from yawnnlib.utils import commons, config
from yawnnlib.neural.modelInput import ModelInput
from yawnnlib.other_classifiers.alternative_classifiers import AlternativeClassifier
import tools.eimuResampler as eimuResampler

from os import listdir, mkdir, path
import numpy as np
import tensorflow as tf
import pickle 

print("Imports loaded.")

MODELS_PATH = config.get("MODELS_PATH")

def makeSequentialModel(layers : list, compile_ : bool = True, learningRate : float = 0.0003) -> tf.keras.models.Sequential:
    """ Creates a sequential model from a list of layers.
    
    Attributes
    ----------
    layers : list
        A list of layers to add to the model.
        
    Returns
    -------
    model : tensorflow.keras.models.Sequential
        The sequential model.
    
    """
    model = tf.keras.models.Sequential()
    for layer in layers:
        model.add(layer)
    if compile_:
        optimizer = tf.keras.optimizers.Adam(learning_rate=learningRate)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

def getTrainTestData(modelType : ModelInput, annotatedData : list[commons.AnnotatedData], shuffle : bool = True, equalPositiveAndNegative : bool = True) -> commons.ModelData:
    """ Gets the training and test data from a list of annotated data.

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

    Returns
    -------
    commons.ModelData
        The training and test data, as a tuple of ((trainX, trainY), (testX, testY)).
    """
    return modelType.fromAnnotatedDataList(annotatedData, shuffle=shuffle, equalPositiveAndNegative=equalPositiveAndNegative)

def trainModel(modelType : ModelInput, model : tf.keras.models.Sequential, data : commons.ModelData, epochs : int, batchSize : int, fracVal : float = 0.1, saveCheckpoints : bool = False, resampleFrequency : int = -1):
    """ Trains a model on the data in a given directory.
    
    Attributes
    ----------
    modelType : commons.ModelType
        The type of model to train.
    model : tensorflow.keras.models.Sequential
        The tensorflow model on which to train.
    data : commons.ModelData
        The annotated data to use, as a tuple of ((trainX, trainY), (testX, testY)).
    epochs : int
        The number of epochs to train for.
    batchSize : int
        The batch size to train with.
    fracVal : float
        The fraction of the training data to use for validation.
    saveCheckpoints : bool
        Whether to save checkpoints during training.
    resampleFrequency : int
        The frequency at which to resample the data. If -1 (default), no resampling is done.
        
    Returns
    -------
    model : tensorflow.keras.models.X
        The trained model.
    history : tensorflow.python.keras.callbacks.History
        The training history.
    """
    print(f"\nTraining {modelType.getType()}:")
    modelNum = len([f for f in listdir(f"{MODELS_PATH}/") if f.startswith(modelType.getType() + '_')])
    
    cpCallback = tf.keras.callbacks.ModelCheckpoint(
        filepath=f"{MODELS_PATH}/checkpoints/{modelType.getType()}_{modelNum}/" + "cp-{epoch:04d}.ckpt", 
        verbose=1, 
        save_weights_only=True,
        save_freq=5*batchSize) # type: ignore
    
    
    if resampleFrequency < 0:
        (trainX, trainY), (testX, testY) = data
    else:
        (trainX, trainY), (testX, testY) = list(map(lambda x: eimuResampler.resampleAnnotatedData(x, 96, resampleFrequency), data))
        
    valX = trainX[-round(len(trainX)*fracVal):]
    valY = trainY[-round(len(trainY)*fracVal):]
    trainX = trainX[:-round(len(trainX)*fracVal)]
    trainY = trainY[:-round(len(trainY)*fracVal)]
    
    print(f"TrainX data shape: {trainX.shape}")
    print(f"TrainY data shape: {trainY.shape}")
    print(f"ValX data shape: {valX.shape}")
    print(f"ValY data shape: {valY.shape}")
    print(f"TestX data shape: {testX.shape}")
    print(f"TestY data shape: {testY.shape}")
    
    history = model.fit(trainX, trainY, epochs=epochs, batch_size=batchSize, validation_data=(valX, valY), shuffle=True, callbacks=[cpCallback] if saveCheckpoints else None)
    
    model.summary()
    model.save(path := f"{MODELS_PATH}/{modelType.getType()}_{modelNum}.h5")
    print(f"Model saved: {path}")
    
    if len(testX) > 0:
        model.evaluate(testX, testY)
    return model, history

def trainAlternatives(classifiers : list[AlternativeClassifier], data : commons.ModelData, resampleFrequency : int = -1):
    if  resampleFrequency > 0:
        (trainX, trainY), (testX, testY) = list(map(lambda x: eimuResampler.resampleAnnotatedData(x, 96, resampleFrequency), data))
    else:
        (trainX, trainY), (testX, testY) = data
        
    samples, windowSize, channels = trainX.shape
    flattenedTrainX = np.reshape(trainX, (samples * windowSize, channels))
    extendedTrainY = np.repeat(trainY, windowSize)
    samples, windowSize, channels = testX.shape
    flattenedTestX = np.reshape(testX, (samples * windowSize, channels))
    extendedTestY = np.repeat(testY, windowSize)

    ALT_MODELS_PATH = MODELS_PATH + "/alternative/"
    if (not path.exists(ALT_MODELS_PATH)):
        mkdir(ALT_MODELS_PATH)

    # iterate over classifiers
    for classifier in classifiers:
        print("Training " + classifier.name + "...")
        classifier.fit(flattenedTrainX, extendedTrainY)
        if len(extendedTrainY) > 0:
            score = classifier.score(flattenedTestX, extendedTestY)
            print("Score: " + str(score))
    
        with open(ALT_MODELS_PATH + classifier.name + ".pkl", "wb") as handler:
            pickle.dump(classifier, handler)
            
        print("Model saved: " + ALT_MODELS_PATH + classifier.name + ".pkl")
        
    