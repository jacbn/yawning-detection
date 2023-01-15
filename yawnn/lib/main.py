print("Loading imports...")

import commons
import filters
from fourierLSTM import FourierLSTMInput
from eimuLSTM import EimuLSTMInput
from os import listdir
import numpy as np

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.callbacks import ModelCheckpoint

print("Imports loaded.")

def makeSequentialModel(layers : list):
    model = Sequential()
    for layer in layers:
        model.add(layer)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def trainModel(modelType : commons.ModelType, model, dataDirectory : str, epochs : int, batchSize : int, saveCheckpoints : bool = True, shuffle : bool = True, equalPositiveAndNegative : bool = True):
    print(f"\nTraining {modelType.getType()}:")
    modelNum = len([f for f in listdir("./yawnn/models/") if f.startswith(modelType.getType())])
    
    cpCallback = ModelCheckpoint(
        filepath=f"./yawnn/models/checkpoints/{modelType.getType()}_{modelNum}/" + "cp-{epoch:04d}.ckpt", 
        verbose=1, 
        save_weights_only=True,
        save_freq=5*batchSize)
    
    (trainX, trainY), (testX, testY) = commons.directoryToModelData(dataDirectory, modelType, shuffle=shuffle, equalPositiveAndNegative=equalPositiveAndNegative)
    
    model.fit(trainX, trainY, epochs=epochs, batch_size=batchSize, callbacks=[cpCallback] if saveCheckpoints else None)
    
    model.save(f"./yawnn/models/{modelType.getType()}_{modelNum}.h5")
    print(f"Model saved: {model.summary()}")
    
    model.evaluate(testX, testY)


def loadModel(modelPath : str):
    model = load_model(modelPath)
    print(f"Model loaded: {model.summary()}")
    return model


def testDataOnModel(model, modelType : commons.ModelType, dataDirectory : str):
    """ Tests the model on the data in a given directory. 
    
    Attributes
    ----------
    model : tensorflow.keras.models.X
        The model to test. Load a pre-trained model with loadModel(path).
    modelType : commons.ModelType
        The type of model to test.
    dataDirectory : str
        The directory containing the data to test.
    """
    _, (X, Y) = commons.directoryToModelData(dataDirectory, modelType, shuffle=False, equalPositiveAndNegative=False, trainSplit=0.0)
    model.evaluate(X, Y)

    
if __name__ == "__main__":
    # trainModel(EimuLSTMInput(), 
    #             makeSequentialModel([
    #                LSTM(units=128, recurrent_dropout=0.2, return_sequences=True),
    #                LSTM(units=64, recurrent_dropout=0.2, return_sequences=True),
    #                Dense(units=1, activation='sigmoid')]),
    #            "./yawnn/data/user-trials", epochs=100, batchSize=32)
    trainModel(FourierLSTMInput(dataFilter=filters.MovingAverageFilter(5)), 
               makeSequentialModel([
                   LSTM(units=128, recurrent_dropout=0.2, return_sequences=True),
                   LSTM(units=64, recurrent_dropout=0.2, return_sequences=True),
                   Dense(units=1, activation='sigmoid')]),
               "./yawnn/data/user-trials", epochs=100, batchSize=32)
    