print("Loading imports...")

from yawnnlib.utils import commons, filters
from yawnnlib.lstm.eimuLSTM import EimuLSTMInput
from yawnnlib.lstm.fourierLSTM import FourierLSTMInput
from yawnnlib.lstm.modelType import ModelType
from os import listdir

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint

print("Imports loaded.")

MODELS_PATH = f"{commons.PROJECT_ROOT}/models"
DATA_PATH = f"{commons.PROJECT_ROOT}/data"

def makeSequentialModel(layers : list) -> Sequential:
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
    model = Sequential()
    for layer in layers:
        model.add(layer)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def trainModel(modelType : ModelType, model, dataDirectory : str, epochs : int, batchSize : int, saveCheckpoints : bool = True, shuffle : bool = True, equalPositiveAndNegative : bool = True):
    """ Trains a model on the data in a given directory.
    
    Attributes
    ----------
    modelType : commons.ModelType
        The type of model to train -- eimu, Fourier, etc.
    model : tensorflow.keras.models.X
        The tensorflow model on which to train.
    dataDirectory : str
        The directory containing the data to train with.
    epochs : int
        The number of epochs to train for.
    batchSize : int
        The batch size to train with.
    saveCheckpoints : bool
        Whether to save checkpoints during training.
    shuffle : bool
        Whether to shuffle the data before training.
    equalPositiveAndNegative : bool
        Whether to equalize the number of positive and negative samples before training.
        
    Returns
    -------
    model : tensorflow.keras.models.X
        The trained model.
    """
    print(f"\nTraining {modelType.getType()}:")
    modelNum = len([f for f in listdir(f"{MODELS_PATH}/") if f.startswith(modelType.getType())])
    
    cpCallback = ModelCheckpoint(
        filepath=f"{MODELS_PATH}/checkpoints/{modelType.getType()}_{modelNum}/" + "cp-{epoch:04d}.ckpt", 
        verbose=1, 
        save_weights_only=True,
        save_freq=5*batchSize)
    
    (trainX, trainY), (testX, testY) = modelType.fromDirectory(dataDirectory, shuffle=shuffle, equalPositiveAndNegative=equalPositiveAndNegative)
    
    model.fit(trainX, trainY, epochs=epochs, batch_size=batchSize, callbacks=[cpCallback] if saveCheckpoints else None)
    
    model.save(f"{MODELS_PATH}/{modelType.getType()}_{modelNum}.h5")
    print(f"Model saved: {model.summary()}")
    
    model.evaluate(testX, testY)
    return model


def loadModel(modelPath : str):
    """ Loads a pre-trained model.
    
    Attributes
    ----------
    modelPath : str
        The path to the model to load.
    
    Returns
    -------
    model : tensorflow.keras.models.X
        The loaded model.
    """
    model = load_model(modelPath)
    print(f"Model loaded: {model.summary()}")
    return model


def testDataOnModel(model, modelType : ModelType, dataDirectory : str):
    """ Tests the model on the data in a given directory. 
    
    Attributes
    ----------
    model : tensorflow.keras.models.X
        The tf model to test. Load a pre-trained model with loadModel(path).
    modelType : commons.ModelType
        The type of model to test.
    dataDirectory : str
        The directory containing the data to test.
    """
    _, (X, Y) = modelType.fromDirectory(dataDirectory, shuffle=False, equalPositiveAndNegative=False, trainSplit=0.0)
    model.evaluate(X, Y)

    
if __name__ == "__main__":
    trainModel( EimuLSTMInput(
                    dataFilter=filters.SmoothFilter(keepData=0.5),
                    sessionGap=32,
                ), 
                makeSequentialModel([
                    LSTM(units=128, recurrent_dropout=0.2, return_sequences=True),
                    Dropout(0.2),
                    LSTM(units=64, recurrent_dropout=0.2, return_sequences=True),
                    Dropout(0.2),
                    Dense(units=1, activation='sigmoid')]
                ),
                f"{DATA_PATH}/user-trials", 
                epochs=30, 
                batchSize=32,
                shuffle=True,
                equalPositiveAndNegative=True
            )
    # trainModel(FourierLSTMInput(dataFilter=filters.MovingAverageFilter(5)), 
    #            makeSequentialModel([
    #                LSTM(units=128, recurrent_dropout=0.2, return_sequences=True),
    #                Dropout(0.2),
    #                LSTM(units=64, recurrent_dropout=0.2, return_sequences=True),
    #                Dropout(0.2),
    #                Dense(units=1, activation='sigmoid')]),
    #            f"{DATA_PATH}/user-trials", epochs=100, batchSize=32)
    