print("Loading imports...")

from yawnnlib.utils import commons, filters
from yawnnlib.neural.eimuLSTM import EimuLSTMInput
from yawnnlib.neural.fourierLSTM import FourierLSTMInput
from yawnnlib.neural.fourierCNN import FourierCNNInput
from yawnnlib.neural.modelType import ModelType
import tools.eimuResampler as eimuResampler
from os import listdir

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Conv2D, MaxPooling2D, TimeDistributed, Flatten, Dropout
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


def trainModel(modelType : ModelType, model, annotatedData : list[commons.AnnotatedData], epochs : int, batchSize : int, saveCheckpoints : bool = True, shuffle : bool = True, equalPositiveAndNegative : bool = True):
    """ Trains a model on the data in a given directory.
    
    Attributes
    ----------
    modelType : commons.ModelType
        The type of model to train -- eimu, Fourier, etc.
    model : tensorflow.keras.models.X
        The tensorflow model on which to train.
    annotatedData : list[commons.AnnotatedData]
        The annotated data to train on.
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
    
    (trainX, trainY), (testX, testY) = modelType.fromAnnotatedDataList(annotatedData, shuffle=shuffle, equalPositiveAndNegative=equalPositiveAndNegative)
    
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
    annotatedData = modelType.fromDirectory(dataDirectory)
    _, (X, Y) = modelType.fromAnnotatedDataList(annotatedData, shuffle=False, equalPositiveAndNegative=False, trainSplit=0.0)
    model.evaluate(X, Y)





MODEL = 1

if __name__ == "__main__":
    if MODEL == 1:
        # Main EimuLSTM on all data @ 96Hz
        modelType = EimuLSTMInput(dataFilter=filters.SmoothFilter(keepData=0.8), sessionGap=32)
        trainModel(
                modelType, 
                makeSequentialModel([
                    LSTM(units=128, recurrent_dropout=0.2, return_sequences=True),
                    LSTM(units=64, recurrent_dropout=0.2, return_sequences=True),
                    Dense(units=1, activation='sigmoid')]
                ),
                modelType.fromDirectory(f"{DATA_PATH}/user-trials"),
                epochs=15, 
                batchSize=32,
                shuffle=True,
                equalPositiveAndNegative=True
        )
    elif MODEL == 2:
        # Main FourierCNN on all data @ 96Hz
        # (2 CNN layers, 2 LSTM layers, 1 dense)
        commons.ENABLE_CACHING = False
        modelType = FourierCNNInput(
            dataFilter= filters.LowPassFilter(96, 5), # filters.MovingAverageFilter(windowSize=5),
            chunkSize=commons.YAWN_TIME*1.5,
            chunkSeparation=commons.YAWN_TIME/4,    
        )
        trainModel(
                modelType, 
                makeSequentialModel([
                    Conv2D(filters=128, kernel_size=(3, 3), activation='relu'),
                    MaxPooling2D(pool_size=(2, 2)),
                    Conv2D(filters=256, kernel_size=(3, 3), activation='relu'),
                    MaxPooling2D(pool_size=(2, 2)),
                    TimeDistributed(Flatten()),
                    LSTM(units=64, recurrent_dropout=0.2, return_sequences=True),
                    LSTM(units=64, recurrent_dropout=0.2, return_sequences=True),
                    Dense(units=1, activation='sigmoid')]
                ),
                modelType.fromDirectory(f"{DATA_PATH}/user-trials"),
                epochs=30, 
                batchSize=16,
                shuffle=True,
                equalPositiveAndNegative=True
        )
    elif MODEL == 3:
        # Main FourierLSTM, 96Hz
        commons.ENABLE_CACHING = False
        modelType = FourierLSTMInput(dataFilter=filters.LowPassFilter(96, 5))
        trainModel(
                modelType, 
                makeSequentialModel([
                    LSTM(units=128, recurrent_dropout=0.2, return_sequences=True),
                    LSTM(units=64, recurrent_dropout=0.2, return_sequences=True),
                    Dense(units=1, activation='sigmoid')]
                ),
                modelType.fromDirectory(f"{DATA_PATH}/user-trials"),
                epochs=30, 
                batchSize=32,
                shuffle=True,
                equalPositiveAndNegative=True
        )
    elif MODEL == 4:
        # EimuLSTM, variable sample rate
        newSampleRate = 48
        commons.ENABLE_CACHING = True
        modelType = EimuLSTMInput(dataFilter=filters.SmoothFilter(keepData=0.8), sessionGap=newSampleRate//3)
        trainModel(
                modelType, 
                makeSequentialModel([
                    LSTM(units=128, recurrent_dropout=0.2, return_sequences=True),
                    LSTM(units=64, recurrent_dropout=0.2, return_sequences=True),
                    Dense(units=1, activation='sigmoid')]
                ),
                list(map(lambda x: eimuResampler.resampleAnnotatedData(x, 96, newSampleRate), modelType.fromDirectory(f"{DATA_PATH}/user-trials"))),
                epochs=15, 
                batchSize=8+newSampleRate//4,
                shuffle=True,
                equalPositiveAndNegative=True
        )