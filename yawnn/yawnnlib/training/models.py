from yawnnlib.utils import filters, config
from yawnnlib.neural.eimuModelInput import EimuModelInput
from yawnnlib.neural.fftModelInput import FFTModelInput
from yawnnlib.neural.spectrogramModelInput import SpectrogramModelInput
from yawnnlib.training.trainingFuncs import getTrainTestData, trainModel, makeSequentialModel

import tensorflow as tf

DATA_PATH = config.get("DATA_PATH")
YAWN_TIME = config.get("YAWN_TIME")

MODEL_INPUTS = {
    'eimuLSTM': EimuModelInput(
                    windowSize=YAWN_TIME*2.5, 
                    windowSep=YAWN_TIME/8, 
                    dataFilter=filters.SmoothFilter(keepData=0.8),
                    name='eimuLSTM'
                ),
    
    'eimuCNN':  EimuModelInput(
                    windowSize=YAWN_TIME*2.5, 
                    windowSep=YAWN_TIME/8, 
                    dataFilter=filters.SmoothFilter(keepData=0.8),
                    name='eimuCNN'
                ),
    
    'fftLSTM':  FFTModelInput(
                    windowSize=YAWN_TIME,
                    windowSep=YAWN_TIME/8,
                    dataFilter=filters.FilterCollection([
                        filters.HighPassFilter(96, 0.01, 30), 
                        # filters.LowPassFilter(96, 300, 10), 
                        filters.NormalisationFilter()
                    ]),
                    nPerSeg=128,
                    nOverlap=96,
                    name='fftLSTM'
                ),                  
    
    'fftCNN':   FFTModelInput(
                    windowSize=YAWN_TIME*1.5, 
                    windowSep=YAWN_TIME/4, 
                    dataFilter=filters.FilterCollection([
                        filters.HighPassFilter(96, 0.1), 
                        filters.LowPassFilter(96, 8, 3), 
                        filters.NormalisationFilter()
                    ]),
                    nPerSeg=128,
                    nOverlap=96,
                    name='fftCNN'
                ),
    
    'specCNN':  SpectrogramModelInput(
                    windowSize=YAWN_TIME*1.5,
                    windowSep=YAWN_TIME/4,
                    dataFilter=filters.FilterCollection([
                        filters.HighPassFilter(96, 0.1), 
                        filters.LowPassFilter(96, 8, 3), 
                        filters.NormalisationFilter()
                    ]),
                    name='specCNN'
                ),
    'tdFftCNN': FFTModelInput(
                    windowSize=YAWN_TIME*1.5, 
                    windowSep=YAWN_TIME/4, 
                    dataFilter=filters.FilterCollection([
                        filters.HighPassFilter(96, 0.1), 
                        filters.LowPassFilter(96, 8, 3), 
                        filters.NormalisationFilter()
                    ]),
                    nPerSeg=128,
                    nOverlap=96,
                    name='tdFftCNN'
                ),
}

# Model 1: EimuLSTM, 3xLSTM + Dense @ variable Hz. ~95% accurate at 96Hz
def trainEimuLSTM(resampleFrequency: int = -1):
    modelType = MODEL_INPUTS['eimuLSTM']
    
    # the data has to be collected here, not inside trainModel, as we use properties of the data (namely shape) to build the model
    # (in this model, it is only used in the optional Input layer, but other models require it for reshapes and so for consistency it is kept)
    ((trainX, trainY), (testX, testY)) = getTrainTestData(
        modelType,
        modelType.fromDirectory(DATA_PATH), 
        shuffle=True, 
        equalPositiveAndNegative=True
    )
    
    return trainModel(
            modelType, 
            makeSequentialModel([
                tf.keras.Input(shape=trainX.shape[1:], name='input'),
                tf.keras.layers.LSTM(units=128, activation='tanh', return_sequences=True),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.LSTM(units=128, activation='tanh', return_sequences=True),
                tf.keras.layers.Dropout(0.4),
                tf.keras.layers.LSTM(units=128,  activation='tanh', return_sequences=False),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(units=1, activation='sigmoid')]
            ),
            ((trainX, trainY), (testX, testY)),
            fracVal=0.1,
            epochs=30, 
            batchSize=64,
            resampleFrequency=resampleFrequency
    )
    
# Model 2: EimuCNN, 4xConv + 2xDense @ 96Hz. ~85% accurate
# todo: only trains successfully occasionally, needs investigation
def trainEimuCNN():
    modelType = MODEL_INPUTS['eimuCNN']
    
    ((trainX, trainY), (testX, testY)) = getTrainTestData(
        modelType,
        modelType.fromDirectory(DATA_PATH), 
        shuffle=True, 
        equalPositiveAndNegative=True
    )
    
    return trainModel(
            modelType, 
            makeSequentialModel([
                tf.keras.Input(shape=trainX.shape[1:], name='input'),
                tf.keras.layers.Reshape((trainX.shape[1], 1, trainX.shape[2])),
                tf.keras.layers.Conv2D(filters=64,  kernel_size=(5, 1), padding='same', activation='relu'),
                tf.keras.layers.MaxPool2D(pool_size=(2, 1)),
                tf.keras.layers.Conv2D(filters=128, kernel_size=(5, 1), padding='same', activation='relu'),
                tf.keras.layers.MaxPool2D(pool_size=(2, 1)),
                tf.keras.layers.Conv2D(filters=256, kernel_size=(5, 1), padding='same', activation='relu'),
                tf.keras.layers.MaxPool2D(pool_size=(2, 1)),
                tf.keras.layers.Conv2D(filters=512, kernel_size=(5, 1), padding='same', activation='relu'),
                tf.keras.layers.MaxPool2D(pool_size=(2, 1)),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(units=64, activation='relu'),
                tf.keras.layers.Dense(units=1, activation='sigmoid')],
                learningRate=1e-4
            ),
            ((trainX, trainY), (testX, testY)),
            fracVal=0.1,
            epochs=45,
            batchSize=64
    )

# Model 3: EimuLSTM, 4xConv + 2xLSTM + Dense @ 96Hz. ~78% accurate
def trainEimuConvLSTM():
    modelType = MODEL_INPUTS['eimuLSTM']
    
    ((trainX, trainY), (testX, testY)) = getTrainTestData(
        modelType,
        modelType.fromDirectory(DATA_PATH), 
        shuffle=True, 
        equalPositiveAndNegative=True,
    )
    
    return trainModel(
            modelType, 
            makeSequentialModel([
                tf.keras.Input(shape=trainX.shape[1:], name='input'),
                tf.keras.layers.Reshape((trainX.shape[1]//4, 4, 1, trainX.shape[2])),
                tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(filters=128,  kernel_size=(5, 1), padding='same', activation='relu')),
                tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPool2D(pool_size=(2, 1))),
                tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(filters=128,  kernel_size=(5, 1), padding='same', activation='relu')),
                tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPool2D(pool_size=(2, 1))),
                tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten()),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.LSTM(units=64, activation='tanh', return_sequences=True),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.LSTM(units=64, activation='tanh', return_sequences=False),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(units=64, activation='relu'),
                tf.keras.layers.Dense(units=1, activation='sigmoid')]
            ),
            ((trainX, trainY), (testX, testY)),
            fracVal=0.1,
            epochs=30, 
            batchSize=64
    )

# Model 4: Spectrogram CNN, 4xConv + 2xDense @ 96Hz. ~85% accurate (overfitting)
def trainSpectrogramCNN():
    modelType = MODEL_INPUTS['specCNN']
    
    ((trainX, trainY), (testX, testY)) = getTrainTestData(
        modelType,
        modelType.fromDirectory(DATA_PATH), 
        shuffle=True, 
        equalPositiveAndNegative=True
    )
    
    return trainModel(
            modelType, 
            makeSequentialModel([
                tf.keras.Input(shape=trainX.shape[1:], name="input"),
                tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu'),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu'),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(units=64, activation='relu'),
                tf.keras.layers.Dense(units=1, activation='sigmoid')]
            ),
            ((trainX, trainY), (testX, testY)),
            fracVal=0.1,
            epochs=30, 
            batchSize=64
    )

# Model 5: FFT CNN, 4xConv + 2xDense @ 96Hz. ~90% accurate, overfitting
def trainFftCNN():
    modelType = MODEL_INPUTS['fftCNN']
    
    ((trainX, trainY), (testX, testY)) = getTrainTestData(
        modelType,
        modelType.fromDirectory(DATA_PATH), 
        shuffle=True, 
        equalPositiveAndNegative=True
    )

    return trainModel(
            modelType, 
            makeSequentialModel([
                tf.keras.Input(shape=trainX.shape[1:], name='input'),
                tf.keras.layers.Reshape((trainX.shape[1], 1, trainX.shape[2])),
                tf.keras.layers.Conv2D(filters=64,  kernel_size=(5, 1), padding='same', activation='relu'),
                tf.keras.layers.Conv2D(filters=128, kernel_size=(5, 1), padding='same', activation='relu'),
                tf.keras.layers.Conv2D(filters=256, kernel_size=(5, 1), padding='same', activation='relu'),
                tf.keras.layers.Conv2D(filters=512, kernel_size=(5, 1), padding='same', activation='relu'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(units=64, activation='relu'),
                tf.keras.layers.Dense(units=1, activation='sigmoid')]
            ),
            ((trainX, trainY), (testX, testY)),
            fracVal=0.1,
            epochs=30,
            batchSize=64
    )


def trainFftLSTM():
    modelType = MODEL_INPUTS['fftLSTM']
    
    ((trainX, trainY), (testX, testY)) = getTrainTestData(
        modelType,
        modelType.fromDirectory(DATA_PATH), 
        shuffle=True, 
        equalPositiveAndNegative=True
    )
    
    return trainModel(
            modelType, 
            makeSequentialModel([
                tf.keras.Input(shape=trainX.shape[1:], name='input'),
                tf.keras.layers.LSTM(units=128, activation='tanh', return_sequences=True),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.LSTM(units=128, activation='tanh', return_sequences=True),
                tf.keras.layers.Dropout(0.4),
                tf.keras.layers.LSTM(units=128, activation='tanh', return_sequences=False),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(units=1, activation='sigmoid')]
            ),
            ((trainX, trainY), (testX, testY)),
            fracVal=0.1,
            epochs=30, 
            batchSize=64,
    )