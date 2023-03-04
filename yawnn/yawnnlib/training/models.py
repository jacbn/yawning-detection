from yawnnlib.utils import commons, filters, config
from yawnnlib.neural.eimuLSTM import EimuLSTMInput
from yawnnlib.neural.fourierCNN_FFT import FourierFFTInput
from yawnnlib.neural.fourierCNN_Spectrogram import FourierCNNInput
from yawnnlib.neural.fourierTimeDistributedCNN import FourierTimeDistributedCNNInput
from yawnnlib.training.trainingFuncs import getTrainTestData, trainModel, makeSequentialModel

import tensorflow as tf

DATA_PATH = config.get("DATA_PATH")
YAWN_TIME = config.get("YAWN_TIME")

# Model 1: EimuLSTM, 3xLSTM + Dense @ variable Hz. ~95% accurate at 96Hz
def trainEimuLSTM(resampleFrequency: int = -1):
    modelType = EimuLSTMInput(
        sessionWidth=YAWN_TIME*2.5, 
        sessionGap=YAWN_TIME/8, 
        dataFilter=filters.SmoothFilter(keepData=0.8)
    )
    
    # the data has to be collected here, not inside trainModel, as we use properties of the data (namely shape) to build the model
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
    
# Model 2: EimuCNN, 4xConv + GlobalMaxPool + 2xDense @ 96Hz. ~??% accurate
def trainEimuCNN():
    modelType = EimuLSTMInput(
        # dataFilter= filters.FilterCollection([filters.HighPassFilter(96, 0.1), filters.LowPassFilter(96, 5)]), #type: ignore  
        # dataFilter=filters.MovingAverageFilter(windowSize=5),
        dataFilter=filters.LowPassFilter(96, 5),
        sessionWidth=YAWN_TIME*2,
        sessionGap=YAWN_TIME/8,    
    )
    
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
                tf.keras.layers.Conv2D(filters=32, kernel_size=(5, 1), padding='same', activation='relu'),
                tf.keras.layers.Conv2D(filters=64, kernel_size=(5, 1), padding='same', activation='relu'),
                tf.keras.layers.Conv2D(filters=128, kernel_size=(5, 1), padding='same', activation='relu'),
                tf.keras.layers.Conv2D(filters=256, kernel_size=(5, 1), padding='same', activation='relu'),
                tf.keras.layers.GlobalMaxPooling2D(),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(units=64, activation='relu'),
                tf.keras.layers.Dense(units=1, activation='softmax')],
                learningRate=0.003
            ),
            ((trainX, trainY), (testX, testY)),
            fracVal=0.1,
            epochs=60, 
            batchSize=64
    )

# Model 3: EimuLSTM, 4xConv 2xLSTM + Dense @ 96Hz. ~78% accurate
def trainEimuConvLSTM():
    modelType = EimuLSTMInput(
        sessionWidth=YAWN_TIME*2.5, 
        sessionGap=YAWN_TIME/8, 
        dataFilter=filters.SmoothFilter(keepData=0.8)
    )
    
    ((trainX, trainY), (testX, testY)) = getTrainTestData(
        modelType,
        modelType.fromDirectory(DATA_PATH), 
        shuffle=True, 
        equalPositiveAndNegative=True,
    )
    
    trainModel(
            modelType, 
            makeSequentialModel([
                tf.keras.Input(shape=trainX.shape[1:], name='input'),
                tf.keras.layers.Reshape((trainX.shape[1], 1, trainX.shape[2])),
                tf.keras.layers.Conv2D(filters=64, kernel_size=(5, 1), padding='same', activation='relu'),
                tf.keras.layers.Conv2D(filters=64, kernel_size=(5, 1), padding='same', activation='relu'),
                tf.keras.layers.Conv2D(filters=128, kernel_size=(5, 1), padding='same', activation='relu'),
                tf.keras.layers.Conv2D(filters=128, kernel_size=(5, 1), padding='same', activation='relu'),
                tf.keras.layers.Reshape((trainX.shape[1], 128)),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.LSTM(units=128, activation='tanh', return_sequences=True),
                tf.keras.layers.Dropout(0.4),
                tf.keras.layers.LSTM(units=128, activation='tanh', return_sequences=False),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(units=1, activation='sigmoid')]
            ),
            ((trainX, trainY), (testX, testY)),
            fracVal=0.1,
            epochs=30, 
            batchSize=64
    )

# Model 4: FourierCNN, 4xConv + GlobalMaxPool + 2xDense @ 96Hz. ~??% accurate
def trainFourierCNN():
    modelType = FourierCNNInput(
        # dataFilter= filters.FilterCollection([filters.HighPassFilter(96, 0.1), filters.LowPassFilter(96, 5)]), #type: ignore  
        # dataFilter=filters.MovingAverageFilter(windowSize=5),
        dataFilter=filters.LowPassFilter(96, 5),
        chunkSize=YAWN_TIME*2,
        chunkSeparation=YAWN_TIME/8,    
    )
    
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
                tf.keras.layers.Conv2D(filters=32,  kernel_size=(3, 3), padding='same', activation='relu'),
                tf.keras.layers.Conv2D(filters=64,  kernel_size=(3, 3), padding='same', activation='relu'),
                tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu'),
                tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu'),
                tf.keras.layers.GlobalMaxPooling2D(),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(units=64, activation='relu'),
                tf.keras.layers.Dense(units=1, activation='softmax')],
                learningRate=0.003
            ),
            ((trainX, trainY), (testX, testY)),
            fracVal=0.1,
            epochs=60, 
            batchSize=64
    )

# Model 5: FourierLSTM, 3xLSTM + Dense @ 96Hz. ~??% accurate
def trainFourierLSTM():
    # Main FourierLSTM, 96Hz
    # todo: rename chunkSize and chunkSeparation to sessionWidth and sessionGap for consistency
    modelType = FourierFFTInput(
        chunkSize=YAWN_TIME*1.5, 
        chunkSeparation=YAWN_TIME/4, 
        dataFilter=filters.FilterCollection([filters.HighPassFilter(96, 0.1), filters.LowPassFilter(96, 8, 3), filters.NormalisationFilter()]),
        nPerSeg=128,
        nOverlap=96
    )
    
    ((trainX, trainY), (testX, testY)) = getTrainTestData(
        modelType,
        modelType.fromDirectory(DATA_PATH), 
        shuffle=True, 
        equalPositiveAndNegative=True
    )

    trainModel(
            modelType, 
            makeSequentialModel([
                tf.keras.Input(shape=trainX.shape[1:], name='input'),
                tf.keras.layers.Reshape((trainX.shape[1], 1, trainX.shape[2])),
                tf.keras.layers.Conv2D(filters=64,  kernel_size=(6, 1), padding='same', activation='relu'),
                tf.keras.layers.Conv2D(filters=128, kernel_size=(6, 1), padding='same', activation='relu'),
                tf.keras.layers.Conv2D(filters=256, kernel_size=(6, 1), padding='same', activation='relu'),
                tf.keras.layers.Conv2D(filters=512, kernel_size=(6, 1), padding='same', activation='relu'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(units=64, activation='relu'),
                tf.keras.layers.Dense(units=1, activation='sigmoid')],
                learningRate=0.0003
            ),
            ((trainX, trainY), (testX, testY)),
            fracVal=0.1,
            epochs=30,
            batchSize=64
    )

# Model 6: TimeDistributedFourierCNN
def trainTDFourierCNN():
    #todo: remove this input type and reshape with C=1 in the model itself?
    modelType = FourierTimeDistributedCNNInput(
        # dataFilter= filters.FilterCollection([filters.HighPassFilter(96, 0.1), filters.LowPassFilter(96, 4)]), #type: ignore  
        # dataFilter=filters.MovingAverageFilter(windowSize=5),
        dataFilter=filters.LowPassFilter(96, 5),
        chunkSize=YAWN_TIME*1.5,
        chunkSeparation=YAWN_TIME/8,    
    )

    ((trainX, trainY), (testX, testY)) = getTrainTestData(
        modelType,
        modelType.fromDirectory(DATA_PATH),
        shuffle=True, 
        equalPositiveAndNegative=True
    )

    trainModel(
            modelType, 
            makeSequentialModel([
                tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu')),
                tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D(pool_size=(2, 2))),
                tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu')),
                tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D(pool_size=(2, 2))),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(units=1, activation='sigmoid', input_shape=(None, 1))]
            ),
            # makeSequentialModel([
            #    ConvLSTM2D(filters=32, kernel_size=(3, 3), activation='relu', return_sequences=True),
            #    ConvLSTM2D(filters=64, kernel_size=(3, 3), activation='relu', return_sequences=True),
            #    ConvLSTM2D(filters=128, kernel_size=(3, 3), activation='relu', return_sequences=True),
            #    Flatten(),
            #    Dense(units=1, activation='sigmoid', input_shape=(None, 1))
            # ]),
            ((trainX, trainY), (testX, testY)),
            fracVal=0.1,
            epochs=5, 
            batchSize=16,
    )