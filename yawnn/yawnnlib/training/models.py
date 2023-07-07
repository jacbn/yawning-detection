from yawnnlib.utils import commons, filters, config
from yawnnlib.preprocessing.eimuModelInput import EimuModelInput
from yawnnlib.preprocessing.fftModelInput import FFTModelInput
from yawnnlib.preprocessing.spectrogramModelInput import SpectrogramModelInput
from yawnnlib.preprocessing.modelData import ModelData
from yawnnlib.alternatives.alternative_classifier import AlternativeClassifier
from yawnnlib.training.trainingFuncs import trainModel, makeSequentialModel, trainAlternatives

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

import tensorflow as tf

DATA_PATH = config.get("DATA_PATH")
HAFAR_PATH = config.get("HAFAR_PATH")
YAWN_TIME = config.get("YAWN_TIME")

MODEL_INPUTS = {
    # these are the models used in the dissertation. the hafar dataset requires strict 2s inputs so the *2.5, /4 etc can't be used 
    # 'eimuLSTM': EimuModelInput(
    #                 windowSize=YAWN_TIME*2.5, 
    #                 windowSep=YAWN_TIME/4, 
    #                 dataFilter=filters.SmoothFilter(keepData=0.8),
    #                 name='eimuLSTM'
    #             ),
    
    # 'eimuCNN':  EimuModelInput(  # *2.5, /4 = 0.8482 acc + good metrics
    #                 windowSize=YAWN_TIME*2.5,
    #                 windowSep=YAWN_TIME/4, 
    #                 dataFilter=filters.SmoothFilter(keepData=0.8),
    #                 name='eimuCNN'
    #             ),
    
    # 'eimuCNN-LSTM': EimuModelInput(
    #                 windowSize=YAWN_TIME, 
    #                 windowSep=YAWN_TIME/8, 
    #                 dataFilter=filters.FilterCollection([
    #                     filters.MovingAverageFilter(5),
    #                     filters.SmoothFilter(keepData=0.8),
    #                 ]),
    #                 name='eimuCNN-LSTM'
    #             ),
    
    # 'fftLSTM':  FFTModelInput(
    #                 windowSize=YAWN_TIME*1.5,
    #                 windowSep=YAWN_TIME/6,
    #                 dataFilter=filters.FilterCollection([
    #                     # filters.SmoothFilter(keepData=0.8),
    #                     filters.HighPassFilter(96, 0.1), 
    #                     filters.LowPassFilter(96, 8, 3), 
    #                     # filters.MovingAverageFilter(5), 
    #                     filters.NormalisationFilter()
    #                 ]),
    #                 nPerSeg=128,
    #                 nOverlap=96,
    #                 name='fftLSTM'
    #             ),                  
    
    # 'fftCNN':   FFTModelInput(
    #                 windowSize=YAWN_TIME*1.5, 
    #                 windowSep=YAWN_TIME/4, 
    #                 dataFilter=filters.FilterCollection([
    #                     # filters.SmoothFilter(keepData=0.8),
    #                     filters.HighPassFilter(96, 0.1), 
    #                     filters.LowPassFilter(96, 8, 3), 
    #                     # filters.MovingAverageFilter(5),
    #                     filters.NormalisationFilter()
    #                 ]),
    #                 nPerSeg=128,
    #                 nOverlap=96,
    #                 name='fftCNN'
    #             ),
    
    # 'fftCNN-LSTM': FFTModelInput(
    #                 windowSize=YAWN_TIME, 
    #                 windowSep=YAWN_TIME/12, 
    #                 dataFilter=filters.FilterCollection([
    #                     # filters.SmoothFilter(keepData=0.8),
    #                     # filters.HighPassFilter(96, 0.01, 30), 
    #                     # filters.LowPassFilter(96, 12, 3), 
    #                     # filters.MovingAverageFilter(5),
    #                     filters.NormalisationFilter()
    #                 ]),
    #                 nPerSeg=128,
    #                 nOverlap=96,
    #                 name='fftCNN-LSTM'
    #             ),
    
    # 'specCNN':  SpectrogramModelInput(
    #                 windowSize=YAWN_TIME*1.5,
    #                 windowSep=YAWN_TIME/4,
    #                 dataFilter=filters.FilterCollection([
    #                     # filters.SmoothFilter(keepData=0.8),
    #                     filters.HighPassFilter(96, 0.1), 
    #                     filters.LowPassFilter(96, 8, 3), 
    #                     # filters.MovingAverageFilter(5),
    #                     filters.NormalisationFilter()
    #                 ]),
    #                 nPerSeg=128,
    #                 nOverlap=96,
    #                 name='specCNN'
    #             ),
    # 'altModels': EimuModelInput(
    #                 windowSize=YAWN_TIME, 
    #                 windowSep=YAWN_TIME, 
    #                 name='altModels'
    #             ),
    'eimuLSTM': EimuModelInput(
                    windowSize=YAWN_TIME, 
                    windowSep=YAWN_TIME/2, 
                    dataFilter=filters.SmoothFilter(keepData=0.8),
                    name='eimuLSTM'
                ),
    
    'eimuCNN':  EimuModelInput(
                    windowSize=YAWN_TIME,
                    windowSep=YAWN_TIME/2, 
                    dataFilter=filters.SmoothFilter(keepData=0.8),
                    name='eimuCNN'
                ),
    
    'eimuCNN-LSTM': EimuModelInput(
                    windowSize=YAWN_TIME, 
                    windowSep=YAWN_TIME/2, 
                    dataFilter=filters.FilterCollection([
                        filters.MovingAverageFilter(5),
                        filters.SmoothFilter(keepData=0.8),
                    ]),
                    name='eimuCNN-LSTM'
                ),
    
    'fftLSTM':  FFTModelInput(
                    windowSize=YAWN_TIME/2,
                    windowSep=YAWN_TIME,
                    dataFilter=filters.FilterCollection([
                        # filters.SmoothFilter(keepData=0.8),
                        filters.HighPassFilter(96, 0.1), 
                        filters.LowPassFilter(96, 8, 3), 
                        # filters.MovingAverageFilter(5), 
                        filters.NormalisationFilter()
                    ]),
                    nPerSeg=32,
                    nOverlap=16,
                    name='fftLSTM'
                ),                  
    
    'fftCNN':   FFTModelInput(
                    windowSize=YAWN_TIME, 
                    windowSep=YAWN_TIME, 
                    dataFilter=filters.FilterCollection([
                        # filters.SmoothFilter(keepData=0.8),
                        filters.HighPassFilter(96, 0.1), 
                        filters.LowPassFilter(96, 8, 3), 
                        # filters.MovingAverageFilter(5),
                        filters.NormalisationFilter()
                    ]),
                    nPerSeg=128,
                    nOverlap=96,
                    name='fftCNN'
                ),
    
    'fftCNN-LSTM': FFTModelInput(
                    windowSize=YAWN_TIME, 
                    windowSep=YAWN_TIME, 
                    dataFilter=filters.FilterCollection([
                        # filters.SmoothFilter(keepData=0.8),
                        # filters.HighPassFilter(96, 0.01, 30), 
                        # filters.LowPassFilter(96, 12, 3), 
                        # filters.MovingAverageFilter(5),
                        filters.NormalisationFilter()
                    ]),
                    nPerSeg=128,
                    nOverlap=96,
                    name='fftCNN-LSTM'
                ),
    
    'specCNN':  SpectrogramModelInput(
                    windowSize=YAWN_TIME,
                    windowSep=YAWN_TIME,
                    dataFilter=filters.FilterCollection([
                        # filters.SmoothFilter(keepData=0.8),
                        filters.HighPassFilter(96, 0.1), 
                        filters.LowPassFilter(96, 8, 3), 
                        # filters.MovingAverageFilter(5),
                        filters.NormalisationFilter()
                    ]),
                    nPerSeg=128,
                    nOverlap=96,
                    name='specCNN'
                ),
    'altModels': EimuModelInput(
                    windowSize=YAWN_TIME, 
                    windowSep=YAWN_TIME, 
                    name='altModels'
                ),
}

# Model 1: EimuLSTM, 3xLSTM + Dense @ variable Hz. ~95% accurate at 96Hz
def trainEimuLSTM(resampleFrequency: int = -1, modelNum : int = 0, totalModels : int = 1):
    modelType = MODEL_INPUTS['eimuLSTM']
    
    # the data has to be collected here, not inside trainModel, as we use properties of the data (namely shape) to build the model
    # (in this model, it is only used in the optional Input layer, but other models require it for reshapes and so for consistency it is kept)
    if config.get("FILE_FORMAT") == "eimu":
        modelData = modelType.fromEimuDirectory(DATA_PATH, shuffle=True, equalPositiveAndNegative=True)
    else:
        modelData = modelType.fromHafarDirectory(HAFAR_PATH, shuffle=True, equalPositiveAndNegative=True)
    
    ((trainX, trainY), (valX, valY), (testX, testY)), trainingSampleWeights = modelData.splitValidationFromTrainTest(
        modelNum=modelNum,
        totalModels=totalModels
    )
    
    #todo: replace all Input layers. requires a reshape of the first dimension by newSampleRate//oldSampleRate
    # (-> could have the original sample rate be a property of modelType? new is resampleFrequency)
    
    return trainModel(
            modelType, 
            makeSequentialModel([
                # tf.keras.Input(shape=trainX.shape[1:], name='input'),
                tf.keras.layers.LSTM(units=128, activation='tanh', return_sequences=True),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.LSTM(units=128, activation='tanh', return_sequences=True),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.LSTM(units=128, activation='tanh', return_sequences=True),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.GlobalAveragePooling1D(),
                tf.keras.layers.Dense(units=1, activation='sigmoid')]
            ),
            ((trainX, trainY), (valX, valY), (testX, testY)),
            trainingSampleWeights,
            modelData.sampleRate,
            epochs=5, # pre-hafar: 15
            batchSize=64,
            resampleFrequency=resampleFrequency
    )
    
# Model 2: EimuCNN, 4xConv + 2xDense @ 96Hz. ~85% accurate
# training can converge wrongly if learning rate too high
def trainEimuCNN(resampleFrequency: int = -1, modelNum : int = 0, totalModels : int = 1):
    modelType = MODEL_INPUTS['eimuCNN']
    
    if config.get("FILE_FORMAT") == "eimu":
        modelData = modelType.fromEimuDirectory(DATA_PATH, shuffle=True, equalPositiveAndNegative=True)
    else:
        modelData = modelType.fromHafarDirectory(HAFAR_PATH, shuffle=True, equalPositiveAndNegative=True)
    
    ((trainX, trainY), (valX, valY), (testX, testY)), trainingSampleWeights = modelData.splitValidationFromTrainTest(
        modelNum=modelNum,
        totalModels=totalModels
    )
    
    return trainModel(
            modelType, 
            makeSequentialModel([
                # tf.keras.Input(shape=trainX.shape[1:], name='input'),
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
            ((trainX, trainY), (valX, valY), (testX, testY)),
            trainingSampleWeights,
            modelData.sampleRate,
            epochs=20,
            batchSize=64,
            resampleFrequency=resampleFrequency
    )

# Model 3: EimuCNN-LSTM, 2xConv + 2xLSTM + 2xDense @ 96Hz. ~92% accurate
def trainEimuConvLSTM(resampleFrequency: int = -1, modelNum : int = 0, totalModels : int = 1):
    modelType = MODEL_INPUTS['eimuCNN-LSTM']
    
    if config.get("FILE_FORMAT") == "eimu":
        modelData = modelType.fromEimuDirectory(DATA_PATH, shuffle=True, equalPositiveAndNegative=True)
    else:
        modelData = modelType.fromHafarDirectory(HAFAR_PATH, shuffle=True, equalPositiveAndNegative=True)
    
    ((trainX, trainY), (valX, valY), (testX, testY)), trainingSampleWeights = modelData.splitValidationFromTrainTest(
        modelNum=modelNum,
        totalModels=totalModels
    )
    
    return trainModel(
            modelType, 
            makeSequentialModel([
                # tf.keras.Input(shape=trainX.shape[1:], name='input'),
                tf.keras.layers.Reshape((trainX.shape[1], trainX.shape[2], 1, *trainX.shape[3:])),
                tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(filters=128,  kernel_size=(5, 1), padding='same', activation='relu')),
                tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPool2D(pool_size=(2, 1))),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(filters=128,  kernel_size=(5, 1), padding='same', activation='relu')),
                tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPool2D(pool_size=(2, 1))),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(filters=128,  kernel_size=(5, 1), padding='same', activation='relu')),
                tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPool2D(pool_size=(2, 1))),
                tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten()),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.LSTM(units=64, activation='tanh', return_sequences=True),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.LSTM(units=64, activation='tanh', return_sequences=False),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(units=1, activation='sigmoid')]
            ),
            ((trainX, trainY), (valX, valY), (testX, testY)),
            trainingSampleWeights,
            modelData.sampleRate,
            epochs=20, 
            batchSize=64,
            resampleFrequency=resampleFrequency
    )

# Model 4: FFT LSTM, 3xLSTM 1xDense @ 96Hz. needs testing
def trainFftLSTM(resampleFrequency: int = -1, modelNum : int = 0, totalModels : int = 1):
    modelType = MODEL_INPUTS['fftLSTM']
    
    if config.get("FILE_FORMAT") == "eimu":
        modelData = modelType.fromEimuDirectory(DATA_PATH, shuffle=True, equalPositiveAndNegative=True)
    else:
        modelData = modelType.fromHafarDirectory(HAFAR_PATH, shuffle=True, equalPositiveAndNegative=True)
    
    ((trainX, trainY), (valX, valY), (testX, testY)), trainingSampleWeights = modelData.splitValidationFromTrainTest(
        modelNum=modelNum,
        totalModels=totalModels
    )
    
    return trainModel(
            modelType, 
            makeSequentialModel([
                # tf.keras.Input(shape=trainX.shape[1:], name='input'),
                tf.keras.layers.LSTM(units=128, activation='tanh', return_sequences=True),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.LSTM(units=128, activation='tanh', return_sequences=True),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.LSTM(units=128, activation='tanh', return_sequences=True),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.GlobalAveragePooling1D(),
                tf.keras.layers.Dense(units=1, activation='sigmoid')],
                learningRate=3e-4
            ),
            ((trainX, trainY), (valX, valY), (testX, testY)),
            trainingSampleWeights,
            modelData.sampleRate,
            epochs=5, # pre-hafar: 60, 
            batchSize=128,
            resampleFrequency=resampleFrequency
    ) # fftLSTM_9: 0.8125, avg pool 3e-4 1 /12 5.9 yawn time (!!)

# Model 5: FFT CNN, 4xConv + 2xDense @ 96Hz. ~90% accurate, overfitting
def trainFftCNN(resampleFrequency: int = -1, modelNum : int = 0, totalModels : int = 1): #!prefers extended YAWN_TIME
    modelType = MODEL_INPUTS['fftCNN']
    
    if config.get("FILE_FORMAT") == "eimu":
        modelData = modelType.fromEimuDirectory(DATA_PATH, shuffle=True, equalPositiveAndNegative=True)
    else:
        modelData = modelType.fromHafarDirectory(HAFAR_PATH, shuffle=True, equalPositiveAndNegative=True)
    
    ((trainX, trainY), (valX, valY), (testX, testY)), trainingSampleWeights = modelData.splitValidationFromTrainTest(
        modelNum=modelNum,
        totalModels=totalModels
    )

    return trainModel(
            modelType, 
            makeSequentialModel([
                # tf.keras.Input(shape=trainX.shape[1:], name='input'),
                tf.keras.layers.Reshape((trainX.shape[1], 1, trainX.shape[2])),
                tf.keras.layers.Conv2D(filters=64,  kernel_size=(5, 1), padding='same', activation='relu'),
                tf.keras.layers.MaxPool2D(pool_size=(2, 1)),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Conv2D(filters=128, kernel_size=(5, 1), padding='same', activation='relu'),
                tf.keras.layers.MaxPool2D(pool_size=(2, 1)),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Conv2D(filters=256, kernel_size=(5, 1), padding='same', activation='relu'),
                tf.keras.layers.MaxPool2D(pool_size=(2, 1)),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Conv2D(filters=512, kernel_size=(5, 1), padding='same', activation='relu'),
                tf.keras.layers.MaxPool2D(pool_size=(2, 1)),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(units=64, activation='relu'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(units=1, activation='sigmoid')]
            ),
            ((trainX, trainY), (valX, valY), (testX, testY)),
            trainingSampleWeights,
            modelData.sampleRate,
            epochs=15, # pre-hafar: 40 
            batchSize=64,
            resampleFrequency=resampleFrequency
    )

# Model 6: FFT CNN-LSTM, 2xConv + 2xLSTM + 2xDense @ 96Hz. ~92% accurate
def trainFftConvLSTM(resampleFrequency: int = -1, modelNum : int = 0, totalModels : int = 1):
    modelType = MODEL_INPUTS['fftCNN-LSTM']
    
    if config.get("FILE_FORMAT") == "eimu":
        modelData = modelType.fromEimuDirectory(DATA_PATH, shuffle=True, equalPositiveAndNegative=True)
    else:
        modelData = modelType.fromHafarDirectory(HAFAR_PATH, shuffle=True, equalPositiveAndNegative=True)
    
    ((trainX, trainY), (valX, valY), (testX, testY)), trainingSampleWeights = modelData.splitValidationFromTrainTest(
        modelNum=modelNum,
        totalModels=totalModels
    )
    
    return trainModel(
            modelType, 
            makeSequentialModel([
                # tf.keras.Input(shape=trainX.shape[1:], name='input'),
                tf.keras.layers.Reshape((trainX.shape[1], trainX.shape[2], 1, *trainX.shape[3:])),
                tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(filters=128,  kernel_size=(5, 1), padding='same', activation='relu')),
                tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPool2D(pool_size=(2, 1))),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(filters=128,  kernel_size=(5, 1), padding='same', activation='relu')),
                tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPool2D(pool_size=(2, 1))),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(filters=128,  kernel_size=(5, 1), padding='same', activation='relu')),
                tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPool2D(pool_size=(2, 1))),
                tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten()),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.LSTM(units=64, activation='tanh', return_sequences=True),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.LSTM(units=64, activation='tanh', return_sequences=False),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(units=1, activation='sigmoid')]
            ),
            ((trainX, trainY), (valX, valY), (testX, testY)),
            trainingSampleWeights,
            modelData.sampleRate,
            epochs=20, # pre-hafar: 40 
            batchSize=64,
            resampleFrequency=resampleFrequency
    ) # fftCNN-LSTM_1: 0.8783 1 /12 5.9 yawn time
    # fftCNN-LSTM_2: 0.8553 1 /12 5.9 yawn time no dense
    
# Model 7: Spectrogram CNN, 4xConv + 2xDense @ 96Hz. ~85% accurate (overfitting)
def trainSpectrogramCNN(resampleFrequency: int = -1, modelNum : int = 0, totalModels : int = 1):
    modelType = MODEL_INPUTS['specCNN']
    
    if config.get("FILE_FORMAT") == "eimu":
        modelData = modelType.fromEimuDirectory(DATA_PATH, shuffle=True, equalPositiveAndNegative=True)
    else:
        modelData = modelType.fromHafarDirectory(HAFAR_PATH, shuffle=True, equalPositiveAndNegative=True)
    
    ((trainX, trainY), (valX, valY), (testX, testY)), trainingSampleWeights = modelData.splitValidationFromTrainTest(
        modelNum=modelNum,
        totalModels=totalModels
    )
    
    return trainModel(
            modelType, 
            makeSequentialModel([
                # tf.keras.Input(shape=trainX.shape[1:], name="input"),
                tf.keras.layers.Conv2D(filters=256, kernel_size=(5, 5), padding='same', activation='relu'),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Conv2D(filters=256, kernel_size=(5, 5), padding='same', activation='relu'),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Conv2D(filters=256, kernel_size=(5, 5), padding='same', activation='relu'),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Conv2D(filters=256, kernel_size=(5, 5), padding='same', activation='relu'),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(units=64, activation='relu'),
                tf.keras.layers.Dense(units=1, activation='sigmoid')]
            ),
            ((trainX, trainY), (valX, valY), (testX, testY)),
            trainingSampleWeights,
            modelData.sampleRate,
            epochs=15, # pre-hafar: 40 
            batchSize=64,
            resampleFrequency=resampleFrequency
    )

# Model 8: Alternative Classifiers
def trainAlternativeClassifiers(resampleFrequency: int = -1):
    modelType = MODEL_INPUTS['altModels']
    
    if config.get("FILE_FORMAT") == "eimu":
        modelData = modelType.fromEimuDirectory(DATA_PATH, shuffle=True, equalPositiveAndNegative=True)
    else:
        modelData = modelType.fromHafarDirectory(HAFAR_PATH, shuffle=True, equalPositiveAndNegative=True)
    
    ((trainX, trainY), _, (testX, testY)), trainingSampleWeights = modelData.splitValidationFromTrainTest(
        modelNum=0,
    )
    
    classifiers = [
        AlternativeClassifier("K-Nearest Neighbors", KNeighborsClassifier(n_neighbors=5)),
        AlternativeClassifier("RBF SVM", SVC(kernel='rbf', C=1)),
        AlternativeClassifier("Random Forest", RandomForestClassifier(n_estimators=100, max_features='sqrt')),
    ]
    trainAlternatives(classifiers, ((trainX, trainY), (testX, testY)), resampleFrequency=resampleFrequency)
