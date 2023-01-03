import commons
from fourierReader import FourierData

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

TIMESTAMP_PREDICATE = lambda tList: sum(map(lambda t: t.type == 'yawn', tList))
LSTM_UNITS = 25
EPOCHS = 25

# convert a single eimu file to a tuple of (trainX, trainY), (testX, testY)
def eimuToFourierLSTMInput(eimuPath : str, trainOnTimeAxis=False) -> tuple[tuple, tuple]:
    session = FourierData.fromPath(eimuPath)
    data, timestamps = session.getFourierData(chunkSize=commons.YAWN_TIME, chunkSeparation=commons.YAWN_TIME/4)
    
    # data format is (axes, chunks, frequencies, times (samples) per chunk).
    ax, ch, fs, ts = data.shape
    # the number of chunks is variable based on input data, the others depend on constants.
    # we can either train on (axes, frequency, time) tuples, or (axes, frequency) tuples.
    # the former will detect patterns through time, but will need significantly more data.
    
    # for first method, we reshape the data to (chunks, axes, frequencies, times), iterating through will give the tuples.
    # for the second, we reshape the data to (chunks, times, axes, frequencies), then squash the first two axes.
    
    if trainOnTimeAxis:
        data = np.reshape(data, (ch, ax, fs, ts))
        predicates = np.array(timestamps)
        # todo: resize
    else:
        data = np.reshape(data, (ch * ts, ax, fs))
        predicates = np.array([timestamps[chunk] for chunk in range(ch) for _ in range(ts)])
        predicates.resize(predicates.shape[0], 1)
    
    trainLength = int(len(data) * commons.TRAIN_PERCENT)
    return (data[:trainLength], predicates[:trainLength]), (data[trainLength:], predicates[trainLength:])


# convert a directory of eimu files to a tuple of (trainX, trainY), (testX, testY)
def directoryToLSTMInput(path : str, shuffle : bool = True) -> tuple[tuple, tuple]:
    inputs = commons.mapToDirectory(eimuToFourierLSTMInput, path)
    # combine all the inputs. each is a tuple of (trainX, trainY), (testX, testY),
    # and the result is a combination of all the trainX, trainY, testX, testY individually
    combined = (np.concatenate(list(map(lambda x: x[0][0], inputs))), np.concatenate(list(map(lambda x: x[0][1], inputs)))), (np.concatenate(list(map(lambda x: x[1][0], inputs))), np.concatenate(list(map(lambda x: x[1][1], inputs))))
    
    if shuffle:
        combined = commons.shuffleTrainingData(combined)
        
    return combined
    
if __name__ == "__main__":
    
    model = Sequential()
    model.add(LSTM(units=LSTM_UNITS, return_sequences=True))
    model.add(Dense(units=1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    (trainX, trainY), (testX, testY) = directoryToLSTMInput("./yawnn/data/96hz")
    # (trainX, trainY), (testX, testY) = eimuToFourierLSTMInput("./yawnn/data/96hz-yawns1.eimu")
    model.fit(trainX, trainY, epochs=EPOCHS, batch_size=32)
    model.evaluate(testX, testY)