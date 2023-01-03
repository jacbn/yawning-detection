import commons
from eimuReader import SessionData

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

TIMESTAMP_PREDICATE = lambda tList: sum(map(lambda t: t.type == 'yawn', tList))
LSTM_UNITS = 10
EPOCHS = 100

# convert a single eimu file to a tuple of (trainX, trainY), (testX, testY)
def eimuToLSTMInput(eimuPath : str) -> tuple[tuple, tuple]:
    session = SessionData.fromPath(eimuPath)
    data, timestamps = session.getEimuData()
    predicates = np.array(list(map(TIMESTAMP_PREDICATE, timestamps)))
    predicates.resize(predicates.shape[0], 1)

    trainLength = int(len(data) * commons.TRAIN_PERCENT)
    return (data[:trainLength], predicates[:trainLength]), (data[trainLength:], predicates[trainLength:])

# convert a directory of eimu files to a tuple of (trainX, trainY), (testX, testY)
def directoryToLSTMInput(path : str, shuffle : bool = True) -> tuple[tuple, tuple]:
    inputs = commons.mapToDirectory(eimuToLSTMInput, path)
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
    
    (trainX, trainY), (testX, testY) = directoryToLSTMInput("./yawnn/data")
    model.fit(trainX, trainY, epochs=EPOCHS, batch_size=32)
    model.evaluate(testX, testY)
    