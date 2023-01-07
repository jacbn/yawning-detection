import commons
from fourierLSTM import FourierLSTMInput
from eimuLSTM import EimuLSTMInput

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

# todo: make an LSTM class that has a variable model
def makeLSTM(modelType : commons.ModelType, dataDirectory : str, lstmUnits : int, epochs : int, shuffle=True, equalPositiveAndNegative=True):
    model = Sequential()
    model.add(LSTM(units=lstmUnits, return_sequences=True))
    model.add(Dense(units=1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    (trainX, trainY), (testX, testY) = commons.directoryToModelData(dataDirectory, modelType, shuffle=shuffle, equalPositiveAndNegative=equalPositiveAndNegative)

    model.fit(trainX, trainY, epochs=epochs, batch_size=32)
    model.evaluate(testX, testY)
    
if __name__ == "__main__":
    # makeLSTM(EimuLSTMInput(), "./yawnn/data/96hz", 10, 100)
    makeLSTM(FourierLSTMInput(), "./yawnn/data/96hz", lstmUnits=10, epochs=100)
    