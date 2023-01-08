import commons
from fourierLSTM import FourierLSTMInput
from eimuLSTM import EimuLSTMInput

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

def makeLSTMDense(lstmUnits : int):
    model = Sequential()
    model.add(LSTM(units=lstmUnits, return_sequences=True))
    model.add(Dense(units=1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def trainModel(modelType : commons.ModelType, dataDirectory : str, model, epochs : int, shuffle=True, equalPositiveAndNegative=True):
    (trainX, trainY), (testX, testY) = commons.directoryToModelData(dataDirectory, modelType, shuffle=shuffle, equalPositiveAndNegative=equalPositiveAndNegative)
    
    model.fit(trainX, trainY, epochs=epochs, batch_size=32)
    model.evaluate(testX, testY)
    
    
if __name__ == "__main__":
    # trainModel(EimuLSTMInput(), makeLSTMDense(lstmUnits=10), "./yawnn/data/96hz", epochs=100)
    trainModel(FourierLSTMInput(), makeLSTMDense(lstmUnits=10), "./yawnn/data/96hz", epochs=100)
    