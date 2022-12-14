import commons
print("Loading imports...")
from fourierLSTM import FourierLSTMInput
from eimuLSTM import EimuLSTMInput

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
print("Imports loaded.")


def makeSequentialModel(layers : list):
    model = Sequential()
    for layer in layers:
        model.add(layer)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def trainModel(modelType : commons.ModelType, model, dataDirectory : str, epochs : int, shuffle : bool = True, equalPositiveAndNegative : bool = True):
    print(f"\nTraining {modelType.getType()}:")
    
    (trainX, trainY), (testX, testY) = commons.directoryToModelData(dataDirectory, modelType, shuffle=shuffle, equalPositiveAndNegative=equalPositiveAndNegative)
    
    model.fit(trainX, trainY, epochs=epochs, batch_size=32)
    model.save(f"./yawnn/models/{modelType.getType()}.h5")
    model.evaluate(testX, testY)
    
    
if __name__ == "__main__":
    # trainModel(EimuLSTMInput(), makeLSTMDense(lstmUnits=10), "./yawnn/data/96hz", epochs=100)
    trainModel(FourierLSTMInput(), 
               makeSequentialModel([
                   LSTM(units=128, recurrent_dropout=0.2, return_sequences=True),
                   LSTM(units=64, recurrent_dropout=0.2, return_sequences=True),
                   Dense(units=1, activation='sigmoid')]),
               "./yawnn/data/user-trials", epochs=100)
    