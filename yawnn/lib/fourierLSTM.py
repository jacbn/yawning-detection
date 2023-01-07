import commons
from fourierReader import FourierData

import numpy as np

TIMESTAMP_PREDICATE = lambda tList: sum(map(lambda t: t.type == 'yawn', tList))

# convert a single eimu file to a tuple of (trainX, trainY), (testX, testY)
def eimuToFourierLSTMInput(eimuPath : str, trainOnTimeAxis=False) -> commons.ModelData:
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
        annotations = np.array(timestamps)
        # todo: resize
    else:
        data = np.reshape(data, (ch * ts, ax, fs))
        annotations = np.array([timestamps[chunk] for chunk in range(ch) for _ in range(ts)])
        annotations.resize(annotations.shape[0], 1)
    
    trainLength = int(len(data) * commons.TRAIN_PERCENT)
    return (data[:trainLength], annotations[:trainLength]), (data[trainLength:], annotations[trainLength:])

class FourierLSTMInput(commons.ModelType):
    def __init__(self, trainOnTimeAxis=False):
        self.trainOnTimeAxis = trainOnTimeAxis
    
    def fromPath(self, path : str) -> commons.ModelData:
        return eimuToFourierLSTMInput(path, self.trainOnTimeAxis)
