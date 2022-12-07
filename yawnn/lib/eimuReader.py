# Converts a raw .eimu file input into a SessionData object
# via getSession(filepath).

import numpy as np
import eimuSplitter
from matplotlib import pyplot as plt 

class SensorReading:
    def __init__(self, accel, gyro):
        self.accel = accel
        self.gyro = gyro
        
class Timestamp:
    def __init__(self, time, ttype):
        self.time = time
        self.type = ttype

def rawAccelConversion(accel):
    return list(map(lambda x: x/8192, accel))

def rawGyroConversion(gyro):
    return list(map(lambda x: x/65.5, gyro))

class SessionData:
    def __init__(self, dataset, timestamps, sampleRate):
        self.accel = list(map(lambda x: rawAccelConversion(x.accel), dataset))
        self.gyro = list(map(lambda x: rawGyroConversion(x.gyro), dataset))
        self.accelLim = max(abs(min(map(min, self.accel))), abs(max(map(max, self.accel))))
        self.gyroLim = max(abs(min(map(min, self.gyro))), abs(max(map(max, self.gyro))))
        self.numPoints = len(dataset)
        self.timestamps = timestamps
        self.sampleRate = sampleRate
        
    def toRaw(self, sessions):
        # return a numpy array of shape (numPoints, 3, 2), with the 3 being the x, y, z axes of accel and gyro
        # and the 2 being the accel and gyro data
        return np.array(list(map(lambda x: np.array([x.accel, x.gyro]), sessions)))
        
    
    def toTensorflowData(self):
        splits = eimuSplitter.splitSession(self)
        return [self.toRaw(s) for s in splits]
    
    def plot(self):
        pass # todo
            
def toSensorReadingString(string):
    splits = string.split('[')
    f = filter(lambda v: v, splits)
    m = map(lambda w: w.replace('[', '').replace(',', '').replace(']', ''), f)
    accelAndGyroLists = list(m)
    # accelAndGyroLists = list(filter(lambda w: w.replace('[', '').replace(',', '').replace(']', ''), lambda v: v), string.split('[')))
    # string.split('[').filter(lambda v: v).map(lambda w: w.replace('[', '').replace(',', '').replace(']', '')).toList()
    
    splits = accelAndGyroLists[0].split(' ')
    f = filter(lambda v: v, splits)
    m1 = map(lambda v: int(v), f)
    splits = accelAndGyroLists[1].split(' ')
    f = filter(lambda v: v, splits)
    m2 = map(lambda v: int(v), f)
    
    reading = SensorReading(list(m1), list(m2))
    assert len(reading.accel) == 3 and len(reading.gyro) == 3
    return reading

def getSession(filepath):
    data = []
    with open(filepath, "r") as f:
        lines = f.read().splitlines()
        version = int(lines[0])
        timestamps = []
        sampleRate = int(lines[2])
        numTimestamps = int(lines[3])
        for i in range(4, 4 + numTimestamps):
            split = lines[i].split(' ')
            timestamps.append(Timestamp(int(split[0]), split[1]))
        for i in range(4 + numTimestamps, len(lines)):
            if lines[i]:
                data.append(toSensorReadingString(lines[i]))

    return SessionData(data, timestamps, sampleRate)

# TODO: build app onto phone, record new data, put in data, getSession (below), fix up toRaw(), then hopefully
# TODO: an input of size (numPoints, 3, 2, t) can be used in the lstm model? or find something that does

# https://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/
# https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/
# https://keras.io/api/layers/recurrent_layers/lstm/

s = getSession("./yawnn/data/1.eimu")
print(s.toRaw())