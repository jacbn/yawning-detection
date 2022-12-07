# Converts a raw .eimu file input into a SessionData object
# via getSession(filepath).

import numpy as np
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
    def __init__(self, dataset : list[SensorReading], timestamps : list[Timestamp], sampleRate : int, version : int):
        self.rawDataset = dataset,
        self.accel = list(map(lambda x: rawAccelConversion(x.accel), dataset))
        self.gyro = list(map(lambda x: rawGyroConversion(x.gyro), dataset))
        self.accelLim = max(abs(min(map(min, self.accel))), abs(max(map(max, self.accel))))
        self.gyroLim = max(abs(min(map(min, self.gyro))), abs(max(map(max, self.gyro))))
        self.numPoints = len(dataset)
        self.timestamps = timestamps
        self.sampleRate = sampleRate
        self.version = version
        
    def toRaw(self):
        splits = self.splitSession()
        # return an array of numpy arrays of shape (64, 3, 2),
        # with 64 being the number of points per session, 
        # 3 being the x, y, z axes, and
        # 2 being the accel and gyro data.
        arr = np.array(list(map(lambda x: np.array([x.accel, x.gyro]), splits)))
        arr.resize(len(splits), 64, 3, 2)
            
        # return these arrays, alongside a list timestamps for each
        return arr, list(map(lambda x: x.timestamps, splits))
        
    
    def toTensorflowData(self):
        splits = self.splitSession(self)
        return [self.toRaw(s) for s in splits]
    
    # Split a SessionData object into a list of SessionData objects,
    # each of the same N-second length.
    def splitSession(self):
        samplesPerGroup = self.sampleRate * 2 # 2 seconds per group
    
        if self.numPoints < samplesPerGroup:
            print("Session too short to split. Returning original session.")
            return [self]
        
        converted = []
        for i in range(self.numPoints - samplesPerGroup):
            converted.append(SessionData(
                self._toSensorReadings(self.accel[i:i+samplesPerGroup], self.gyro[i:i+samplesPerGroup]),
                self._getRelevantTimestamps(self.timestamps, i, i + samplesPerGroup),
                self.sampleRate,
                self.version
            ))
        return converted
    
    # Convert a list of accel and gyro data into a list of SensorReading objects
    def _toSensorReadings(self, accel, gyro):
        return list(map(lambda x: SensorReading(x[0], x[1]), zip(accel, gyro)))
    
    # filter only those timestamps in the range, then shift their times to match
    def _getRelevantTimestamps(self, timestamps, start, end):
        return list(map(lambda x: x - start, filter(lambda t: t.time >= start and t.time <= end, timestamps)))
    
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

    return SessionData(data, timestamps, sampleRate, version)

# TODO: build app onto phone, record new data, put in data, getSession (below), fix up toRaw(), then hopefully
# TODO: an input of size (numPoints, 3, 2, t) can be used in the lstm model? or find something that does

# https://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/
# https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/
# https://keras.io/api/layers/recurrent_layers/lstm/

s = getSession("./yawnn/data/yawn-1.eimu")
print(s.toRaw())