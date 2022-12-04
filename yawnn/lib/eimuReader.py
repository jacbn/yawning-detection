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
    def __init__(self, dataset, timestamps, sampleRate):
        self.accel = list(map(lambda x: rawAccelConversion(x.accel), dataset))
        self.gyro = list(map(lambda x: rawGyroConversion(x.gyro), dataset))
        self.accelLim = max(abs(min(map(min, self.accel))), abs(max(map(max, self.accel))))
        self.gyroLim = max(abs(min(map(min, self.gyro))), abs(max(map(max, self.gyro))))
        self.numPoints = len(dataset)
        self.timestamps = timestamps
        self.sampleRate = sampleRate
        
            
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


