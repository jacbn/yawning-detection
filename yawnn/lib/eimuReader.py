# Includes the SessionData, SensorReading and Timestamp classes.
# Create a SessionData via SessionData.fromPath(filepath), where filepath the path to a .eimu file
import commons

import numpy as np
from matplotlib import pyplot as plt 

class SensorReading:
    def __init__(self, accel : list[float], gyro : list[float]):
        self.accel = accel
        self.gyro = gyro
        
    @classmethod
    def fromString(cls, string : str):
        nonEmptySplits = filter(lambda v: v, string.split('['))
        accelAndGyroLists = list(map(lambda w: w.replace('[', '').replace(',', '').replace(']', ''), nonEmptySplits))
        
        m1 = map(float, filter(lambda v: v, accelAndGyroLists[0].split(' ')))
        m2 = map(float, filter(lambda v: v, accelAndGyroLists[1].split(' ')))
        
        reading = cls(list(m1), list(m2))
        assert len(reading.accel) == 3 and len(reading.gyro) == 3
        return reading
        
class Timestamp:
    def __init__(self, time : int, ttype : str):
        self.time = time
        self.type = ttype

# a session is one entire recording, with no set length.
# session.toRaw() returns a tuple of (data, timestamps), 
#  where data is a list of 2 second chunks of data
class SessionData:
    def __init__(self, dataset : list[SensorReading], timestamps : list[Timestamp], sampleRate : int, version : int):
        self.rawDataset = dataset,
        # TODO: is converting x.accel and x.gyro to standard units useful for the NN or do we only need it for graphing purposes?
        self.accel = list(map(lambda x: self.accelConversion(x.accel), dataset))
        self.gyro = list(map(lambda x: self.gyroConversion(x.gyro), dataset))
        self.accelLim = max(abs(min(map(min, self.accel))), abs(max(map(max, self.accel))))
        self.gyroLim = max(abs(min(map(min, self.gyro))), abs(max(map(max, self.gyro))))
        self.numPoints = len(dataset)
        self.timestamps = timestamps
        self.sampleRate = sampleRate
        self.version = version
        
    @classmethod
    def fromPath(cls, filepath : str):
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
                    data.append(SensorReading.fromString(lines[i]))

        return cls(data, timestamps, sampleRate, version)
        
    def getEimuData(self):
        splits = self.splitSession()
        # return an array of numpy arrays of shape (64, 6),
        # with 64 being the number of points per session, 
        # 6 being the x, y, z axes for the accel and gyro data, respectively
        arr = np.array(list(map(lambda x: np.array([*x.accel, *x.gyro]), splits)))
        arr.resize(len(splits), 64, 6)
            
        # return these arrays, alongside a list timestamps for each
        return arr, list(map(lambda x: x.timestamps, splits))
        # final format is an np array (length = len(splits)) of (64, 3, 2) data arrays,
        # paired with another array (length = len(splits)) of lists of timestamps per split
    
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
    
    # turn the 2 lists of 3D vectors into one list of 6D vectors
    def get6DDataVector(self):
        return list(map(lambda x: sum(x, start=[]), zip(self.accel, self.gyro)))
    
    # return a list of 0s and 1s for each point in the session, where 1s represent the presence of a yawn at most one YAWN_TIME//2 seconds before or after the point
    def getYawnIndices(self):
        yawnTimes = sum(list(map(lambda x: list(range(max(0, x.time-self.sampleRate*commons.YAWN_TIME//2), min(self.numPoints, x.time+self.sampleRate*commons.YAWN_TIME//2+1))), list(filter(lambda x: x.type == "yawn", self.timestamps)))), start=[])
        t = np.zeros(self.numPoints)
        t[yawnTimes] = 1
        return t
    
    # Convert a list of accel and gyro data into a list of SensorReading objects
    @staticmethod
    def _toSensorReadings(accel : list[list[float]], gyro : list[list[float]]) -> list[SensorReading]:
        return list(map(lambda x: SensorReading(x[0], x[1]), zip(accel, gyro)))
    
    # filter only those timestamps in the range, then shift their times to match
    @staticmethod
    def _getRelevantTimestamps(timestamps : list[Timestamp], start : int, end : int) -> list[Timestamp]:
        return list(map(lambda x: Timestamp(x.time - start, x.type), filter(lambda t: t.time >= start and t.time <= end, timestamps)))
    
    # convert raw accel units to m/s^2
    @staticmethod
    def accelConversion(accel : list[float]) -> list[float]:
        return list(map(lambda x: x/8192, accel))

    # convert raw gyro units to deg/s
    @staticmethod
    def gyroConversion(gyro : list[float]) -> list[float]:
        return list(map(lambda x: x/65.5, gyro))
    
    @staticmethod
    def combineSessions(sessions):
        # todo
        pass
    
    def plot(self, show=True, figure : int = 1):
        plt.figure(figure)
        ax1 = plt.subplot(211)
        ax1.set_title("Accelerometer", fontsize=8)
        ax1.set_ylabel("Acceleration (m/s^2)")
        for i, axis in enumerate('xyz'):
            plt.plot(range(0, self.numPoints), list(map(lambda x: x[i], self.accel)), label=axis)
        for timestamp in self.timestamps:
            ax1.axvline(timestamp.time, color='black')
        plt.grid()
        plt.tick_params('x', labelbottom=False)
        plt.legend(loc="upper right")

        ax2 = plt.subplot(212, sharex=ax1)
        ax2.set_title("Gyroscope", fontsize=8)
        for i, axis in enumerate('xyz'):
            plt.plot(range(0, self.numPoints), list(map(lambda x: x[i], self.gyro)), label=axis)
        for timestamp in self.timestamps:
            ax2.axvline(timestamp.time, color='black')
        ax2.set_xlabel(f"Samples ({self.sampleRate} = 1 sec)")
        ax2.set_ylabel("Gyro (deg/s)")
        
        tick = pow(2, np.ceil(np.log(self.numPoints)/np.log(2)))/16
        plt.xticks(np.arange(0, self.numPoints, tick))
        plt.grid()
        plt.suptitle("Session Data")
        if show:
            plt.show()

if __name__ == "__main__":
    s = SessionData.fromPath("./yawnn/data/long1.eimu")
    print(s.getEimuData()[0].shape)
    
    
# https://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/
# https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/
# https://keras.io/api/layers/recurrent_layers/lstm/
