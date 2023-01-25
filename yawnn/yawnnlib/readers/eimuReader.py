# Includes the SessionData, SensorReading and Timestamp classes.
# Create a SessionData via SessionData.fromPath(filepath), where filepath the path to a .eimu file
from yawnnlib.commons import commons

import numpy as np
from matplotlib import pyplot as plt 

class SensorReading:
    """ A class representing a single reading from the sensor. """
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
    """ A class representing a timestamp as found in a .eimu file, i.e. a single point in time. """
    def __init__(self, time : int, ttype : str):
        self.time = time
        self.type = ttype

class SessionData:
    """ A class representing a single session of data. 
    
    Attributes
    ----------
    rawDataset : list[SensorReading]
        the unmodified data as read from the .eimu file this session is loaded from
    accel : list[list[float]]
        the acceleration data, in standard units (m/s^2)
    gyro : list[list[float]]
        the gyroscope data, in standard units (rad/s)
    accelLim : float
        maximum absolute value of the acceleration data
    gyroLim : float
        maximum absolute value of the gyroscope data
    numPoints : int
        number of points in the session
    timestamps : list[Timestamp]
        the timestamps for the session
    sampleRate : int
        sample rate of the session
    version : int
        version of the .eimu file this session is loaded from
    fileNum : int
        used exclusively for splitting a session; the split number. -1 if not used
    totalFiles : int
        used exclusively for splitting a session; the total number of splits. -1 if not used 
    """
    
    def __init__(self, dataset : list[SensorReading], timestamps : list[Timestamp], sampleRate : int, version : int, fileNum : int = -1, totalFiles : int = -1):
        """ Constructs the necessary attributes for the SessionData object.

        Attributes
        ----------
        dataset : list[SensorReading]
            the dataset represented as a series of (6-dimensional) SensorReadings
        timestamps : list[Timestamp]
            the timestamps for the session
        sampleRate : int
            sample rate of the session
        version : int
            version of the .eimu file this session is loaded from
        fileNum : optional int
            used exclusively for splitting a session; the split number. -1 if not used
        totalFiles : optional int
            used exclusively for splitting a session; the total number of splits. -1 if not used
        """
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
        self.fileNum = fileNum
        self.totalFiles = totalFiles
        
    @classmethod
    def fromPath(cls, filepath : str, fileNum : int = -1, totalFiles : int = -1):
        """ Create a SessionData object from a .eimu file. """
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

        return cls(data, timestamps, sampleRate, version, fileNum, totalFiles)
    
    @classmethod
    def from6DDataVectors(cls, data : list[list[float]], timestamps : list[Timestamp], sampleRate : int, version : int, fileNum : int = -1, totalFiles : int = -1):
        """ Create a SessionData object from a 6D data vector. """
        sensorReadings = list(map(lambda x: SensorReading(x[:3], x[3:]), data))
        return cls(sensorReadings, timestamps, sampleRate, version, fileNum, totalFiles)
        
    def getEimuData(self):
        """ Returns the data required to input to the LSTM model.

        Returns:
            np.ndarray: an array of arrays of shape (YAWN_TIME * sampleRate, 6), with each row a 6D vector of the accel and gyro data
            list[int]: a list of timestamps for each point in the session
        """
        splits = self.splitSession()
        arr = np.array(list(map(lambda x: x.get6DDataVector(), splits)))
        arr.resize(len(splits), commons.YAWN_TIME * self.sampleRate, 6)
        return arr, list(map(lambda x: x.timestamps, splits))
    

    def splitSession(self, sessionGap : int = 3):
        """Splits one SessionData into a list of smaller SessionData, each of length commons.YAWN_TIME seconds.
        sessionGap represents how many samples forward to move between each split. Minimum 1, default 3.

        Returns:
            list[SessionData]: the list of smaller SessionData objects
        """
        assert sessionGap > 0
        
        samplesPerGroup = commons.YAWN_TIME * self.sampleRate
    
        if self.numPoints < samplesPerGroup:
            print("Session too short to split. Returning original session.")
            return [self]
        
        converted = []
        for i in range(0, self.numPoints - samplesPerGroup, sessionGap):
            converted.append(SessionData(
                self._toSensorReadings(self.accel[i:i+samplesPerGroup], self.gyro[i:i+samplesPerGroup]),
                self._getRelevantTimestamps(self.timestamps, i, i + samplesPerGroup),
                self.sampleRate,
                self.version
            ))
        return converted
    
    def get6DDataVector(self):
        """ Convert accel and gyro into one list of 6D vectors. """
        return np.array(list(map(lambda x: sum(x, start=[]), zip(self.accel, self.gyro)))) #type: ignore
    
    def getYawnIndices(self):
        """ Return a list of 0s and 1s for each point in the session, where 1s represent the presence of a yawn timestamp at most one YAWN_TIME//2 seconds before or after the point. """
        yawnTimes = sum(list(map(lambda x: list(range(max(0, x.time-self.sampleRate*commons.YAWN_TIME//2), min(self.numPoints, x.time+self.sampleRate*commons.YAWN_TIME//2+1))), list(filter(lambda x: x.type == "yawn", self.timestamps)))), start=[])
        t = np.zeros(self.numPoints)
        t[yawnTimes] = 1
        return t
    
    
    @staticmethod
    def _toSensorReadings(accel : list[list[float]], gyro : list[list[float]]) -> list[SensorReading]:
        """ Convert a list of accel and gyro data into a list of SensorReading objects. """
        return list(map(lambda x: SensorReading(x[0], x[1]), zip(accel, gyro)))
    
    @staticmethod
    def _getRelevantTimestamps(timestamps : list[Timestamp], start : int, end : int) -> list[Timestamp]:
        """ Filter only those timestamps in the range, then shift their times to match. """
        return list(map(lambda x: Timestamp(x.time - start, x.type), filter(lambda t: t.time >= start and t.time <= end, timestamps)))
    
    @staticmethod
    def accelConversion(accel : list[float]) -> list[float]:
        """ Convert raw accel units to m/s^2. """
        return list(map(lambda x: x/8192, accel))

    @staticmethod
    def gyroConversion(gyro : list[float]) -> list[float]:
        """ Convert raw gyro units to deg/s. """
        return list(map(lambda x: x/65.5, gyro))
    
    @staticmethod
    def combineSessions(sessions):
        # todo
        pass
    
    def plot(self, show=True, figure : int = 1):
        """ Plot the accel and gyro data for this session. """
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
    