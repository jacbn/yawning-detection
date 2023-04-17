# Includes the SessionData, SensorReading and Timestamp classes.
# Create a SessionData via SessionData.fromPath(filepath), where filepath the path to a .eimu file
from yawnnlib.utils import commons, filters
from yawnnlib.structure.sensorReading import SensorReading
from yawnnlib.structure.timestamp import Timestamp

import numpy as np
from matplotlib import pyplot as plt 

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
        self.accel = list(map(lambda x: x.accel, dataset))
        self.gyro = list(map(lambda x: x.gyro, dataset))
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
        """ Create a SessionData object from a .eimu file. 
        Note that this will apply a correction to the yawn times. """
        data = []
        with open(filepath, "r") as f:
            lines = f.read().splitlines()
            version = int(lines[0])
            timestamps = []
            sampleRate = int(lines[2])
            numTimestamps = int(lines[3])
            for i in range(4, 4 + numTimestamps):
                split = lines[i].split(' ')
                correction = round(commons.YAWN_CORRECTION * sampleRate)
                timestamps.append(Timestamp(max(0, int(split[0]) + correction), split[1]))
            for i in range(4 + numTimestamps, len(lines)):
                if lines[i]:
                    data.append(SensorReading.fromString(lines[i]))

        return cls(data, timestamps, sampleRate, version, fileNum, totalFiles)
    
    @classmethod
    def from6DDataVectors(cls, data : list[list[float]], timestamps : list[Timestamp], sampleRate : int, version : int, fileNum : int = -1, totalFiles : int = -1):
        """ Create a SessionData object from a 6D data vector. 
        Note that this will not apply a correction to the yawn times. """
        sensorReadings = list(map(lambda x: SensorReading(x[:3], x[3:]), data))
        return cls(sensorReadings, timestamps, sampleRate, version, fileNum, totalFiles)
        
    def getEimuData(self, windowSize : float, windowSep : float, dataFilter : filters.DataFilter = filters.NoneFilter()) -> tuple[np.ndarray, list[list[Timestamp]]]:
        """ Returns the data to input to the model.
        
        Attributes
        ----------
        dataFilter : filters.DataFilter = filters.NoneFilter()
            the filter to apply to the data
        windowSize : float
            the number of seconds in each split.
        windowSep : float
            the number of seconds between each split.

        Returns:
            np.ndarray: an array of arrays of shape (windowSize, 6), with each row a 6D vector of the accel and gyro data
            list[Timestamp]: a list of timestamps for each point in the session
        """
        
        trueWindowSize = int(windowSize * self.sampleRate)
        trueWindowSep = int(windowSep * self.sampleRate)
        
        session = self
        
        if dataFilter.getApplyType() in [filters.ApplyType.SESSION, filters.ApplyType.MULTIPLE]:
            session = self.applyFilter(self, dataFilter, filters.ApplyType.SESSION)
        
        splits = session.splitSession(windowSize=trueWindowSize, windowSep=trueWindowSep)
        
        if dataFilter.getApplyType() in [filters.ApplyType.SPLIT, filters.ApplyType.MULTIPLE]:
            splits = list(map(lambda x: self.applyFilter(x, dataFilter, filters.ApplyType.SPLIT), splits))
            
        arr = np.array(list(map(lambda x: x.get6DDataVectors(), splits)))

        # resize arr if the split's length is less than the expected session width
        if arr.shape[1] < trueWindowSize:
            arr.resize((len(splits), trueWindowSize, 6))
        
        return arr, list(map(lambda x: x.timestamps, splits))
    
    @staticmethod
    def applyFilter(session : 'SessionData', dataFilter : filters.DataFilter, filterType : filters.ApplyType):
        """ Applies a filter to the session data.

        Attributes
        ----------
        sessionData : SessionData
            the session data to apply the filter to
        dataFilter : filters.DataFilter
            the filter to apply

        Returns:
            SessionData: the filtered session data
        """
        
        data = session.get6DDataVectors()
        
        if isinstance(dataFilter, filters.TimestampedDataFilter):
            timestamps = dataFilter.manageTimestamps(data, session.timestamps)
        else:
            timestamps = session.timestamps
        
        if isinstance(dataFilter, filters.FilterCollection):
            applied = dataFilter.applyByType(data, filterType).tolist()
        else:
            applied = dataFilter.apply(data).tolist()
        
        return session.from6DDataVectors(
            applied,
            timestamps,
            session.sampleRate,
            session.version,
            session.fileNum,
            session.totalFiles
        )
        
        
    def splitSession(self, windowSize : int, windowSep : int):
        """Splits one SessionData into a list of smaller SessionData, each of length commons.YAWN_TIME seconds.
        
        Attributes
        ----------
        
        windowSep : int
            The number of samples forward to move between each split. Minimum 1.
        windowSize : int
            The number of samples in each split. Minimum 1.

        Returns
        -------
            list[SessionData]: The list of smaller SessionData objects.
        """
        assert windowSize > 0
        assert windowSep > 0
        
        if self.numPoints < windowSize:
            print("Session too short to split. Padding original session.")
            return [self]
        
        converted = []
        for i in range(0, self.numPoints - windowSize + 1, windowSep):
            converted.append(SessionData(
                self._toSensorReadings(self.accel[i:i+windowSize], self.gyro[i:i+windowSize]),
                self._getRelevantTimestamps(self.timestamps, i, i + windowSize),
                self.sampleRate,
                self.version
            ))
        return converted
    
    def get6DDataVectors(self):
        """ Convert accel and gyro into one list of 6D vectors. """
        return np.array(list(map(lambda x: sum(x, start=[]), zip(self.accel, self.gyro)))) #type: ignore
    
    def getYawnIndices(self):
        """ Return a list of 0s and 1s for each point in the session, where 1s represent the presence of a yawn timestamp at most one YAWN_TIME//2 seconds before or after the point. """

        # get the timestamps of when the yawn button was pressed
        yawns = list(filter(lambda x: x.type == "yawn", self.timestamps))
                
        # define functions for obtaining the indices of points within YAWN_TIME//2 seconds of a yawn
        rangeMin = lambda x: max(0, int(x.time-self.sampleRate*commons.YAWN_TIME//2))
        rangeMax = lambda x: min(self.numPoints, int(x.time+self.sampleRate*commons.YAWN_TIME//2+1))
        
        # apply these to each yawn timestamp, then flatten the list of lists into a single list
        yawnTimes = sum(list(map(lambda x: list(range(rangeMin(x), rangeMax(x))), yawns)), start=[])
        
        # create a numpy array of 0s and 1s, where 1s represent the presence of a yawn nearby
        t = np.zeros(self.numPoints)
        t[yawnTimes] = 1
        return t
    
    
    @staticmethod
    def _toSensorReadings(accel : list[list[float]], gyro : list[list[float]]) -> list[SensorReading]:
        """ Convert a list of accel and gyro data into a list of SensorReading objects. """
        return list(map(lambda x: SensorReading(x[0], x[1]), zip(accel, gyro)))
    
    @staticmethod
    def _getRelevantTimestamps(timestamps : list[Timestamp], start : int, end : int) -> list[Timestamp]:
        """ Filter only those timestamps in the range (inclusive, measured in number of samples), then shift their times to match. """
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
    
    def plot(self, show=True, figure : int = 1, unitConversion : bool = True):
        """ Plot the accel and gyro data for this session. """
        plt.figure(figure)
        ax1 = plt.subplot(211)
        ax1.set_title("Accelerometer", fontsize=8)
        ax1.set_ylabel(f"Acceleration ({'m/s^2' if unitConversion else 'normalised'})")
        for i, axis in enumerate('xyz'):
            plt.plot(range(0, self.numPoints), list(map(lambda x: (self.accelConversion(x) if unitConversion else x)[i], self.accel)), label=axis)
        for timestamp in self.timestamps:
            ax1.axvline(timestamp.time, color='black')
        plt.grid()
        plt.tick_params('x', labelbottom=False)
        plt.legend(loc="upper right")

        ax2 = plt.subplot(212, sharex=ax1)
        ax2.set_title("Gyroscope", fontsize=8)
        for i, axis in enumerate('xyz'):
            plt.plot(range(0, self.numPoints), list(map(lambda x: (self.gyroConversion(x) if unitConversion else x)[i], self.gyro)), label=axis)
        for timestamp in self.timestamps:
            ax2.axvline(timestamp.time, color='black')
        ax2.set_xlabel(f"Samples ({self.sampleRate} = 1 sec)")
        ax2.set_ylabel(f"Gyro ({'deg/s' if unitConversion else 'normalised'})")
        
        tick = pow(2, np.ceil(np.log(self.numPoints)/np.log(2)))/16
        plt.xticks(np.arange(0, self.numPoints, tick))
        plt.grid()
        plt.suptitle("Session Data")
        if show:
            plt.show()
            
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SessionData):
            return False
        return self.accel == other.accel and self.gyro == other.gyro and self.timestamps == other.timestamps and self.sampleRate == other.sampleRate and self.version == other.version

if __name__ == "__main__":
    # s = SessionData.fromPath(f"{commons.PROJECT_ROOT}/data/tests/96hz/96hz-yawns1.eimu")
    # print(s.getEimuData(commons.YAWN_TIME//2, commons.YAWN_TIME//4)[0].shape)
    # s.plot(show=True)
    
    # below shows a great example of how the smooth filter works
    s = SessionData.fromPath(f"{commons.PROJECT_ROOT}/data/tests/96hz/96hz-yawns1.eimu")
    s.plot(show=False, figure=3)
    
    # smooth filter application:
    # w = s.sampleRate * commons.YAWN_TIME * 1.5
    # g = s.sampleRate * commons.YAWN_TIME / 5.3
    # splits = s.splitSession(windowSize=int(w), windowSep=int(g))
    # fSplit = SessionData.applyFilter(splits[2], dataFilter = filters.SmoothFilter(0.7), filterType=filters.ApplyType.SPLIT)
    # splits[2].plot(show=False)
    # fSplit.plot(show=True, figure=2)
    
    s2 = SessionData.applyFilter(s, dataFilter = filters.HighPassFilter(96, 0.05, 30), filterType=filters.ApplyType.SESSION)
    s2.plot(show=True, figure=2)