class SensorReading:
    """ A class representing a single reading from the sensor. """
    def __init__(self, accel : list[float], gyro : list[float]) -> None:
        self.accel = accel
        self.gyro = gyro
        
    @classmethod
    def fromString(cls, string : str) -> 'SensorReading':
        nonEmptySplits = filter(lambda v: v, string.split('['))
        accelAndGyroLists = list(map(lambda w: w.replace('[', '').replace(',', '').replace(']', ''), nonEmptySplits))
        
        m1 = map(float, filter(lambda v: v, accelAndGyroLists[0].split(' ')))
        m2 = map(float, filter(lambda v: v, accelAndGyroLists[1].split(' ')))
        
        reading = cls(list(m1), list(m2))
        assert len(reading.accel) == 3 and len(reading.gyro) == 3
        return reading
    
    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, SensorReading):
            return False
        return self.accel == __o.accel and self.gyro == __o.gyro