import unittest
from yawnnlib.structure import sensorReading

class TestSessionData(unittest.TestCase):
    def test_fromString(self):
        self.assertEqual(sensorReading.SensorReading.fromString("[1, 2, 3] [4, 5, 6]"), sensorReading.SensorReading([1, 2, 3], [4, 5, 6]))
        
if __name__ == "__main__":
    unittest.main()
    