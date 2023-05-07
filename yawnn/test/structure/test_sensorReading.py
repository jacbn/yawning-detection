import unittest
from yawnnlib.structure import sensorReading

class TestSessionData(unittest.TestCase):
    def test_fromString(self):
        self.assertEqual(sensorReading.SensorReading.fromString("[1, 2, 3] [4, 5, 6]"), sensorReading.SensorReading([1, 2, 3], [4, 5, 6]))
        self.assertEqual(sensorReading.SensorReading.fromString("[-1, 0, 10420] [-12402, 65535, -65536]"), sensorReading.SensorReading([-1, 0, 10420], [-12402, 65535, -65536]))
        self.assertRaises(ValueError, lambda: sensorReading.SensorReading.fromString("[a, b, c] [d, e, f]"))
        
if __name__ == "__main__": # pragma: no cover
    unittest.main()
    