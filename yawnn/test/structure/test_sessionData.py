import unittest
from yawnnlib.structure import sessionData, timestamp
from yawnnlib.commons import commons

class TestSessionData(unittest.TestCase):
    def test_fromPath(self):
        session = sessionData.SessionData.fromPath("./yawnn/test/test_data/basic2.eimu")
        self.assertEqual(session.version, 4)
        self.assertEqual(session.sampleRate, 32)
        self.assertListEqual(session.timestamps, [timestamp.Timestamp(17, 'yawn')])
        
    def test_splitSession(self):
        session = sessionData.SessionData.fromPath("./yawnn/test/test_data/basic1.eimu")
        commons.YAWN_TIME = 2
        self.assertEqual(len(session.splitSession(sessionGap=1)), 10)
        self.assertEqual(len(session.splitSession(sessionGap=3)), 4)
        self.assertEqual(len(session.splitSession(sessionGap=9)), 2)
        
    def test_getYawnIndices(self):
        session = sessionData.SessionData.fromPath("./yawnn/test/test_data/basic2.eimu")
        self.assertEqual(len(session.getYawnIndices()), 1 * commons.YAWN_TIME * session.sampleRate)
        
    def test_getRelevantTimestamps(self):
        session = sessionData.SessionData.fromPath("./yawnn/test/test_data/basic2.eimu")
        self.assertEqual(len(session._getRelevantTimestamps(session.timestamps, 0, 16)), 0)
        self.assertEqual(len(session._getRelevantTimestamps(session.timestamps, 17, 17)), 1)
        self.assertEqual(len(session._getRelevantTimestamps(session.timestamps, 0, 32)), 1)
        self.assertEqual(len(session._getRelevantTimestamps(session.timestamps, 18, len(session.accel))), 0)
        
        session = sessionData.SessionData.fromPath("./yawnn/test/test_data/timestamps.eimu")
        self.assertEqual(len(session._getRelevantTimestamps(session.timestamps, 0, len(session.accel))), 4)
        self.assertEqual(len(session._getRelevantTimestamps(session.timestamps, 0, 51)), 2)
    
    def test_6DDataVectors(self):
        session = sessionData.SessionData.fromPath("./yawnn/test/test_data/basic1.eimu")
        vectors = session.get6DDataVectors()
        session2 = sessionData.SessionData.from6DDataVectors(
            vectors.tolist(),
            session.timestamps,
            session.sampleRate,
            session.version
        )
        self.assertListEqual(vectors[0].tolist(), session.accel[0] + session.gyro[0])
        # equality function can be found in sessionData.py
        self.assertEqual(session, session2)
        
if __name__ == "__main__":
    unittest.main()
