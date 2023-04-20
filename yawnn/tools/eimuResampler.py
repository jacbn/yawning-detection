import numpy as np
from scipy import signal
from yawnnlib.utils import commons
from yawnnlib.structure.sessionData import SessionData, Timestamp

def _resample(data : np.ndarray, timestamps : list, timestampsAreObjs : bool, oldRate : int, newRate : int, numPoints : int) -> commons.AnnotatedData:
    """ Internal resampling function.

    Parameters
    ----------
    data : np.ndarray
        The data to be resampled.
    timestamps : list[commons.Timestamp] | list[int]
        The timestamps in the data. These can be either Timestamp objects (e.g. from a SessionData) or ints (e.g. from an AnnotatedData)
    timestampsAreObjs : bool
        Differentiate between the two types of timestamps.
    oldRate : int
        The old sample rate.
    newRate : int
        The new sample rate.
    numPoints : int
        The number of points to generate at the new sample rate.

    Returns
    -------
    commons.AnnotatedData
        The resampled data and timestamps.
    """
    newData = list(map(list, signal.resample(data.tolist(), numPoints * newRate // oldRate)))
    if timestampsAreObjs:
        newTimestamps = np.array(map(lambda x: Timestamp(x.time * newRate // oldRate, x.type), timestamps))
    else:
        newTimestamps = []
        for i in range(len(newData)):
            newTimestamps.append(timestamps[i * oldRate // newRate])
        newTimestamps = np.array(newTimestamps).reshape(-1, 1)

    return np.array(newData), newTimestamps

def resampleSession(session : SessionData, newRate : int) -> SessionData:
    """Resample the session to a new rate.

    Parameters
    ----------
    session : SessionData
        Session to be resampled.
    newRate : int
        New sample rate.

    Returns
    -------
    SessionData
        Resampled session.

    """
    data, timestamps = _resample(session.get6DDataVectors(), session.timestamps, True, session.sampleRate, newRate, session.numPoints)
    # return new session
    return SessionData.from6DDataVectors(data.tolist(), timestamps.tolist(), newRate, session.version)

def resampleAnnotatedData(annotatedData : commons.AnnotatedData, oldRate : int, newRate : int) -> commons.AnnotatedData:
    data, annotations = annotatedData
    if len(data) == 0:
        return data, annotations
    return _resample(data, annotations.squeeze().tolist(), False, oldRate, newRate, data.shape[0])

if __name__ == "__main__":
    session = SessionData.fromPath(f"{commons.PROJECT_ROOT}/data/tests/96hz/96hz-yawns1.eimu")
    session.plot(show=False)
    newSession = resampleSession(session, 32)
    newSession.plot(figure=2)
