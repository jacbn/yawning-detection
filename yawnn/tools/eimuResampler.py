import numpy as np
from scipy import signal
from yawnnlib.utils import commons, config
from yawnnlib.structure.sessionData import SessionData, Timestamp

def _resample(data : np.ndarray, timestamps : list, timestampsAreObjs : bool, oldRate : int, newRate : int, axis : int = 1) -> commons.AnnotatedData:
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
    axis : int, optional
        The axis to resample, i.e. the axis representing the frequency domain. By default 1.

    Returns
    -------
    commons.AnnotatedData
        The resampled data and timestamps.
    """
    newData = list(map(list, signal.resample(data.tolist(), int(data.shape[axis] * newRate / oldRate), axis=axis)))
    if timestampsAreObjs:
        newTimestamps = np.array(map(lambda x: Timestamp(int(x.time * newRate / oldRate), x.type), timestamps))
    else:
        newTimestamps = []
        for i in range(len(newData)):
            newTimestamps.append(timestamps[int(i * newRate / oldRate)])
        newTimestamps = np.array(newTimestamps).reshape(-1, 1)

    return np.array(newData), newTimestamps

def resampleSession(session : SessionData, windowSize : float, newRate : int) -> SessionData:
    """Resample the session to a new rate.

    Parameters
    ----------
    session : SessionData
        Session to be resampled.
    windowSize : float
        Window size of the original session in seconds.
    newRate : int
        New sample rate.

    Returns
    -------
    SessionData
        Resampled session.

    """
    data, timestamps = _resample(session.get6DDataVectors(), session.timestamps, True, session.sampleRate, newRate, axis=1)
    # return new session
    return SessionData.from6DDataVectors(data.tolist(), timestamps.tolist(), newRate, session.version)

def resampleAnnotatedData(annotatedData : commons.AnnotatedData, oldRate : int, newRate : int, axis : int = 1) -> commons.AnnotatedData:
    print(f"old: {annotatedData[0].shape} {annotatedData[1].shape}")
    data, annotations = annotatedData
    if len(data) == 0:
        # ideally we would just return (data, annotations), but we still need to reshape the axes that are not 0-width else tf sees a shape mismatch
        shape = list(map(float, list(data.shape)))
        shape[axis] *= newRate / oldRate
        data = np.reshape(data, list(map(int, shape)))
        print(f"new: {data.shape} {annotations.shape}")
        return data, annotations
    final = _resample(data, annotations.squeeze().tolist(), False, oldRate, newRate, axis=axis)
    print(f"new: {final[0].shape} {final[1].shape}")
    return final

if __name__ == "__main__":
    session = SessionData.fromPath(f"{commons.PROJECT_ROOT}/data/tests/96hz/96hz-yawns1.eimu")
    session.plot(show=False)
    newSession = resampleSession(session, 64, 32)
    newSession.plot(figure=2)
