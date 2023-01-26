from scipy import signal
from yawnnlib.structure.sessionData import SessionData, Timestamp

def resample(session : SessionData, newRate : int) -> SessionData:
    """Resample the session to a new rate.

    Parameters
    ----------
    session : Session
        Session to be resampled.
    newRate : int
        New sample rate.

    Returns
    -------
    Session
        Resampled session.

    """
    # load data to session format
    data = session.get6DDataVectors()

    # resample
    data = list(map(list, signal.resample(data, session.numPoints * newRate // session.sampleRate)))
    timestamps = list(map(lambda x: Timestamp(x.time * newRate // session.sampleRate, x.type), session.timestamps))

    # return new session
    return SessionData.from6DDataVectors(data, timestamps, newRate, session.version)
    
if __name__ == "__main__":
    session = SessionData.fromPath("./yawnn/data/tests/96hz/96hz-yawns1.eimu")
    session.plot(show=False)
    newSession = resample(session, 32)
    newSession.plot(figure=2)
