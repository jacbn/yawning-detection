# Splits a SessionData object into a list of SessionData objects,
# each of the same N-second length.
N = 2

import eimuReader

def convertSession(session : eimuReader.SessionData):
    samplesPerGroup = session.sampleRate * N
    
    if session.numPoints > samplesPerGroup:
        print("Session too short to split. Returning original session.")
        return [session]
    
    converted = []
    for i in range(session.numPoints - samplesPerGroup):
        converted.append(eimuReader.SessionData(
            session.accel[i:i+samplesPerGroup], 
            session.gyro[i:i+samplesPerGroup], 
            getRelevantTimestamps(session.timestamps, i, i + samplesPerGroup),
            session.sampleRate
        ))
    return converted
        
def getRelevantTimestamps(timestamps, start, end):
    # filter only those timestamps in the range, then shift their times to match
    return list(map(lambda x: x - start, filter(lambda t: t.time >= start and t.time <= end, timestamps)))

