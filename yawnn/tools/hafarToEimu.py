from yawnnlib.utils import commons, config

import pandas as pd
import numpy as np
import os

HAFAR_FREQ = 32 # hz
WINDOW_SIZE = 2 # seconds

def convert(directoryPath : str, specificUsers : set[int] = set(range(1, 24))) -> commons.AnnotatedData:
    
    # convert the specified users to their IDs
    specificUserIDs = set(map(lambda x: f"w{x:03d}", specificUsers))
    if "w010" in specificUserIDs:
        specificUserIDs.remove("w010")
        specificUserIDs = specificUserIDs.union(set(["w010a", "w010b", "w010c"]))
    users = sorted(list(set([f.split("_")[0] for f in os.listdir(directoryPath)]).intersection(specificUserIDs)))
    
    # load data from specified users
    data = []
    timestamps = []
    for user in users:
        user_dfs = [pd.read_csv(f"{directoryPath}/{user}_{f}_eSense.csv") for f in ["acc", "gyro", "activity"]]
        # "user" is also another file type, but has no useful info for this task (and is missing for user 10)
        df = user_dfs[0].merge(user_dfs[1], on=["timestamp", "packetId", "activityId"])
        activities = user_dfs[2].drop(columns=["date", "duration", "engagement", "absorption"]).set_index("activityId").to_dict()['activity']
    
        del user_dfs
    
        df = df.drop(columns=["timestamp", "packetId"])
        # for each unique activity ID, get all data represented by this (all samples with the same ID are sequential) so we can split into frames
        for i in df["activityId"].unique():
            frame = df[df["activityId"] == i].drop(columns=["activityId"]).to_numpy(dtype=np.float32)
            if len(frame) < HAFAR_FREQ * WINDOW_SIZE:
                # then frame too short for a WINDOW_SIZE second sample -- ignore
                continue
            for j in range(0, len(frame) - HAFAR_FREQ * WINDOW_SIZE, HAFAR_FREQ):
                # todo: we currently always ignore the rightmost part of the frame. could experiment with ignoring the leftmost part instead
                fixedLengthFrame = frame[j : j + HAFAR_FREQ * WINDOW_SIZE, :]
                data.append(fixedLengthFrame)
                timestamps.append(int(activities[i] == "Yawn")) # use the timestamp of the current activity
            
        del df
        del activities
    
    assert len(data) == len(timestamps)
    return np.array(data, dtype=np.float32), np.array(timestamps)

if __name__ == "__main__":
    for i in range(1, 24):
        d, t = convert(config.get("HAFAR_PATH"), specificUsers=set([i]))
        # print(d.shape)
        # print(t.shape)
        print(f"{i}: {np.sum(t)}")