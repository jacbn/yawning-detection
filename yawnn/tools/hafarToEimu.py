from yawnnlib.utils import commons, config
from yawnnlib.preprocessing.modelData import ModelData

import pandas as pd
import numpy as np
import os

HAFAR_FREQ = config.get("HAFAR_SAMPLE_RATE") # hz
WINDOW_SIZE = 2 # seconds

def convert(directoryPath : str, specificUsers : set[int] = set(range(1, 24)), poiUsers : set[int] = set(), poiTrainSplit : float = 0.2, isTrain : bool = True) -> commons.WeightedAnnotatedData:
    
    if not isTrain:
        # if we're using this to obtain the test data relating to the poi users, set the users to the poi users
        if len(poiUsers) == 0:
            raise ValueError("Must specify poiUsers if isTrain is False.")
        specificUsers = set(specificUsers).intersection(poiUsers)
    
    # convert the specified users to their IDs
    specificUserIDs = set(map(lambda x: f"w{x:03d}", specificUsers))
    if "w010" in specificUserIDs:
        specificUserIDs.remove("w010")
        specificUserIDs = specificUserIDs.union(set(["w010a", "w010b", "w010c"]))
    users = sorted(list(set([f.split("_")[0] for f in os.listdir(directoryPath)]).intersection(specificUserIDs)))
    
    # load data from specified users
    data = []
    timestamps = []
    weights = []
    
    for user in users:
        user_dfs = [pd.read_csv(f"{directoryPath}/{user}_{f}_eSense.csv") for f in ["acc", "gyro", "activity"]]
        # "user" is also another file type, but has no useful info for this task (and is missing for user 10)
        df = user_dfs[0].merge(user_dfs[1], on=["timestamp", "packetId", "activityId"])
        activities = user_dfs[2].drop(columns=["date", "duration", "engagement", "absorption"]).set_index("activityId").to_dict()['activity']
    
        del user_dfs
        
        if user in poiUsers:
            # then keep only poiTrainSplit*100% of the data if isTrain, else keep only the rest
            if isTrain:
                df = df[:int(len(df) * poiTrainSplit)]
            else:
                df = df[int(len(df) * poiTrainSplit):]
    
        df = df.drop(columns=["timestamp", "packetId"])
        # for each unique activity ID, get all data represented by this (all samples with the same ID are sequential) so we can split into frames
        for i in df["activityId"].unique():
            frame = df[df["activityId"] == i].drop(columns=["activityId"]).to_numpy(dtype=np.float32)
            if len(frame) < HAFAR_FREQ * WINDOW_SIZE:
                # then frame too short for a WINDOW_SIZE second sample -- ignore
                continue
            for j in range(0, len(frame) - HAFAR_FREQ * WINDOW_SIZE, HAFAR_FREQ): # windows overlap, hence the movement by only 1 second
                # todo: we currently always ignore the rightmost part of the frame. could experiment with ignoring other parts
                fixedLengthFrame = frame[j : j + HAFAR_FREQ * WINDOW_SIZE, :]
                data.append(fixedLengthFrame)
                timestamps.append(int(activities[i] == "Yawn")) # use the timestamp of the current activity
        
        weights += [round(59282 / max(400, len(df) / HAFAR_FREQ)) for _ in range(len(data) - len(weights))]
        # 59282 is the number of samples in the longest file (user 7), # todo: obtain automatically
        # we take the max of the number of samples in the file and 400 to avoid ridiculous weights for very short files
        # the division by HAFAR_FREQ obtains the number of windows in the file, used as the weighting metric
        # this is then rounded for an integer weight
        
        del df
        del activities
    
    assert len(data) == len(timestamps) == len(weights), f"{len(data)}, {len(timestamps)}, {len(weights)} not equal."
    ad, w = (np.array(data, dtype=np.float32), np.array(timestamps)), np.array(weights, dtype=np.float32)
    trainTestBorder = int(len(ad) * poiTrainSplit)
    # return (ad[:trainTestBorder], w[:trainTestBorder]), (ad[trainTestBorder:], w[trainTestBorder:])
    return ad, w

if __name__ == "__main__":
    for i in range(1, 24):
        (d, t), w = convert(config.get("HAFAR_PATH"), specificUsers=set([i]))
        # print(d.shape)
        # print(t.shape)
        print(f"{i:02d}: {len(d): 6d}, {np.sum(t): 6d}, {w[0]}")
        
    # print(convert(config.get("HAFAR_PATH"), poiUsers=set([7]), isTrain=False))