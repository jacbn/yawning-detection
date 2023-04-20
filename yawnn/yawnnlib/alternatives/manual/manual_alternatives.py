from yawnnlib.structure.sessionData import SessionData
from yawnnlib.preprocessing.eimuModelInput import EimuModelInput
from yawnnlib.utils import commons

import numpy as np
import svm_sk as csvm
import knn as cknn
import knn_scipy as cknnscipy

def trainSVM(path : str):
    # when training the SVM we prefer to use continuous data, not split into 64-sample windows
    s = SessionData.fromPath(path)
    data, yawns = s.get6DDataVectors(), s.getYawnIndices()
    
    clf = csvm.fitSVM(data, yawns)
    
    # print(clf.predict([sample])) 
    print(clf.score(data, yawns))
    print(clf.get_params())
    
def trainKNN(dataPath : str, useScipy : bool = True):
    modelType = EimuModelInput(windowSize=commons.YAWN_TIME*1.5, windowSep=commons.YAWN_TIME/2)
    annotatedData = modelType.fromDirectory(dataPath)
    (trainX, trainY), (testX, testY) = modelType.fromAnnotatedDataList(
        annotatedData,
        shuffle=True,
        equalPositiveAndNegative=True,
        trainSplit=0.8
    )
    trainY = np.repeat(trainY, trainX.shape[1])
    testY = np.repeat(testY, testX.shape[1])
    trainX = trainX.reshape(-1, 6)
    testX = testX.reshape(-1, 6)
    
    if useScipy:
        classification = cknnscipy.classifyMultiple(testX, trainX, trainY)
    else:
        classification = cknn.classifyMultiple(testX, trainX, trainY)
    
    score(classification, testY)
    
def score(result : np.ndarray, ground : np.ndarray):
    print(f"Accuracy: {sum(result == ground) / len(result)}")
    print(f"Precision: {sum(result.astype(int) & ground.astype(int)) / sum(result)}")
    print(f"Recall: {sum(result.astype(int) & ground.astype(int)) / sum(ground)}")
    print(f"F1: {2 * sum(result.astype(int) & ground.astype(int)) / (sum(result) + sum(ground))}")
    
if __name__ == "__main__":
    # trainSVM(f"{utils.PROJECT_ROOT}/data/96hz-long2.eimu")
    trainKNN(f"{commons.PROJECT_ROOT}/data/user_trials/")