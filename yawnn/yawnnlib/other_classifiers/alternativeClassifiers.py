from yawnn.yawnnlib.structure.sessionData import SessionData
from yawnnlib.commons import commons
import numpy as np
import classifiers.svm_sk as csvm
import classifiers.knn as cknn
import classifiers.knn_scipy as cknnscipy

def trainSVM(path : str):
    # when training the SVM we prefer to use continuous data, not split into 64-sample chunks
    s = SessionData.fromPath(path)
    data, yawns = s.get6DDataVector(), s.getYawnIndices()
    
    clf = csvm.fitSVM(data, yawns)
    
    # print(clf.predict([sample])) 
    print(clf.score(data, yawns))
    print(clf.get_params())
    
def trainKNN(dataPath : str):
    sessions = commons.mapToDirectory(SessionData.fromPath, dataPath)
    data = np.concatenate([x.get6DDataVector() for x in sessions])
    yawns = np.concatenate([x.getYawnIndices() for x in sessions])
    
    # train = SessionData.fromPath(trainPaths[0])
    # trainData = train.get6DDataVector()
    # trainYawns = train.getYawnIndices()
    
    # for path in trainPaths[1:]:
    #     train = SessionData.fromPath(path)
    #     trainData = np.concatenate((trainData, train.get6DDataVector()))
    #     trainYawns = np.concatenate((trainYawns, train.getYawnIndices()))
    
    indices = np.random.permutation(len(data))
    data = data[indices]
    yawns = yawns[indices]
    
    trainAmount = int(commons.TRAIN_SPLIT * len(data))
    
    classification = cknn.classifyMultiple(data[trainAmount:], data[:trainAmount], yawns[:trainAmount])
    # classification = cknnscipy.classifyMultiple(data[trainAmount:], data[:trainAmount], yawns[:trainAmount])
    
    score(classification, yawns[trainAmount:])
    
def score(result : np.ndarray, ground : np.ndarray):
    print(f"Accuracy: {sum(result == ground) / len(result)}")
    print(f"Precision: {sum(result.astype(int) & ground.astype(int)) / sum(result)}")
    print(f"Recall: {sum(result.astype(int) & ground.astype(int)) / sum(ground)}")
    print(f"F1: {2 * sum(result.astype(int) & ground.astype(int)) / (sum(result) + sum(ground))}")
    
if __name__ == "__main__":
    # trainSVM("./yawnn/data/96hz-long2.eimu")
    trainKNN("./yawnn/data/user-trials/")