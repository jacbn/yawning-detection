from eimuReader import *
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
    
def trainKNN(trainPaths : list, testPath : str):
    train = SessionData.fromPath(trainPaths[0])
    trainData = train.get6DDataVector()
    trainYawns = train.getYawnIndices()
    
    for path in trainPaths[1:]:
        train = SessionData.fromPath(path)
        trainData = np.concatenate((trainData, train.get6DDataVector()))
        trainYawns = np.concatenate((trainYawns, train.getYawnIndices()))
    
    test = SessionData.fromPath(testPath)
    testData, testYawns = test.get6DDataVector(), test.getYawnIndices()
    
    classification = cknn.classifyMultiple(testData, trainData, trainYawns)
    # classification = cknnscipy.classifyMultiple(testData, trainData, trainYawns)
    
    score(classification, testYawns)
    
def score(result : np.ndarray, ground : np.ndarray):
    print(f"Accuracy: {sum(result == ground) / len(result)}")
    print(f"Precision: {sum(result.astype(int) & ground.astype(int)) / sum(result)}")
    print(f"Recall: {sum(result.astype(int) & ground.astype(int)) / sum(ground)}")
    print(f"F1: {2 * sum(result.astype(int) & ground.astype(int)) / (sum(result) + sum(ground))}")
    
if __name__ == "__main__":
    # trainSVM("./yawnn/data/96hz-long2.eimu")
    trainKNN(["./yawnn/data/96hz-yawns1.eimu", "./yawnn/data/96hz-walk1.eimu"], "./yawnn/data/96hz-long2.eimu")