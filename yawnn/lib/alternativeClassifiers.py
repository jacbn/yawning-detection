from eimuReader import *
import classifiers.svm as csvm
import classifiers.knn as cknn

def trainSVM(path : str):
    # when training the SVM we prefer to use continuous data, not split into 64-sample chunks
    s = SessionData.fromPath(path)
    data, yawns = s.get6DDataVector(), s.getYawnIndices()
    
    clf = csvm.fitSVM(data, yawns)
    
    # print(clf.predict([sample])) 
    print(clf.score(data, yawns))
    print(clf.get_params())
    
def trainKNN(testPath : str, trainPath : str):
    train = SessionData.fromPath(trainPath)
    trainData, trainYawns = train.get6DDataVector(), train.getYawnIndices()
    
    test = SessionData.fromPath(testPath)
    testData, testYawns = test.get6DDataVector(), test.getYawnIndices()
    
    classification = cknn.classifyMultiple(testData, trainData, trainYawns)
    
    score(classification, testYawns)
    
def score(result : np.ndarray, ground : np.ndarray):
    print(f"Accuracy: {sum(result == ground) / len(result)}")
    print(f"Precision: {sum(result & ground) / sum(result)}")
    print(f"Recall: {sum(result & ground) / sum(ground)}")
    
if __name__ == "__main__":
    trainSVM("./yawnn/data/96hz-long2.eimu")