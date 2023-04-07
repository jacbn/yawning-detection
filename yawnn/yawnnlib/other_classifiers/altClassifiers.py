from sklearn.preprocessing import StandardScaler
from sklearn import pipeline

class ClassifierNotTrainedException(Exception):
    def __init__(self, msg : str = "The classifier has not yet been trained."):
        super().__init__(msg)

class AlternativeClassifier():
    def __init__(self, name : str, classifier):
        self.name = name
        self.classifier = classifier
        self.clf = None
    
    def getModel(self):
        return pipeline.make_pipeline(StandardScaler(), self.classifier)
    
    def fit(self, trainX, trainY):
        self.clf = self.getModel()
        self.clf.fit(trainX, trainY)

    def isTrained(self):
        return self.clf is not None
    
    def getCLF(self) -> pipeline.Pipeline:
        if self.clf is None:
            raise ClassifierNotTrainedException()
        return self.clf
    