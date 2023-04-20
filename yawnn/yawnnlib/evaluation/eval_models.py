from yawnnlib.evaluation import metrics, test_model
from yawnnlib.utils import commons, filters, config
from yawnnlib.preprocessing import modelInput
from yawnnlib.training import models

import os
import pickle

MODELS_PATH = config.get("MODELS_PATH")
TEST_PATH = config.get("TEST_PATH")

# for each parameter, assign a list of all tests done on that parameter

class ModelEvaluation:
    def __init__(self, name):
        self.name = name
        self.path = f"{MODELS_PATH}/{name}"
        self.results = {}
    
    def evaluate(self):
        modelPaths = list(filter(lambda x: x.endswith(".h5"), os.listdir(self.path)))
        for modelPath in modelPaths:
            model = test_model.loadModel(f"{self.path}/{modelPath}")
            modelType = models.MODEL_INPUTS[modelPath.split("_")[0]]
            if modelType.getType() not in self.results:
                self.results[modelType.getType()] = {}
            accuracy, prec, recall, f1 = test_model.testDataOnModel(model, modelType, TEST_PATH)
            self.results[modelType.getType()][modelPath] = [accuracy, prec, recall, f1]
            del model
            
    def save(self):
        pickle.dump(self.results, open(f"{MODELS_PATH}/evaluation/{self.name}_results.pkl", "wb"))

paramEval = ModelEvaluation("fft_128") 
paramEval.evaluate()
paramEval.save()
print(paramEval.results)
