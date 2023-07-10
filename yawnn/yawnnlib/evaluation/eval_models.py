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
    
    def evaluate(self, isHafar : bool = False):
        modelPaths = list(filter(lambda x: x.endswith(".h5"), os.listdir(self.path)))
        for modelPath in modelPaths:
            model = test_model.loadModel(f"{self.path}/{modelPath}")
            modelType = models.MODEL_INPUTS[modelPath.split("_")[0]]
            if modelType.getType() not in self.results:
                self.results[modelType.getType()] = {}
            if isHafar:
                accuracy, prec, recall, f1 = test_model.testDataOnModel(model, modelType, config.get("HAFAR_PATH"), isHafar=True)
            else:
                accuracy, prec, recall, f1 = test_model.testDataOnModel(model, modelType, TEST_PATH)
            self.results[modelType.getType()][modelPath] = [accuracy, prec, recall, f1]
            self.save(modelPaths.index(modelPath))
            del model
            
    def save(self, ckpt : int = -1):
        if ckpt == -1:
            pickle.dump(self.results, open(f"{MODELS_PATH}/evaluation/{self.name}_results.pkl", "wb"))
        else:
            pickle.dump(self.results, open(f"{MODELS_PATH}/evaluation/{self.name}_results_{ckpt}.pkl", "wb"))

paramEval = ModelEvaluation("W_highpass") 
paramEval.evaluate(isHafar=True)
paramEval.save()
print(paramEval.results)

# plan:
# 1. change MODEL_INPUTS to be the baseline
# 2. run this
# 3. change MODEL_INPUTS to be lowpass
# 4. run this
# 5. when done run plot_paper_eval.py (note that the list [2, 1, 0, 5, 4, 3, 6] will need to be changed)
