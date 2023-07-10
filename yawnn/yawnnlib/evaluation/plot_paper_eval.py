from yawnnlib.utils import config, commons

import pickle
import numpy as np
from matplotlib import pyplot as plt

EVAL_PATH = config.get("MODELS_PATH") + "/evaluation"
MARKERS = ['o', 's', 'D', 'o', 's', 'D', '^']

def loadResults(names : list[str]):
    return [pickle.load(open(f"{EVAL_PATH}/{name}_results.pkl", "rb")) for name in names]

def bootstrap(data : list[float], n=5000, func=np.mean):
    return np.array([func(np.random.choice(data, len(data))) for _ in range(n)])

def calcMeanPrecisionAndError(modelRepetitions : dict):
    vals = list(map(lambda x: x[1], modelRepetitions.values())) # gets the F1 score for each repetition
    return _bootstrapMeanAndError(vals)

def calcMeanRecallAndError(modelRepetitions : dict):
    vals = list(map(lambda x: x[2], modelRepetitions.values())) # gets the F1 score for each repetition
    return _bootstrapMeanAndError(vals)

def calcMeanF1AndError(modelRepetitions : dict):
    vals = list(map(lambda x: x[3], modelRepetitions.values())) # gets the F1 score for each repetition
    return _bootstrapMeanAndError(vals)

def _bootstrapMeanAndError(vals : list[float]):
    bootstrappedVals = np.sort(bootstrap(vals)) # bootstrap to obtain a larger sample size with consistent distribution properties
    mean = np.mean(bootstrappedVals)
    interval = bootstrappedVals[int(len(bootstrappedVals) * 0.2)], bootstrappedVals[int(len(bootstrappedVals) * 0.5)], bootstrappedVals[int(len(bootstrappedVals) * 0.8) + 1]
    return mean, [interval[1] - interval[0], interval[2] - interval[1]]

def calcMeanAndSD(data : list[float]):
    mean = np.mean(data)
    sd = np.std(data)
    return mean, sd

def runContinuousModelEval(results : list[dict], xVals : list[float], xlabel : str, title : str = '') -> None:
    i = 0
    for modelType in [list(results[0].keys())[i] for i in [2, 1, 0, 5, 4, 3, 6]]:
        # (reorder so model types follow order of description)
        y = []
        err = []
        for result in results:
            mean, errorWidth = calcMeanF1AndError(result[modelType])
            y.append(mean)
            err.append(errorWidth)
        plt.errorbar(xVals, y, yerr=err, fmt=MARKERS[i], markersize=6, capsize=6, label=modelType, linewidth=2, linestyle="-")
        i += 1
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel("F1 score", fontsize=16)
    plt.title(title, fontsize=24)
    plt.legend(loc='lower left', fontsize=20)
    plt.show()
    
def runDiscreteModelEval(results : list[dict], xVals : list[str], xlabel : str, title : str = '', calcStat = calcMeanF1AndError, sepGroup : bool = False) -> None:
    i = 0
    x = np.arange(len(xVals)) * 2
    width = 1/len(results)
    
    fig, ax = plt.subplots()
    ax.set_axisbelow(True)
    ax.grid(axis='y')
    
    ax.set_title(title, fontsize=72)
    # for modelType in [list(results[0].keys())[i] for i in [1, 0, 3, 2, 4]]:
    for modelType in [list(results[0].keys())[i] for i in [0, 1, 2, 3, 4]]:
        # (reorder so model types follow order of description)
        y = []
        err = []
        for result in results:
            mean, errorWidth = calcStat(result[modelType])
            y.append(mean)
            err.append(errorWidth)
        ax.bar(x + width*i, y, width=width, yerr=np.array(err).T, capsize=6, label=modelType)
        if (sepGroup):
            i += 1.5
        else:
            i += 1
    ax.set_xticks(x + width * (len(results) + 1) / 2, xVals)
    ax.set_xlabel(xlabel, labelpad=15, fontsize=52)
    labelTerms = {calcMeanPrecisionAndError: "Precision", calcMeanRecallAndError: "Recall", calcMeanF1AndError: "F1 Score"}
    ax.set_ylabel(labelTerms[calcStat], fontsize=52)
    ax.legend(loc='lower left', fontsize=32)
    plt.show()
    
def runMainEval():
    print(loadResults(["W_baseline", "W_moving_average", "W_lowpass", "W_highpass"]))
    c = 0
    for modelResult in loadResults(["W_baseline", "W_moving_average", "W_lowpass", "W_highpass"]):
        print(f'\n\n\n{["W_baseline", "W_moving_average", "W_lowpass", "W_highpass"][c]}')
        for stat in [calcMeanPrecisionAndError, calcMeanRecallAndError, calcMeanF1AndError]:
            print(f"\n{stat.__name__}:")
            for m in ['eimuLSTM', 'eimuCNN', 'fftLSTM', 'fftCNN', 'specCNN']:
                x = stat(modelResult[m])
                print(f"{x[0]:.5f} +/- {np.mean(x[1]):.5f}")
        c += 1

def plotFilterEval():
    for stat in [calcMeanPrecisionAndError, calcMeanRecallAndError, calcMeanF1AndError]:
        filterEval = loadResults(["W_baseline", "W_moving_average", "W_lowpass", "W_highpass"]) # , "W_moving_avg", "W_lowpass", "W_highpass"
        x = ["Required", "Moving Average", "Lowpass (12Hz)", "DC-Removal Highpass"]
        labelTerms = {calcMeanPrecisionAndError: "Precision", calcMeanRecallAndError: "Recall", calcMeanF1AndError: "F1 Score"}
        runDiscreteModelEval(filterEval, x, "Filter Type", f"Model {labelTerms[stat]} with Varying Filters", calcStat=stat)
    
def plotMainEval():
    results = loadResults(["W_baseline", "W_moving_average", "W_lowpass", "W_highpass"])
    for stat in [(calcMeanPrecisionAndError, 'Precision'), (calcMeanRecallAndError, 'Recall'), (calcMeanF1AndError, 'F1')]:
        runDiscreteModelEval(results, [''], "Model Type", f"Model {stat[1]}", calcStat=stat[0])
    
def plotDissEval():
    for stat in [calcMeanPrecisionAndError, calcMeanRecallAndError, calcMeanF1AndError]:
        filterEval = loadResults(["DISS"]) #loadResults(["W_baseline", "W_moving_average", "W_lowpass"]) # , "W_moving_avg", "W_lowpass", "W_highpass"
        x = [""]#, "Moving Average", "Lowpass (12Hz)"] # , "DC-Removal Highpass"
        labelTerms = {calcMeanPrecisionAndError: "Precision", calcMeanRecallAndError: "Recall", calcMeanF1AndError: "F1 Score"}
        runDiscreteModelEval(filterEval, x, "Model", f"{labelTerms[stat]} of Best Models on Dataset A", calcStat=stat, sepGroup=True)
    
    
def runResearcherDataOnlyEval(n : int):
    # to get data: train n models of each type as required (main.py), ensure each model is saved as modelType_0 to modelType_n-1, then run this function.
    from yawnnlib.evaluation import test_model
    from yawnnlib.training.models import MODEL_INPUTS
    MODELS_PATH = config.get("MODELS_PATH")
    TEST_PATH = config.get("TEST_PATH")
    for modelTypeName in [ "fftLSTM", ]: #, "eimuLSTM", "eimuCNN", "eimuCNN-LSTM",    "fftCNN", "fftCNN-LSTM", "specCNN"
        f1s = []
        for i in range(n):
            modelType = MODEL_INPUTS[modelTypeName]
            model = test_model.loadModel(f"{MODELS_PATH}/{modelTypeName}_{i}.h5")
            f1s.append(test_model.testDataOnModel(model, modelType, f"{TEST_PATH}/")[3])
        mean, sd = calcMeanAndSD(f1s)
        print(f"{modelTypeName}: {mean} +/- {sd}")
    
if __name__ == "__main__":
    # runMainEval()
    # plotFilterEval()
    # plotMainEval()
    plotDissEval()
    # runResearcherDataOnlyEval(5)
    