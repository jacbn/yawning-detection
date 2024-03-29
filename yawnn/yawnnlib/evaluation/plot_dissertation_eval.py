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

def calcMeanF1AndError(modelRepetitions : dict):
    vals = list(map(lambda x: x[3], modelRepetitions.values())) # gets the F1 score for each repetition
    bootstrappedVals = np.sort(bootstrap(vals)) # bootstrap to obtain a larger sample size with consistent distribution properties
    mean = np.mean(bootstrappedVals)
    interval = bootstrappedVals[int(len(bootstrappedVals) * 0.025)], bootstrappedVals[int(len(bootstrappedVals) * 0.975) + 1]
    return mean, interval[1] - interval[0]

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
    
def runDiscreteModelEval(results : list[dict], xVals : list[str], xlabel : str, title : str = '') -> None:
    i = 0
    x = np.arange(len(xVals)) * 2
    width = 1/len(results)
    
    fig, ax = plt.subplots()
    ax.set_axisbelow(True)
    ax.grid(axis='y')
    
    ax.set_title(title, fontsize=24)
    for modelType in [list(results[0].keys())[i] for i in [2, 1, 0, 5, 4, 3, 6]]:
        # (reorder so model types follow order of description)
        y = []
        err = []
        for result in results:
            mean, errorWidth = calcMeanF1AndError(result[modelType])
            y.append(mean)
            err.append(errorWidth)
        ax.bar(x + width*i, y, width=width, yerr=err, capsize=6, label=modelType)
        i += 1
    ax.set_xticks(x + width * (len(results) + 1) / 2, xVals)
    ax.set_xlabel(xlabel, labelpad=15, fontsize=16)
    ax.set_ylabel("F1 score", fontsize=16)
    ax.legend(loc='lower left', fontsize=20)
    plt.show()
    
def runFFTEval(results : list[dict], xVals : list[float], xlabel : str, title : str = '') -> None:
    i = 0
    for modelType in [list(results[0].keys())[i] for i in [2, 1, 0, 3]]:
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
    plt.legend(fontsize=20)
    plt.show()

def runWindowSizeEval():
    windowSizeEval = loadResults(["window_size_quarter", "window_size_third", "window_size_half", "window_size_two_thirds", "window_size_five_sixths", "baseline", "window_size_seven_sixths", "window_size_four_thirds", "window_size_nine_sixths"])
    x = [commons.YAWN_TIME/6, commons.YAWN_TIME/3, commons.YAWN_TIME/2, commons.YAWN_TIME*2/3, commons.YAWN_TIME*5/6, commons.YAWN_TIME, commons.YAWN_TIME*7/6, commons.YAWN_TIME*8/6, commons.YAWN_TIME*9/6]
    runContinuousModelEval(windowSizeEval, x, "Window Size (seconds)", "Model Performance with Varying Window Size")
    
def runWindowSepEval():
    windowSizeEval = loadResults(["window_sep_eighth", "baseline", "window_sep_three_eighths", "window_sep_half"])
    x = [1/8, 2/8, 3/8, 4/8]
    runContinuousModelEval(windowSizeEval, x, "Window Separation (seconds)", "Model Performance with Varying Window Separation")
    
def runFilterEval():
    filterEval = loadResults(["baseline", "filters_smooth", "filters_moving_avg", "filters_smooth_highpass", "filters_lowpass"])
    x = ["Required", "Smooth", "Moving Average", "DC-Removal Highpass", "Lowpass (12Hz)"]
    runDiscreteModelEval(filterEval, x, "Filter Type", "Model Performance with Varying Filters")
    
def runFftNPerSegEval():
    fftEval = loadResults(["fft_128", "fft_256", "fft_384"])
    x = [128., 256., 384.]
    runFFTEval(fftEval, x, "Samples Per FFT Segment", "Model Performance with Varying FFT Parameters")
    
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
    # runWindowSizeEval()
    # runWindowSepEval()
    # runFilterEval()
    runFftNPerSegEval()
    # runResearcherDataOnlyEval(5)
    